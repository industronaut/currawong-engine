//! Engine-driven render-object pass.
//!
//! Owns the per-frame walk: sim → [`RenderId`] → [`RenderTemplate`] →
//! declared proxy → frustum cull with hysteresis → fan-out per node.
//! View code supplies a per-mesh-node callback (and optionally a
//! per-emitter-node callback) that does the actual draw-attrib push; the
//! engine owns the traversal.
//!
//! Visibility is **view-side state** on [`RenderProxy`], set by the
//! template's update closure via [`RenderObjectTraversal::update_instances`]
//! and read by the extract walk. `root_visible` gates the whole instance
//! (no callbacks, no hit-ID reservation); per-node `nodes[slot].visible`
//! gates individual nodes. The engine cascades these down the template's
//! tree after the user closure runs, writing
//! [`NodeState::effective_visible`](super::render_proxy::NodeState::effective_visible)
//! into each node's state so extract reads a single composed flag. Defaults
//! are all `true` — templates and instances without opinions on visibility
//! are unaffected.
//!
//! All sim→view translation happens in the per-instance update closure,
//! which reads [`Components`] and mutates [`RenderProxy`]. Extract
//! callbacks see only the persistent [`RenderProxy`] (plus the
//! template's per-node payload and the node's composed world transform) —
//! they never touch the sim directly. CLAUDE.md invariant: extract is a
//! pure GPU-attrib write step.

use std::hash::Hash;

use glam::Mat4;

use crate::sim::{Components, WorldObjectRef, Zones};

use super::render_object::{EmitterPart, MeshPart, NodeKind, RenderRegistry, RenderTemplate};
use super::render_proxy::{RenderProxies, RenderProxy};
use super::renderer::Renderer;
use super::visibility::Frustum;

/// Engine helper that drives the per-frame render-object walk. Stateless;
/// the associated functions take all dependencies as arguments.
pub struct RenderObjectTraversal;

impl RenderObjectTraversal {
    /// Phase 1: walk `zones` and declare a live proxy on `proxies` for
    /// each object whose components include an `R` render id, then cull
    /// against `frustum`. Calls [`RenderProxies::begin_frame`] for you.
    ///
    /// Templates with no `visual_bounds` produce proxies with no AABB,
    /// which [`RenderProxies::cull`] treats as always-visible — matching
    /// CLAUDE.md's "templates without bounds are never culled" rule.
    pub fn declare_and_cull<R, M, MK, E, S>(
        zones: &Zones,
        templates: &RenderRegistry<R, M, MK, E, S>,
        proxies: &mut RenderProxies<R>,
        frustum: &Frustum,
    ) where
        R: Clone + Eq + Hash + 'static,
    {
        proxies.begin_frame();
        for (zone_id, zone) in zones.iter() {
            for (id, obj) in zone.iter() {
                let Some(rid) = zone.components().get::<R>(id) else {
                    continue;
                };
                let Some(template) = templates.get(rid) else {
                    continue;
                };
                // sim→view seam: convert integer SimPos / Facing to glam.
                let object_xform =
                    Mat4::from_rotation_translation(obj.facing.to_quat(), obj.position.to_vec3());
                let world_aabb = template
                    .visual_bounds()
                    .map(|local| local.transformed(object_xform));
                proxies.declare(
                    WorldObjectRef { zone: zone_id, id },
                    rid.clone(),
                    object_xform,
                    world_aabb,
                    template.node_count(),
                );
            }
        }
        proxies.cull(frustum);
    }

    /// Phase 1.5: per-instance update. Iterate alive proxies and invoke
    /// `on_instance` so the user can read [`Components`] and mutate the
    /// view-side [`RenderProxy`] — typically `instance.root_visible`,
    /// `instance.node_mut(slot).visible`, plus any future cached
    /// view-state on the instance.
    ///
    /// After the closure returns for a proxy, the engine walks its
    /// template tree in topological order and writes each node's
    /// composed `world_xform` and cascaded `effective_visible` into
    /// `instance.nodes[slot]` — so by the time
    /// [`Self::for_each_alive_part`] runs, every node has the values
    /// extract needs and visibility has cascaded down the tree.
    ///
    /// Call this between [`Self::declare_and_cull`] and
    /// [`Self::for_each_alive_part`] / [`Self::for_each_alive`]. The
    /// extract pass reads only `RenderProxy` state, so all sim →
    /// view-state translation must land here.
    ///
    /// CLAUDE.md invariants this enforces:
    /// - Sim → view translation lives in **one** place per frame.
    /// - Extract is a pure GPU-attrib write step.
    pub fn update_instances<R, M, MK, E, S>(
        zones: &Zones,
        templates: &RenderRegistry<R, M, MK, E, S>,
        proxies: &mut RenderProxies<R>,
        mut on_instance: impl FnMut(
            WorldObjectRef,
            &R,
            &Components,
            &RenderTemplate<M, MK, E, S>,
            &mut RenderProxy,
        ),
    ) where
        R: Clone + Eq + Hash + 'static,
    {
        for (parent, rid, obj) in proxies.iter_mut() {
            let Some(template) = templates.get(rid) else {
                continue;
            };
            let zone = zones
                .get(parent.zone)
                .expect("zone alive while proxy lives");
            on_instance(parent, rid, zone.components(), template, obj);
            compose_node_state(template, obj);
        }
    }

    /// Phase 2 (mesh-only): iterate alive proxies and invoke `on_part`
    /// for every [`NodeKind::Mesh`] node, passing the node's already-composed
    /// world transform.
    ///
    /// Thin wrapper over [`Self::for_each_alive`] with a no-op emitter
    /// callback; the traversal lives in one place.
    pub fn for_each_alive_part<R, M, MK, E, S>(
        zones: &Zones,
        templates: &RenderRegistry<R, M, MK, E, S>,
        proxies: &RenderProxies<R>,
        on_part: impl FnMut(WorldObjectRef, &R, &MeshPart<M, MK>, Mat4),
    ) where
        R: Clone + Eq + Hash + 'static,
    {
        Self::for_each_alive(zones, templates, proxies, on_part, |_, _, _, _| {});
    }

    /// Phase 2 (mesh + emitter): iterate alive proxies and invoke
    /// `on_part` for every [`NodeKind::Mesh`] node and `on_emitter` for
    /// every [`NodeKind::Emitter`] node, each receiving the node's
    /// composed world transform.
    ///
    /// Visibility is **view-side state** read from the
    /// [`RenderProxy`]: instances with `root_visible == false` are
    /// skipped entirely, and any node whose `effective_visible` is false
    /// (set or hidden by the cascade in [`Self::update_instances`]) is
    /// skipped individually. These fields are set by the user's update
    /// closure passed to [`Self::update_instances`] (or by visibility
    /// cascading down from a hidden ancestor).
    ///
    /// Use this when a template carries emitter nodes (see
    /// `examples/campfire.rs`). Pure-mesh views can call
    /// [`Self::for_each_alive_part`] for the same walk without the empty
    /// emitter closure.
    pub fn for_each_alive<R, M, MK, E, S>(
        zones: &Zones,
        templates: &RenderRegistry<R, M, MK, E, S>,
        proxies: &RenderProxies<R>,
        mut on_part: impl FnMut(WorldObjectRef, &R, &MeshPart<M, MK>, Mat4),
        mut on_emitter: impl FnMut(WorldObjectRef, &R, &EmitterPart<E, S>, Mat4),
    ) where
        R: Clone + Eq + Hash + 'static,
    {
        let _ = zones;
        for (parent, rid, object) in proxies.iter() {
            if !object.root_visible {
                continue;
            }
            let Some(template) = templates.get(rid) else {
                continue;
            };

            for (slot, node) in template.nodes().iter().enumerate() {
                let state = &object.nodes[slot];
                if !state.effective_visible {
                    continue;
                }
                match &node.kind {
                    NodeKind::Empty => {}
                    NodeKind::Mesh(part) => on_part(parent, rid, part, state.world_xform),
                    NodeKind::Emitter(part) => on_emitter(parent, rid, part, state.world_xform),
                }
            }
        }
    }

    /// Phase 2 (mesh + emitter), with GPU hit-ID reservation (#56 PR 3).
    ///
    /// Same shape as [`Self::for_each_alive`], with one extra step: per
    /// alive parent, reserve a single hit ID via
    /// [`Renderer::reserve_object`] and pass it to **both** the mesh and
    /// emitter callbacks. All mesh nodes of one sim object share the same
    /// hit ID, so a cursor over any of them resolves back to the same
    /// `WorldObjectId` via the engine's hit-ID readback. Emitters (which
    /// don't write to the hit-ID attachment in v1) receive the same ID
    /// for symmetry — view code typically ignores it.
    ///
    /// The view's per-instance attribute writer (typically
    /// [`MeshInstanceAttribs::with_hit_id`](super::MeshInstanceAttribs::with_hit_id))
    /// is responsible for stamping the ID onto the draw — this helper
    /// just allocates and threads it.
    pub fn for_each_alive_with_hit_id<R, M, MK, E, S>(
        zones: &Zones,
        templates: &RenderRegistry<R, M, MK, E, S>,
        proxies: &RenderProxies<R>,
        renderer: &Renderer,
        on_part: impl FnMut(WorldObjectRef, &R, &MeshPart<M, MK>, Mat4, u32),
        on_emitter: impl FnMut(WorldObjectRef, &R, &EmitterPart<E, S>, Mat4, u32),
    ) where
        R: Clone + Eq + Hash + 'static,
    {
        for_each_alive_reserving(
            zones,
            templates,
            proxies,
            |parent| renderer.reserve_object(parent.zone, parent.id),
            on_part,
            on_emitter,
        );
    }

    /// Mesh-only variant of [`Self::for_each_alive_with_hit_id`] — the
    /// convenient default for templates that carry no emitter nodes.
    pub fn for_each_alive_part_with_hit_id<R, M, MK, E, S>(
        zones: &Zones,
        templates: &RenderRegistry<R, M, MK, E, S>,
        proxies: &RenderProxies<R>,
        renderer: &Renderer,
        on_part: impl FnMut(WorldObjectRef, &R, &MeshPart<M, MK>, Mat4, u32),
    ) where
        R: Clone + Eq + Hash + 'static,
    {
        Self::for_each_alive_with_hit_id(
            zones,
            templates,
            proxies,
            renderer,
            on_part,
            |_, _, _, _, _| {},
        );
    }
}

/// Walk `template`'s node tree in topological order (slots are dense and
/// added parent-first by [`RenderTemplate::with_node`], so a single
/// linear pass is sufficient) and fill each node's composed
/// `world_xform` + cascaded `effective_visible` into `proxy.nodes[slot]`.
///
/// Runs once per proxy from inside [`RenderObjectTraversal::update_instances`],
/// after the user's closure has had a chance to flip `visible` flags. No
/// allocation; the dense vec is in-place.
fn compose_node_state<M, MK, E, S>(
    template: &RenderTemplate<M, MK, E, S>,
    proxy: &mut RenderProxy,
) {
    debug_assert_eq!(
        proxy.nodes.len(),
        template.node_count(),
        "proxy node count out of sync with template `{}`",
        template.label,
    );
    for slot in 0..proxy.nodes.len() {
        let node = &template.nodes()[slot];
        // Inherit from parent first; parent is guaranteed to have been
        // composed already by the topological ordering invariant.
        let (parent_world, parent_visible) = match node.parent {
            None => (proxy.world_xform, proxy.root_visible),
            Some(pid) => {
                let pslot = template
                    .slot_of(pid)
                    .expect("parent slot present (RenderTemplate enforces parent-first)")
                    as usize;
                let p = &proxy.nodes[pslot];
                (p.world_xform, p.effective_visible)
            }
        };
        let state = &mut proxy.nodes[slot];
        state.world_xform = parent_world * node.local_transform;
        state.effective_visible = parent_visible && state.visible;
    }
}

/// Inner traversal shared by [`RenderObjectTraversal::for_each_alive_with_hit_id`]
/// and its tests. The `reserve` closure is called once per *visible* parent
/// — invisible parents short-circuit before reservation, so the hit-ID
/// counter doesn't advance for objects the user can't click.
fn for_each_alive_reserving<R, M, MK, E, S>(
    zones: &Zones,
    templates: &RenderRegistry<R, M, MK, E, S>,
    proxies: &RenderProxies<R>,
    mut reserve: impl FnMut(WorldObjectRef) -> u32,
    mut on_part: impl FnMut(WorldObjectRef, &R, &MeshPart<M, MK>, Mat4, u32),
    mut on_emitter: impl FnMut(WorldObjectRef, &R, &EmitterPart<E, S>, Mat4, u32),
) where
    R: Clone + Eq + Hash + 'static,
{
    let _ = zones;
    for (parent, rid, object) in proxies.iter() {
        if !object.root_visible {
            continue;
        }
        let Some(template) = templates.get(rid) else {
            continue;
        };

        let hit_id = reserve(parent);

        for (slot, node) in template.nodes().iter().enumerate() {
            let state = &object.nodes[slot];
            if !state.effective_visible {
                continue;
            }
            match &node.kind {
                NodeKind::Empty => {}
                NodeKind::Mesh(part) => on_part(parent, rid, part, state.world_xform, hit_id),
                NodeKind::Emitter(part) => on_emitter(parent, rid, part, state.world_xform, hit_id),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::render_object::{NodeId, RenderTemplate, TemplateNode};
    use crate::render::render_proxy::NodeState;
    use crate::render::visibility::Aabb;
    use crate::sim::{Facing, SimPos, WorldTransform, Zone, ZoneId, Zones};
    use glam::{Vec3, Vec4};

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Rid {
        Tree,
    }

    /// Sim-side appearance component the tests stand in for the kind of
    /// per-object state that the update hook is expected to read.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Appearance {
        height: f32,
    }

    fn always_inside() -> Frustum {
        // Same trick as render_proxy tests: every plane accepts.
        Frustum {
            planes: [Vec4::new(0.0, 0.0, 0.0, 1.0); 6],
        }
    }

    fn always_outside() -> Frustum {
        Frustum {
            planes: [Vec4::new(0.0, 0.0, 0.0, -1.0); 6],
        }
    }

    fn seed_zone() -> (Zones, ZoneId) {
        let mut zones = Zones::new();
        let zid = zones.insert(Zone::new());
        (zones, zid)
    }

    fn insert_tree(zones: &mut Zones, zid: ZoneId, position: Vec3, appearance: Option<Appearance>) {
        let zone = zones.get_mut(zid).expect("zone");
        let id = zone.insert(WorldTransform {
            position: SimPos::from_vec3(position),
            facing: Facing::ZERO,
        });
        zone.components_mut().insert(id, Rid::Tree);
        if let Some(a) = appearance {
            zone.components_mut().insert(id, a);
        }
    }

    fn tree_template() -> RenderTemplate {
        RenderTemplate::new("tree").with_visual_bounds(Aabb::new(
            Vec3::new(-0.5, -0.5, 0.0),
            Vec3::new(0.5, 0.5, 2.0),
        ))
    }

    #[test]
    fn declare_and_cull_creates_proxy_per_render_id() {
        let (mut zones, zid) = seed_zone();
        insert_tree(&mut zones, zid, Vec3::ZERO, None);
        insert_tree(&mut zones, zid, Vec3::new(2.0, 0.0, 0.0), None);

        let mut templates: RenderRegistry<Rid> = RenderRegistry::new();
        templates.register(Rid::Tree, tree_template());

        let mut live = RenderProxies::<Rid>::new(30);
        RenderObjectTraversal::declare_and_cull(&zones, &templates, &mut live, &always_inside());

        assert_eq!(live.len(), 2);
    }

    #[test]
    fn declare_and_cull_skips_objects_without_render_id() {
        let (mut zones, zid) = seed_zone();
        // Object with no RenderId component.
        let zone = zones.get_mut(zid).unwrap();
        zone.insert(WorldTransform::default());

        let mut templates: RenderRegistry<Rid> = RenderRegistry::new();
        templates.register(Rid::Tree, tree_template());

        let mut live = RenderProxies::<Rid>::new(30);
        RenderObjectTraversal::declare_and_cull(&zones, &templates, &mut live, &always_inside());

        assert!(live.is_empty());
    }

    #[test]
    fn declare_and_cull_drops_offscreen_first_time() {
        let (mut zones, zid) = seed_zone();
        insert_tree(&mut zones, zid, Vec3::ZERO, None);

        let mut templates: RenderRegistry<Rid> = RenderRegistry::new();
        templates.register(Rid::Tree, tree_template());

        let mut live = RenderProxies::<Rid>::new(30);
        RenderObjectTraversal::declare_and_cull(&zones, &templates, &mut live, &always_outside());

        // New proxy, never visible → dropped immediately.
        assert!(live.is_empty());
    }

    #[test]
    fn for_each_alive_part_visits_each_mesh_node() {
        let (mut zones, zid) = seed_zone();
        insert_tree(
            &mut zones,
            zid,
            Vec3::ZERO,
            Some(Appearance { height: 1.7 }),
        );

        let mut templates: RenderRegistry<Rid, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32>::new("tree")
                .with_mesh_part(0, 0, Mat4::IDENTITY)
                .with_mesh_part(1, 1, Mat4::from_translation(Vec3::Z)),
        );

        let mut live = RenderProxies::<Rid>::new(30);
        RenderObjectTraversal::declare_and_cull(&zones, &templates, &mut live, &always_inside());
        // No-op update — the engine still needs to run compose_node_state
        // so per-node world transforms are filled in.
        RenderObjectTraversal::update_instances(&zones, &templates, &mut live, |_, _, _, _, _| {});

        let mut visits = 0usize;
        RenderObjectTraversal::for_each_alive_part(
            &zones,
            &templates,
            &live,
            |_parent, _rid, part, _world| {
                visits += 1;
                let _ = part.mesh;
            },
        );

        assert_eq!(visits, 2, "two mesh nodes → two callback invocations");
    }

    #[test]
    fn update_reads_components_and_writes_view_state() {
        let (mut zones, zid) = seed_zone();
        // First tree carries an Appearance; second has none.
        insert_tree(
            &mut zones,
            zid,
            Vec3::ZERO,
            Some(Appearance { height: 2.0 }),
        );
        insert_tree(&mut zones, zid, Vec3::new(2.0, 0.0, 0.0), None);

        let mut templates: RenderRegistry<Rid, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32>::new("tree").with_mesh_part(0, 0, Mat4::IDENTITY),
        );

        let mut live = RenderProxies::<Rid>::new(30);
        RenderObjectTraversal::declare_and_cull(&zones, &templates, &mut live, &always_inside());

        // Hide any tree without an Appearance — exercises the fallback path
        // (component missing) and the present path together.
        RenderObjectTraversal::update_instances(
            &zones,
            &templates,
            &mut live,
            |parent, _rid, components, _template, instance| {
                if components.get::<Appearance>(parent.id).is_none() {
                    instance.root_visible = false;
                }
            },
        );

        let mut visits = 0usize;
        RenderObjectTraversal::for_each_alive_part(&zones, &templates, &live, |_, _, _, _| {
            visits += 1;
        });
        assert_eq!(
            visits, 1,
            "only the tree with an Appearance component stays visible",
        );
    }

    #[test]
    fn update_setting_root_invisible_skips_all_parts() {
        let (mut zones, zid) = seed_zone();
        insert_tree(&mut zones, zid, Vec3::ZERO, None);

        let mut templates: RenderRegistry<Rid, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32>::new("tree")
                .with_mesh_part(0, 0, Mat4::IDENTITY)
                .with_mesh_part(1, 1, Mat4::from_translation(Vec3::Z)),
        );

        let mut live = RenderProxies::<Rid>::new(30);
        RenderObjectTraversal::declare_and_cull(&zones, &templates, &mut live, &always_inside());
        // Translation step: hide the whole instance.
        RenderObjectTraversal::update_instances(
            &zones,
            &templates,
            &mut live,
            |_, _, _, _, instance| {
                instance.root_visible = false;
            },
        );

        let mut visits = 0usize;
        RenderObjectTraversal::for_each_alive_part(&zones, &templates, &live, |_, _, _, _| {
            visits += 1;
        });

        assert_eq!(visits, 0, "root_invisible must skip every part");
    }

    #[test]
    fn update_can_hide_individual_mesh_nodes() {
        let (mut zones, zid) = seed_zone();
        insert_tree(&mut zones, zid, Vec3::ZERO, None);

        let mut templates: RenderRegistry<Rid, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32>::new("tree")
                .with_mesh_part(0, 0, Mat4::IDENTITY)
                .with_mesh_part(1, 1, Mat4::IDENTITY),
        );

        let mut live = RenderProxies::<Rid>::new(30);
        RenderObjectTraversal::declare_and_cull(&zones, &templates, &mut live, &always_inside());
        RenderObjectTraversal::update_instances(
            &zones,
            &templates,
            &mut live,
            |_, _, _, _, instance| {
                // Hide the second mesh node only.
                instance.nodes[1].visible = false;
            },
        );

        let mut visited: Vec<u32> = Vec::new();
        RenderObjectTraversal::for_each_alive_part(&zones, &templates, &live, |_, _, part, _| {
            visited.push(part.mesh);
        });
        assert_eq!(
            visited,
            vec![0],
            "only the node left visible by update should be visited",
        );
    }

    #[test]
    fn update_can_hide_individual_emitter_nodes() {
        let (mut zones, zid) = seed_zone();
        insert_tree(&mut zones, zid, Vec3::ZERO, None);

        let mut templates: RenderRegistry<Rid, u32, u32, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32, u32, u32>::new("tree")
                .with_emitter_part(0, 0, Mat4::IDENTITY)
                .with_emitter_part(1, 1, Mat4::IDENTITY),
        );

        let mut live = RenderProxies::<Rid>::new(30);
        RenderObjectTraversal::declare_and_cull(&zones, &templates, &mut live, &always_inside());
        RenderObjectTraversal::update_instances(
            &zones,
            &templates,
            &mut live,
            |_, _, _, _, instance| {
                instance.nodes[0].visible = false;
            },
        );

        let mut visited: Vec<u32> = Vec::new();
        RenderObjectTraversal::for_each_alive(
            &zones,
            &templates,
            &live,
            |_, _, _, _| {},
            |_, _, part, _| visited.push(part.template),
        );
        assert_eq!(
            visited,
            vec![1],
            "only the emitter node left visible by update should be visited",
        );
    }

    #[test]
    fn update_setting_root_invisible_skips_hit_id_reservation() {
        let (mut zones, zid) = seed_zone();
        // Two trees: one stays visible, one is hidden by update.
        insert_tree(&mut zones, zid, Vec3::ZERO, None);
        let hidden_position = Vec3::new(2.0, 0.0, 0.0);
        insert_tree(&mut zones, zid, hidden_position, None);

        let mut templates: RenderRegistry<Rid, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32>::new("tree").with_mesh_part(0, 0, Mat4::IDENTITY),
        );

        let mut live = RenderProxies::<Rid>::new(30);
        RenderObjectTraversal::declare_and_cull(&zones, &templates, &mut live, &always_inside());
        RenderObjectTraversal::update_instances(
            &zones,
            &templates,
            &mut live,
            |_, _, _, _, instance| {
                if (instance.world_xform.w_axis.truncate() - hidden_position).length() < 0.01 {
                    instance.root_visible = false;
                }
            },
        );

        let mut reservations = 0u32;
        let mut next_id = 1u32;
        super::for_each_alive_reserving(
            &zones,
            &templates,
            &live,
            |_parent| {
                reservations += 1;
                let id = next_id;
                next_id += 1;
                id
            },
            |_, _, _, _, _| {},
            |_, _, _, _, _| {},
        );
        assert_eq!(
            reservations, 1,
            "only the visible parent should reserve a hit ID",
        );
    }

    #[test]
    fn root_invisible_skips_emitter_nodes() {
        let (mut zones, zid) = seed_zone();
        insert_tree(&mut zones, zid, Vec3::ZERO, None);

        let mut templates: RenderRegistry<Rid, u32, u32, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32, u32, u32>::new("tree")
                .with_mesh_part(0, 0, Mat4::IDENTITY)
                .with_emitter_part(0, 0, Mat4::IDENTITY),
        );

        let mut live = RenderProxies::<Rid>::new(30);
        RenderObjectTraversal::declare_and_cull(&zones, &templates, &mut live, &always_inside());
        RenderObjectTraversal::update_instances(
            &zones,
            &templates,
            &mut live,
            |_, _, _, _, instance| {
                instance.root_visible = false;
            },
        );

        let mut mesh_visits = 0usize;
        let mut emitter_visits = 0usize;
        RenderObjectTraversal::for_each_alive(
            &zones,
            &templates,
            &live,
            |_, _, _, _| mesh_visits += 1,
            |_, _, _, _| emitter_visits += 1,
        );
        assert_eq!(mesh_visits, 0);
        assert_eq!(emitter_visits, 0);
    }

    #[test]
    fn parent_hidden_cascades_to_descendants() {
        let (mut zones, zid) = seed_zone();
        insert_tree(&mut zones, zid, Vec3::ZERO, None);

        // Three-node hierarchy: empty root, mesh child of root, mesh
        // grandchild of the mesh. Hiding the root must cascade to both.
        let root = NodeId(0);
        let mid = NodeId(1);
        let leaf = NodeId(2);
        let mut templates: RenderRegistry<Rid, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32>::new("tree")
                .with_node(TemplateNode::empty(root, "root", None, Mat4::IDENTITY))
                .with_node(TemplateNode::mesh(
                    mid,
                    "mid",
                    Some(root),
                    Mat4::IDENTITY,
                    super::MeshPart::new(0, 0),
                ))
                .with_node(TemplateNode::mesh(
                    leaf,
                    "leaf",
                    Some(mid),
                    Mat4::IDENTITY,
                    super::MeshPart::new(1, 1),
                ))
                .with_visual_bounds(Aabb::cube(2.0)),
        );

        let mut live = RenderProxies::<Rid>::new(30);
        RenderObjectTraversal::declare_and_cull(&zones, &templates, &mut live, &always_inside());

        // Hiding the root must cascade through both descendants.
        RenderObjectTraversal::update_instances(
            &zones,
            &templates,
            &mut live,
            |_, _, _, template, instance| {
                let slot = template.slot_of(root).unwrap();
                instance.nodes[slot as usize].visible = false;
            },
        );

        let mut visits = 0usize;
        RenderObjectTraversal::for_each_alive_part(&zones, &templates, &live, |_, _, _, _| {
            visits += 1;
        });
        assert_eq!(visits, 0, "hidden ancestor must cascade to descendants");
    }

    #[test]
    fn composed_world_transform_includes_ancestor_chain() {
        let (mut zones, zid) = seed_zone();
        insert_tree(&mut zones, zid, Vec3::new(5.0, 0.0, 0.0), None);

        let root = NodeId(0);
        let leaf = NodeId(1);
        let mut templates: RenderRegistry<Rid, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32>::new("tree")
                .with_node(TemplateNode::empty(
                    root,
                    "root",
                    None,
                    Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0)),
                ))
                .with_node(TemplateNode::mesh(
                    leaf,
                    "leaf",
                    Some(root),
                    Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5)),
                    super::MeshPart::new(0, 0),
                ))
                .with_visual_bounds(Aabb::cube(4.0)),
        );

        let mut live = RenderProxies::<Rid>::new(30);
        RenderObjectTraversal::declare_and_cull(&zones, &templates, &mut live, &always_inside());
        RenderObjectTraversal::update_instances(&zones, &templates, &mut live, |_, _, _, _, _| {});

        let mut saw: Option<Mat4> = None;
        RenderObjectTraversal::for_each_alive_part(&zones, &templates, &live, |_, _, _, world| {
            saw = Some(world)
        });
        let world = saw.expect("leaf node visited");
        // proxy at x=5, root lifts to z=1, leaf to z=1.5 → world = (5, 0, 1.5).
        let translation = world.col(3).truncate();
        assert!(
            (translation - Vec3::new(5.0, 0.0, 1.5)).length() < 1e-4,
            "expected composed translation (5, 0, 1.5), got {translation:?}",
        );
    }

    // Smoke test: NodeState default has effective_visible = true so a freshly
    // declared proxy with no update_instances call still gates correctly
    // until the engine fills in the cascade.
    #[test]
    fn node_state_default_is_visible() {
        let state = NodeState::default();
        assert!(state.visible);
        assert!(state.effective_visible);
    }
}
