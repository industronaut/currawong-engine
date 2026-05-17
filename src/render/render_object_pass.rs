//! Engine-driven render-object pass.
//!
//! Owns the per-frame walk that the [`render_objects`](../../examples/render_objects.rs)
//! example used to do by hand: sim → [`RenderId`] → [`RenderTemplate`] →
//! declared proxy → frustum cull with hysteresis → fan-out per part.
//! View code supplies a per-part callback (and optionally a per-emitter
//! callback) that does the actual draw-attrib push; the engine owns the
//! traversal and validates slot values against the template schema.
//!
//! Slot-value convention: this helper expects per-object [`SlotValues`] to
//! live as a [`Components`](crate::sim::components::Components) entry on
//! the parent object. Templates declare the *schema* (names + kinds +
//! routings); the sim attaches a matching `SlotValues` component per
//! object. Views that need a different storage shape can build the loop
//! themselves — `RenderObjectPass` is the convenient default.
//!
//! Visibility is **view-side state** on [`LiveRenderObject`], set by the
//! template's update closure via [`RenderObjectPass::update_instances`]
//! and read by the extract walk. `root_visible` gates the whole instance
//! (no mesh / emitter callbacks, no hit-ID reservation); per-part
//! `mesh_parts[i].visible` / `emitter_parts[i].visible` gate individual
//! callbacks. Defaults are all `true` — templates and instances without
//! opinions on visibility are unaffected.

use std::hash::Hash;

use glam::Mat4;

use crate::sim::{Components, WorldObjectRef, Zones};

use super::live_render_objects::{LiveRenderObject, LiveRenderObjects};
use super::render_object::{EmitterPart, MeshPart, RenderRegistry, RenderTemplate, SlotValues};
use super::renderer::Renderer;
use super::visibility::Frustum;

/// Engine helper that drives the per-frame render-object walk. Stateless;
/// the associated functions take all dependencies as arguments.
pub struct RenderObjectPass;

impl RenderObjectPass {
    /// Phase 1: walk `zones` and declare a live proxy on `live_objects` for
    /// each object whose components include an `R` render id, then cull
    /// against `frustum`. Calls [`LiveRenderObjects::begin_frame`] for you.
    ///
    /// Templates with no `visual_bounds` produce proxies with no AABB,
    /// which [`LiveRenderObjects::cull`] treats as always-visible — matching
    /// CLAUDE.md's "templates without bounds are never culled" rule.
    pub fn declare_and_cull<R, M, MK, E, S>(
        zones: &Zones,
        templates: &RenderRegistry<R, M, MK, E, S>,
        live_objects: &mut LiveRenderObjects<R>,
        frustum: &Frustum,
    ) where
        R: Copy + Eq + Hash + 'static,
    {
        live_objects.begin_frame();
        for (zone_id, zone) in zones.iter() {
            for (id, obj) in zone.iter() {
                let Some(&rid) = zone.components().get::<R>(id) else {
                    continue;
                };
                let Some(template) = templates.get(rid) else {
                    continue;
                };
                let object_xform = Mat4::from_rotation_translation(obj.rotation, obj.position);
                let world_aabb = template
                    .visual_bounds()
                    .map(|local| local.transformed(object_xform));
                live_objects.declare(
                    WorldObjectRef { zone: zone_id, id },
                    rid,
                    object_xform,
                    world_aabb,
                    template.mesh_parts().len(),
                    template.emitter_parts().len(),
                );
            }
        }
        live_objects.cull(frustum);
    }

    /// Phase 1.5: per-instance update. Iterate alive proxies, validate
    /// each parent's `SlotValues` against the template schema, then
    /// invoke `on_instance` so the user can read slots / components and
    /// mutate the view-side [`LiveRenderObject`] — typically
    /// `instance.root_visible`, `instance.mesh_parts[i].visible`,
    /// `instance.emitter_parts[i].visible`, plus any future cached
    /// view-state on the instance.
    ///
    /// Call this between [`Self::declare_and_cull`] and
    /// [`Self::for_each_alive_part`] / [`Self::for_each_alive`]. The
    /// extract pass reads only `LiveRenderObject` state, so all sim →
    /// view-state translation must land here.
    ///
    /// CLAUDE.md invariants this enforces:
    /// - Sim → view translation lives in **one** place per frame.
    /// - Extract is a pure GPU-attrib write step.
    /// - Slot names are sim-vocabulary; view-side decisions
    ///   (`mesh_parts[i].visible`, etc.) are derived here, not published
    ///   by the sim.
    pub fn update_instances<R, M, MK, E, S>(
        zones: &Zones,
        templates: &RenderRegistry<R, M, MK, E, S>,
        live_objects: &mut LiveRenderObjects<R>,
        mut on_instance: impl FnMut(WorldObjectRef, R, &SlotValues, &Components, &mut LiveRenderObject),
    ) where
        R: Copy + Eq + Hash + 'static,
    {
        let empty = SlotValues::new();
        for (parent, rid, obj) in live_objects.iter_mut() {
            let Some(template) = templates.get(rid) else {
                continue;
            };
            let zone = zones
                .get(parent.zone)
                .expect("zone alive while proxy lives");
            let slots = zone
                .components()
                .get::<SlotValues>(parent.id)
                .unwrap_or(&empty);
            validate_slot_values(template, slots);
            on_instance(parent, rid, slots, zone.components(), obj);
        }
    }

    /// Phase 2 (mesh-only): iterate alive proxies, validate each parent's
    /// `SlotValues` against the template schema, then invoke `on_part` for
    /// every [`MeshPart`] with its world transform composed against the
    /// proxy's world transform.
    ///
    /// The slot-values lookup is a single `Components::get::<SlotValues>(id)`
    /// per alive proxy; objects without that component are treated as
    /// having an empty `SlotValues` (templates may then fall back to
    /// defaults in `on_part`).
    ///
    /// Thin wrapper over [`Self::for_each_alive`] with a no-op emitter
    /// callback; the traversal lives in one place.
    pub fn for_each_alive_part<R, M, MK, E, S>(
        zones: &Zones,
        templates: &RenderRegistry<R, M, MK, E, S>,
        live_objects: &LiveRenderObjects<R>,
        on_part: impl FnMut(WorldObjectRef, R, &MeshPart<M, MK>, Mat4, &SlotValues),
    ) where
        R: Copy + Eq + Hash + 'static,
    {
        Self::for_each_alive(zones, templates, live_objects, on_part, |_, _, _, _, _| {});
    }

    /// Phase 2 (mesh + emitter): iterate alive proxies, validate each
    /// parent's `SlotValues` against the template schema, then invoke
    /// `on_part` for every [`MeshPart`] and `on_emitter` for every
    /// [`EmitterPart`]. Slot validation happens once per alive proxy;
    /// both callbacks see the same `&SlotValues`.
    ///
    /// Visibility is **view-side state** read from the
    /// [`LiveRenderObject`]: instances with `root_visible == false`
    /// are skipped entirely, and per-part `mesh_parts[i].visible` /
    /// `emitter_parts[i].visible` gate individual callbacks. These
    /// fields are set by the user's update closure passed to
    /// [`Self::update_instances`].
    ///
    /// Use this when a template carries [`EmitterPart`]s — the demo at
    /// [`examples/render_objects.rs`](../../examples/render_objects.rs)
    /// does. Pure-mesh views can call [`Self::for_each_alive_part`] for the
    /// same walk without the empty emitter closure.
    pub fn for_each_alive<R, M, MK, E, S>(
        zones: &Zones,
        templates: &RenderRegistry<R, M, MK, E, S>,
        live_objects: &LiveRenderObjects<R>,
        mut on_part: impl FnMut(WorldObjectRef, R, &MeshPart<M, MK>, Mat4, &SlotValues),
        mut on_emitter: impl FnMut(WorldObjectRef, R, &EmitterPart<E, S>, Mat4, &SlotValues),
    ) where
        R: Copy + Eq + Hash + 'static,
    {
        let empty = SlotValues::new();
        for (parent, rid, object) in live_objects.iter() {
            if !object.root_visible {
                continue;
            }
            let Some(template) = templates.get(rid) else {
                continue;
            };
            let zone = zones
                .get(parent.zone)
                .expect("zone alive while proxy lives");
            let slots = zone
                .components()
                .get::<SlotValues>(parent.id)
                .unwrap_or(&empty);
            validate_slot_values(template, slots);

            for (i, part) in template.mesh_parts().iter().enumerate() {
                if !object.mesh_parts[i].visible {
                    continue;
                }
                let world = object.world_xform * part.local_transform;
                on_part(parent, rid, part, world, slots);
            }
            for (i, part) in template.emitter_parts().iter().enumerate() {
                if !object.emitter_parts[i].visible {
                    continue;
                }
                let world = object.world_xform * part.local_transform;
                on_emitter(parent, rid, part, world, slots);
            }
        }
    }

    /// Phase 2 (mesh + emitter), with GPU hit-ID reservation (#56 PR 3).
    ///
    /// Same shape as [`Self::for_each_alive`], with one extra step: per
    /// alive parent, reserve a single hit ID via
    /// [`Renderer::reserve_object`] and pass it to **both** the mesh and
    /// emitter callbacks. All mesh parts of one sim object share the same
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
        live_objects: &LiveRenderObjects<R>,
        renderer: &Renderer,
        on_part: impl FnMut(WorldObjectRef, R, &MeshPart<M, MK>, Mat4, &SlotValues, u32),
        on_emitter: impl FnMut(WorldObjectRef, R, &EmitterPart<E, S>, Mat4, &SlotValues, u32),
    ) where
        R: Copy + Eq + Hash + 'static,
    {
        for_each_alive_reserving(
            zones,
            templates,
            live_objects,
            |parent| renderer.reserve_object(parent.zone, parent.id),
            on_part,
            on_emitter,
        );
    }

    /// Mesh-only variant of [`Self::for_each_alive_with_hit_id`] — the
    /// convenient default for templates that carry no emitter parts.
    pub fn for_each_alive_part_with_hit_id<R, M, MK, E, S>(
        zones: &Zones,
        templates: &RenderRegistry<R, M, MK, E, S>,
        live_objects: &LiveRenderObjects<R>,
        renderer: &Renderer,
        on_part: impl FnMut(WorldObjectRef, R, &MeshPart<M, MK>, Mat4, &SlotValues, u32),
    ) where
        R: Copy + Eq + Hash + 'static,
    {
        Self::for_each_alive_with_hit_id(
            zones,
            templates,
            live_objects,
            renderer,
            on_part,
            |_, _, _, _, _, _| {},
        );
    }
}

/// Inner traversal shared by [`RenderObjectPass::for_each_alive_with_hit_id`]
/// and its tests. The `reserve` closure is called once per *visible* parent
/// — invisible parents short-circuit before reservation, so the hit-ID
/// counter doesn't advance for objects the user can't click.
fn for_each_alive_reserving<R, M, MK, E, S>(
    zones: &Zones,
    templates: &RenderRegistry<R, M, MK, E, S>,
    live_objects: &LiveRenderObjects<R>,
    mut reserve: impl FnMut(WorldObjectRef) -> u32,
    mut on_part: impl FnMut(WorldObjectRef, R, &MeshPart<M, MK>, Mat4, &SlotValues, u32),
    mut on_emitter: impl FnMut(WorldObjectRef, R, &EmitterPart<E, S>, Mat4, &SlotValues, u32),
) where
    R: Copy + Eq + Hash + 'static,
{
    let empty = SlotValues::new();
    for (parent, rid, object) in live_objects.iter() {
        if !object.root_visible {
            continue;
        }
        let Some(template) = templates.get(rid) else {
            continue;
        };
        let zone = zones
            .get(parent.zone)
            .expect("zone alive while proxy lives");
        let slots = zone
            .components()
            .get::<SlotValues>(parent.id)
            .unwrap_or(&empty);
        validate_slot_values(template, slots);

        let hit_id = reserve(parent);

        for (i, part) in template.mesh_parts().iter().enumerate() {
            if !object.mesh_parts[i].visible {
                continue;
            }
            let world = object.world_xform * part.local_transform;
            on_part(parent, rid, part, world, slots, hit_id);
        }
        for (i, part) in template.emitter_parts().iter().enumerate() {
            if !object.emitter_parts[i].visible {
                continue;
            }
            let world = object.world_xform * part.local_transform;
            on_emitter(parent, rid, part, world, slots, hit_id);
        }
    }
}

/// Validate that every value in `values` whose name appears in the
/// template's schema has a matching [`SlotKind`].
///
/// Extra values whose names aren't in the schema are silently ignored —
/// templates are the authority on what they consume. Missing values are
/// also tolerated; the per-part callback is responsible for falling back.
///
/// [`SlotRouting::Uniform`] is rejected at template-build time by
/// [`RenderTemplate::with_routed_slot`], so it cannot reach here.
///
/// Panics on schema violations; these are programming errors, not runtime
/// conditions. The body is gated on `debug_assertions` — in release builds
/// this is a no-op so the per-frame, per-alive-instance call from
/// [`RenderObjectPass`] costs nothing once the developer's templates are
/// known good.
pub fn validate_slot_values<M, MK, E, S>(
    template: &RenderTemplate<M, MK, E, S>,
    values: &SlotValues,
) {
    if !cfg!(debug_assertions) {
        return;
    }
    for (name, value) in values.iter() {
        if let Some(slot) = template.slot(name) {
            assert_eq!(
                slot.kind,
                value.kind(),
                "slot '{}' on template '{}' expects {:?} but SlotValues has {:?}",
                name,
                template.label,
                slot.kind,
                value.kind(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::render_object::{SlotKind, SlotValue};
    use crate::render::visibility::Aabb;
    use crate::sim::{WorldTransform, Zone, ZoneId, Zones};
    use glam::{Quat, Vec3, Vec4};

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Rid {
        Tree,
    }

    fn always_inside() -> Frustum {
        // Same trick as live_render_objects tests: every plane accepts.
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

    fn insert_tree(zones: &mut Zones, zid: ZoneId, position: Vec3, slots: Option<SlotValues>) {
        let zone = zones.get_mut(zid).expect("zone");
        let id = zone.insert(WorldTransform {
            position,
            rotation: Quat::IDENTITY,
        });
        zone.components_mut().insert(id, Rid::Tree);
        if let Some(s) = slots {
            zone.components_mut().insert(id, s);
        }
    }

    fn tree_template() -> RenderTemplate {
        RenderTemplate::new("tree")
            .with_slot("height", SlotKind::F32)
            .with_slot("tint", SlotKind::Color)
            .with_visual_bounds(Aabb::new(
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

        let mut live = LiveRenderObjects::<Rid>::new(30);
        RenderObjectPass::declare_and_cull(&zones, &templates, &mut live, &always_inside());

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

        let mut live = LiveRenderObjects::<Rid>::new(30);
        RenderObjectPass::declare_and_cull(&zones, &templates, &mut live, &always_inside());

        assert!(live.is_empty());
    }

    #[test]
    fn declare_and_cull_drops_offscreen_first_time() {
        let (mut zones, zid) = seed_zone();
        insert_tree(&mut zones, zid, Vec3::ZERO, None);

        let mut templates: RenderRegistry<Rid> = RenderRegistry::new();
        templates.register(Rid::Tree, tree_template());

        let mut live = LiveRenderObjects::<Rid>::new(30);
        RenderObjectPass::declare_and_cull(&zones, &templates, &mut live, &always_outside());

        // New proxy, never visible → dropped immediately.
        assert!(live.is_empty());
    }

    #[test]
    fn for_each_alive_part_visits_each_template_part() {
        let (mut zones, zid) = seed_zone();
        let slots = SlotValues::new()
            .with("height", SlotValue::F32(1.7))
            .with("tint", SlotValue::Color(Vec4::new(0.2, 0.8, 0.3, 1.0)));
        insert_tree(&mut zones, zid, Vec3::ZERO, Some(slots));

        let mut templates: RenderRegistry<Rid, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32>::new("tree")
                .with_slot("height", SlotKind::F32)
                .with_mesh_part(0, 0, Mat4::IDENTITY)
                .with_mesh_part(1, 1, Mat4::from_translation(Vec3::Z)),
        );

        let mut live = LiveRenderObjects::<Rid>::new(30);
        RenderObjectPass::declare_and_cull(&zones, &templates, &mut live, &always_inside());

        let mut visits = 0usize;
        let mut height = 0.0f32;
        RenderObjectPass::for_each_alive_part(
            &zones,
            &templates,
            &live,
            |_parent, _rid, part, _world, slot_values| {
                visits += 1;
                // Height should be visible across all parts of one alive proxy.
                if let Some(SlotValue::F32(h)) = slot_values.get("height") {
                    height = h;
                }
                // Sanity: the second part's mesh index matches the registered ordering.
                let _ = part.mesh;
            },
        );

        assert_eq!(visits, 2, "two mesh parts → two callback invocations");
        assert_eq!(height, 1.7);
    }

    #[test]
    fn for_each_alive_part_tolerates_missing_slot_values_component() {
        let (mut zones, zid) = seed_zone();
        // No SlotValues on this tree.
        insert_tree(&mut zones, zid, Vec3::ZERO, None);

        let mut templates: RenderRegistry<Rid, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32>::new("tree")
                .with_slot("height", SlotKind::F32)
                .with_mesh_part(0, 0, Mat4::IDENTITY),
        );

        let mut live = LiveRenderObjects::<Rid>::new(30);
        RenderObjectPass::declare_and_cull(&zones, &templates, &mut live, &always_inside());

        let mut visits = 0usize;
        RenderObjectPass::for_each_alive_part(&zones, &templates, &live, |_, _, _, _, slots| {
            visits += 1;
            assert!(slots.is_empty(), "fallback empty SlotValues expected");
        });
        assert_eq!(visits, 1);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "slot 'tint' on template 'tree' expects Color")]
    fn validate_panics_on_kind_mismatch() {
        let template = tree_template();
        let mismatched = SlotValues::new().with("tint", SlotValue::F32(0.0));
        validate_slot_values(&template, &mismatched);
    }

    #[test]
    fn validate_ignores_extra_slot_values() {
        let template = tree_template();
        let extra = SlotValues::new()
            .with("height", SlotValue::F32(1.0))
            .with("nonexistent", SlotValue::Bool(true));
        validate_slot_values(&template, &extra); // must not panic
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

        let mut live = LiveRenderObjects::<Rid>::new(30);
        RenderObjectPass::declare_and_cull(&zones, &templates, &mut live, &always_inside());
        // Translation step: hide the whole instance.
        RenderObjectPass::update_instances(
            &zones,
            &templates,
            &mut live,
            |_, _, _, _, instance| {
                instance.root_visible = false;
            },
        );

        let mut visits = 0usize;
        RenderObjectPass::for_each_alive_part(&zones, &templates, &live, |_, _, _, _, _| {
            visits += 1;
        });

        assert_eq!(visits, 0, "root_invisible must skip every part");
    }

    #[test]
    fn update_can_hide_individual_mesh_parts() {
        let (mut zones, zid) = seed_zone();
        insert_tree(&mut zones, zid, Vec3::ZERO, None);

        let mut templates: RenderRegistry<Rid, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32>::new("tree")
                .with_mesh_part(0, 0, Mat4::IDENTITY)
                .with_mesh_part(1, 1, Mat4::IDENTITY),
        );

        let mut live = LiveRenderObjects::<Rid>::new(30);
        RenderObjectPass::declare_and_cull(&zones, &templates, &mut live, &always_inside());
        RenderObjectPass::update_instances(
            &zones,
            &templates,
            &mut live,
            |_, _, _, _, instance| {
                // Hide the second mesh part only.
                instance.mesh_parts[1].visible = false;
            },
        );

        let mut visited: Vec<u32> = Vec::new();
        RenderObjectPass::for_each_alive_part(&zones, &templates, &live, |_, _, part, _, _| {
            visited.push(part.mesh);
        });
        assert_eq!(
            visited,
            vec![0],
            "only the part left visible by update should be visited",
        );
    }

    #[test]
    fn update_can_hide_individual_emitter_parts() {
        let (mut zones, zid) = seed_zone();
        insert_tree(&mut zones, zid, Vec3::ZERO, None);

        let mut templates: RenderRegistry<Rid, u32, u32, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32, u32, u32>::new("tree")
                .with_emitter_part(0, 0, Mat4::IDENTITY)
                .with_emitter_part(1, 1, Mat4::IDENTITY),
        );

        let mut live = LiveRenderObjects::<Rid>::new(30);
        RenderObjectPass::declare_and_cull(&zones, &templates, &mut live, &always_inside());
        RenderObjectPass::update_instances(
            &zones,
            &templates,
            &mut live,
            |_, _, _, _, instance| {
                instance.emitter_parts[0].visible = false;
            },
        );

        let mut visited: Vec<u32> = Vec::new();
        RenderObjectPass::for_each_alive(
            &zones,
            &templates,
            &live,
            |_, _, _, _, _| {},
            |_, _, part, _, _| visited.push(part.template),
        );
        assert_eq!(
            visited,
            vec![1],
            "only the emitter left visible by update should be visited",
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

        let mut live = LiveRenderObjects::<Rid>::new(30);
        RenderObjectPass::declare_and_cull(&zones, &templates, &mut live, &always_inside());
        RenderObjectPass::update_instances(
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
            |_, _, _, _, _, _| {},
            |_, _, _, _, _, _| {},
        );
        assert_eq!(
            reservations, 1,
            "only the visible parent should reserve a hit ID",
        );
    }

    #[test]
    fn root_invisible_skips_emitter_parts() {
        let (mut zones, zid) = seed_zone();
        insert_tree(&mut zones, zid, Vec3::ZERO, None);

        let mut templates: RenderRegistry<Rid, u32, u32, u32, u32> = RenderRegistry::new();
        templates.register(
            Rid::Tree,
            RenderTemplate::<u32, u32, u32, u32>::new("tree")
                .with_mesh_part(0, 0, Mat4::IDENTITY)
                .with_emitter_part(0, 0, Mat4::IDENTITY),
        );

        let mut live = LiveRenderObjects::<Rid>::new(30);
        RenderObjectPass::declare_and_cull(&zones, &templates, &mut live, &always_inside());
        RenderObjectPass::update_instances(
            &zones,
            &templates,
            &mut live,
            |_, _, _, _, instance| {
                instance.root_visible = false;
            },
        );

        let mut mesh_visits = 0usize;
        let mut emitter_visits = 0usize;
        RenderObjectPass::for_each_alive(
            &zones,
            &templates,
            &live,
            |_, _, _, _, _| mesh_visits += 1,
            |_, _, _, _, _| emitter_visits += 1,
        );
        assert_eq!(mesh_visits, 0);
        assert_eq!(emitter_visits, 0);
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn validate_is_no_op_in_release() {
        // Same inputs that panic in debug — must be silently accepted in release.
        let template = tree_template();
        let mismatched = SlotValues::new().with("tint", SlotValue::F32(0.0));
        validate_slot_values(&template, &mismatched);
    }
}
