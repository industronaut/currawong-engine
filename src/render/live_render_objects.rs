//! Live render-object proxies with frustum-cull hysteresis.
//!
//! Pairs with [`RenderTemplate`](super::RenderTemplate) the same way
//! [`EmitterReconciler`](super::EmitterReconciler) pairs with
//! [`EmitterTemplate`](super::EmitterTemplate): the View walks the sim
//! each frame and *declares* one [`LiveRenderObject`] per
//! `(WorldObjectRef, R)` key, supplying its world transform, visual
//! AABB, and the template's mesh/emitter part counts. After the sim
//! walk, [`LiveRenderObjects::cull`] tests each object's AABB against
//! the camera frustum, updates a per-object hysteresis counter, and
//! drops objects that have been outside the frustum longer than the
//! configured window.
//!
//! Each proxy also carries persistent **view-side per-instance state**:
//! [`LiveRenderObject::root_visible`] and per-part visibility in
//! [`mesh_parts`](LiveRenderObject::mesh_parts) /
//! [`emitter_parts`](LiveRenderObject::emitter_parts). This is where the
//! sim→view translation lands each frame: the template's `update` hook
//! reads slots/components from the parent sim object and mutates these
//! fields; the extract walk then reads only this state, never the sim
//! directly. CLAUDE.md "All sim→view translation happens in the
//! per-instance update."
//!
//! Lifecycle matches CLAUDE.md's invariants:
//! - Objects created on first visibility (a new object whose AABB is
//!   outside the frustum is dropped on its first `cull`).
//! - Destroyed on cull-past-hysteresis or when the sim object disappears
//!   (no `declare` call this frame).
//! - View state lives only as long as the proxy — no history beyond
//!   the hysteresis counter and per-part flags, which are recomputed
//!   from sim state each frame via the update hook.
//!
//! Templates with no `visual_bounds` are treated as always visible: their
//! proxies are never frustum-culled and are dropped only when the sim
//! object disappears.

use std::collections::HashMap;
use std::hash::Hash;

use glam::Mat4;

use crate::sim::WorldObjectRef;

use super::visibility::{Aabb, Frustum};

/// View-side per-part state carried by [`LiveRenderObject`]. Mutated by
/// the template's `update` hook; read by the extract walk.
///
/// `visible` defaults to `true` on first declare. Setting it `false`
/// means the engine skips this part's extract callback this frame —
/// equivalent to "extract produced no attribs."
#[derive(Clone, Debug)]
pub struct RenderPartState {
    pub visible: bool,
}

impl Default for RenderPartState {
    fn default() -> Self {
        Self { visible: true }
    }
}

/// Per-object state carried by [`LiveRenderObjects`]: world transform,
/// optional world-space visual AABB, the hysteresis counter, and
/// persistent view-side per-instance state ([`root_visible`](Self::root_visible)
/// plus per-part visibility).
///
/// This is the view-side proxy for one sim object × render template
/// pair — analogous to UE's `PrimitiveSceneProxy`. The `*_parts` Vecs
/// are sized at declare time from the template's mesh/emitter part
/// counts and resized in place if those ever change.
#[derive(Clone, Debug)]
pub struct LiveRenderObject {
    /// Composed world transform of the sim object owning this proxy.
    /// World-space transform of a drawn part is
    /// `world_xform * part.local_transform`.
    pub world_xform: Mat4,
    /// World-space AABB used for frustum culling. `None` means the
    /// proxy is never frustum-culled (treated as always visible).
    pub world_aabb: Option<Aabb>,
    /// Whole-instance visibility. When `false`, the engine skips every
    /// mesh and emitter callback for this proxy — no hit-ID is reserved
    /// either. Defaults to `true`; set in the update hook.
    pub root_visible: bool,
    /// Per-mesh-part view-side state, parallel to
    /// [`RenderTemplate::mesh_parts`](super::RenderTemplate::mesh_parts).
    pub mesh_parts: Vec<RenderPartState>,
    /// Per-emitter-part view-side state, parallel to
    /// [`RenderTemplate::emitter_parts`](super::RenderTemplate::emitter_parts).
    pub emitter_parts: Vec<RenderPartState>,
    frames_since_visible: u32,
    declared_this_frame: bool,
}

impl LiveRenderObject {
    /// Frames since this proxy was last inside the frustum. `0` means
    /// visible *this frame*; positive values mean within the hysteresis
    /// window. Useful for fade-out shaders or pop-out diagnostics.
    pub fn frames_since_visible(&self) -> u32 {
        self.frames_since_visible
    }

    /// True if the proxy was inside the frustum on the most recent
    /// [`LiveRenderObjects::cull`] call.
    pub fn is_visible(&self) -> bool {
        self.frames_since_visible == 0
    }
}

/// Live proxies of render-object templates, keyed by
/// `(WorldObjectRef, R)`. Generic over the render-id type `R`.
pub struct LiveRenderObjects<R: Copy + Eq + Hash> {
    objects: HashMap<(WorldObjectRef, R), LiveRenderObject>,
    hysteresis_frames: u32,
}

impl<R: Copy + Eq + Hash> LiveRenderObjects<R> {
    /// Create an empty reconciler. `hysteresis_frames` is the number of
    /// frames a proxy is kept alive after it leaves the frustum;
    /// CLAUDE.md commits to ~30 as a starting point.
    pub fn new(hysteresis_frames: u32) -> Self {
        Self {
            objects: HashMap::new(),
            hysteresis_frames,
        }
    }

    pub fn hysteresis_frames(&self) -> u32 {
        self.hysteresis_frames
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Mark every existing proxy as not-yet-seen this frame. Call at
    /// the start of each frame before [`declare`](Self::declare)ing.
    pub fn begin_frame(&mut self) {
        for obj in self.objects.values_mut() {
            obj.declared_this_frame = false;
        }
    }

    /// Declare a proxy for this frame. Lazy-creates on first call for
    /// the `(parent, render_id)` key; otherwise refreshes its
    /// `world_xform` and `world_aabb` and marks it declared. New
    /// proxies are initialised just outside the hysteresis window, so
    /// they're dropped on the next [`cull`](Self::cull) unless visible —
    /// matching CLAUDE.md's "created on first visibility" semantics.
    ///
    /// `mesh_parts_count` and `emitter_parts_count` size the per-part
    /// state Vecs from the template; if they change on a later declare
    /// (e.g. template was replaced), the Vecs are resized in place,
    /// new entries default to `visible = true`. Per-part visibility
    /// from prior frames is preserved across declares — the update hook
    /// is responsible for re-asserting it, since view state is
    /// recoverable from sim state.
    pub fn declare(
        &mut self,
        parent: WorldObjectRef,
        render_id: R,
        world_xform: Mat4,
        world_aabb: Option<Aabb>,
        mesh_parts_count: usize,
        emitter_parts_count: usize,
    ) {
        let init_frames = self.hysteresis_frames.saturating_add(1);
        let obj = self
            .objects
            .entry((parent, render_id))
            .or_insert_with(|| LiveRenderObject {
                world_xform,
                world_aabb,
                root_visible: true,
                mesh_parts: vec![RenderPartState::default(); mesh_parts_count],
                emitter_parts: vec![RenderPartState::default(); emitter_parts_count],
                // Just past the window so a never-visible new proxy is
                // dropped on the first cull, instead of lingering 30 frames.
                frames_since_visible: init_frames,
                declared_this_frame: true,
            });
        obj.world_xform = world_xform;
        obj.world_aabb = world_aabb;
        obj.declared_this_frame = true;
        if obj.mesh_parts.len() != mesh_parts_count {
            obj.mesh_parts
                .resize(mesh_parts_count, RenderPartState::default());
        }
        if obj.emitter_parts.len() != emitter_parts_count {
            obj.emitter_parts
                .resize(emitter_parts_count, RenderPartState::default());
        }
    }

    /// Test each declared proxy against `frustum`, update its
    /// hysteresis counter, and drop proxies that either are no longer
    /// declared (sim object gone) or have been outside the frustum for
    /// more than [`hysteresis_frames`](Self::hysteresis_frames) frames.
    ///
    /// Proxies with no `world_aabb` are treated as always visible.
    pub fn cull(&mut self, frustum: &Frustum) {
        let hysteresis = self.hysteresis_frames;
        self.objects.retain(|_, obj| {
            if !obj.declared_this_frame {
                return false;
            }
            let visible = match &obj.world_aabb {
                Some(aabb) => frustum.contains_aabb(aabb),
                None => true,
            };
            if visible {
                obj.frames_since_visible = 0;
            } else {
                obj.frames_since_visible = obj.frames_since_visible.saturating_add(1);
            }
            obj.frames_since_visible <= hysteresis
        });
    }

    /// Iterate alive proxies as `(parent, render_id, &object)`.
    /// Alive means: declared this frame AND currently inside the frustum
    /// or within the hysteresis window.
    pub fn iter(&self) -> impl Iterator<Item = (WorldObjectRef, R, &LiveRenderObject)> + '_ {
        self.objects
            .iter()
            .map(|((parent, rid), obj)| (*parent, *rid, obj))
    }

    /// Mutable variant of [`Self::iter`], for the per-instance update
    /// pass. Used by `RenderObjectPass::update_instances` to invoke the
    /// template's `update` hook with `&mut LiveRenderObject`.
    pub fn iter_mut(
        &mut self,
    ) -> impl Iterator<Item = (WorldObjectRef, R, &mut LiveRenderObject)> + '_ {
        self.objects
            .iter_mut()
            .map(|((parent, rid), obj)| (*parent, *rid, obj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{Zone, ZoneId, Zones};
    use glam::{Vec3, Vec4};

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Rid {
        A,
        B,
    }

    fn make_parent(zones: &mut Zones) -> WorldObjectRef {
        let zid: ZoneId = zones.insert(Zone::new());
        let oid = zones.get_mut(zid).unwrap().insert(Default::default());
        WorldObjectRef { zone: zid, id: oid }
    }

    /// A frustum that accepts everything. Saves having to construct a
    /// view-proj for tests that don't care about cull geometry.
    fn always_inside() -> Frustum {
        Frustum {
            planes: [Vec4::new(0.0, 0.0, 0.0, 1.0); 6],
        }
    }

    /// A frustum that rejects everything (plane `0x + 0y + 0z + -1 <= 0`
    /// for the AABB centre; radius is 0 so reject succeeds).
    fn always_outside() -> Frustum {
        Frustum {
            planes: [Vec4::new(0.0, 0.0, 0.0, -1.0); 6],
        }
    }

    #[test]
    fn declare_and_cull_inside_frustum_creates_object() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: LiveRenderObjects<Rid> = LiveRenderObjects::new(30);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 0, 0);
        live.cull(&always_inside());

        assert_eq!(live.len(), 1);
        let (_, _, obj) = live.iter().next().unwrap();
        assert!(obj.is_visible());
    }

    #[test]
    fn new_object_outside_frustum_is_dropped_immediately() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: LiveRenderObjects<Rid> = LiveRenderObjects::new(30);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 0, 0);
        live.cull(&always_outside());

        // CLAUDE.md: "Instances are created on first visibility." A new
        // proxy that was never visible is dropped, not lingering.
        assert_eq!(live.len(), 0);
    }

    #[test]
    fn visible_then_invisible_lingers_for_hysteresis_window() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: LiveRenderObjects<Rid> = LiveRenderObjects::new(5);
        // Establish visibility.
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 0, 0);
        live.cull(&always_inside());
        assert_eq!(live.len(), 1);

        // Now invisible — should linger up to hysteresis frames.
        for f in 1..=5 {
            live.begin_frame();
            live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 0, 0);
            live.cull(&always_outside());
            assert_eq!(live.len(), 1, "proxy dropped at frame {f} of hysteresis");
            let (_, _, obj) = live.iter().next().unwrap();
            assert_eq!(obj.frames_since_visible(), f);
            assert!(!obj.is_visible());
        }
        // One more frame past the window → drop.
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 0, 0);
        live.cull(&always_outside());
        assert_eq!(live.len(), 0);
    }

    #[test]
    fn returning_to_frustum_resets_counter() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: LiveRenderObjects<Rid> = LiveRenderObjects::new(5);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 0, 0);
        live.cull(&always_inside());

        // Three frames invisible.
        for _ in 0..3 {
            live.begin_frame();
            live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 0, 0);
            live.cull(&always_outside());
        }
        let (_, _, obj) = live.iter().next().unwrap();
        assert_eq!(obj.frames_since_visible(), 3);

        // Now visible again — counter should reset.
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 0, 0);
        live.cull(&always_inside());
        let (_, _, obj) = live.iter().next().unwrap();
        assert!(obj.is_visible());
    }

    #[test]
    fn undeclared_object_is_dropped() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: LiveRenderObjects<Rid> = LiveRenderObjects::new(30);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 0, 0);
        live.cull(&always_inside());
        assert_eq!(live.len(), 1);

        // Sim object went away — not declared this frame.
        live.begin_frame();
        live.cull(&always_inside());
        assert_eq!(live.len(), 0);
    }

    #[test]
    fn template_without_bounds_is_never_culled() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: LiveRenderObjects<Rid> = LiveRenderObjects::new(2);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, None, 0, 0);
        live.cull(&always_outside()); // would normally drop on first cull

        assert_eq!(live.len(), 1);
        let (_, _, obj) = live.iter().next().unwrap();
        assert!(obj.is_visible());
    }

    #[test]
    fn part_state_is_sized_from_declare_and_defaults_visible() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: LiveRenderObjects<Rid> = LiveRenderObjects::new(30);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 3, 2);
        live.cull(&always_inside());

        let (_, _, obj) = live.iter().next().unwrap();
        assert_eq!(obj.mesh_parts.len(), 3);
        assert_eq!(obj.emitter_parts.len(), 2);
        assert!(obj.root_visible);
        assert!(obj.mesh_parts.iter().all(|p| p.visible));
        assert!(obj.emitter_parts.iter().all(|p| p.visible));
    }

    #[test]
    fn part_state_preserved_across_redeclare() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: LiveRenderObjects<Rid> = LiveRenderObjects::new(30);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 2, 0);
        live.cull(&always_inside());

        // Mutate per-part state (this is what the update hook will do).
        for (_, _, obj) in live.iter_mut() {
            obj.mesh_parts[0].visible = false;
            obj.root_visible = false;
        }

        // Re-declare on the next frame — state should persist; the update
        // hook re-asserts it. (No auto-reset to defaults on redeclare.)
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 2, 0);

        let (_, _, obj) = live.iter().next().unwrap();
        assert!(
            !obj.root_visible,
            "root_visible must persist across redeclare"
        );
        assert!(
            !obj.mesh_parts[0].visible,
            "per-part visibility must persist across redeclare"
        );
        assert!(obj.mesh_parts[1].visible);
    }

    #[test]
    fn part_state_resizes_when_count_changes() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: LiveRenderObjects<Rid> = LiveRenderObjects::new(30);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 1, 1);

        // Template was swapped — more mesh parts, fewer emitters.
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 3, 0);

        let (_, _, obj) = live.iter().next().unwrap();
        assert_eq!(obj.mesh_parts.len(), 3);
        assert_eq!(obj.emitter_parts.len(), 0);
        // New entries default to visible.
        assert!(obj.mesh_parts.iter().all(|p| p.visible));
    }

    #[test]
    fn multiple_render_ids_on_same_parent_are_distinct() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: LiveRenderObjects<Rid> = LiveRenderObjects::new(30);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)), 0, 0);
        live.declare(
            parent,
            Rid::B,
            Mat4::from_translation(Vec3::X),
            Some(Aabb::cube(0.5)),
            0,
            0,
        );
        live.cull(&always_inside());

        assert_eq!(live.len(), 2);
    }
}
