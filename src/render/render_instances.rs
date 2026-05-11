//! Live render-object instances with frustum-cull hysteresis.
//!
//! Pairs with [`RenderTemplate`](super::RenderTemplate) the same way
//! [`EmitterReconciler`](super::EmitterReconciler) pairs with
//! [`EmitterTemplate`](super::EmitterTemplate): the View walks the sim
//! each frame and *declares* one instance per `(WorldObjectRef, R)` key,
//! supplying the instance's world transform and visual AABB. After the
//! sim walk, [`RenderInstances::cull`] tests each instance's AABB against
//! the camera frustum, updates a per-instance hysteresis counter, and
//! drops instances that have been outside the frustum longer than the
//! configured window.
//!
//! Lifecycle matches CLAUDE.md's invariants:
//! - Instances created on first visibility (a new instance whose AABB is
//!   outside the frustum is dropped on its first `cull`).
//! - Destroyed on cull-past-hysteresis or when the sim object disappears
//!   (no `declare` call this frame).
//! - View state lives only as long as the instance — no history beyond
//!   the hysteresis counter, which is a pure function of recent
//!   visibility.
//!
//! Templates with no `visual_bounds` are treated as always visible: their
//! instances are never frustum-culled and are dropped only when the sim
//! object disappears.

use std::collections::HashMap;
use std::hash::Hash;

use glam::Mat4;

use crate::sim::WorldObjectRef;

use super::visibility::{Aabb, Frustum};

/// Per-instance state carried by [`RenderInstances`]: world transform,
/// optional world-space visual AABB, and the hysteresis counter.
#[derive(Clone, Debug)]
pub struct RenderInstance {
    /// Composed world transform of the sim object owning this instance.
    /// World-space transform of a drawn part is
    /// `world_xform * part.local_transform`.
    pub world_xform: Mat4,
    /// World-space AABB used for frustum culling. `None` means the
    /// instance is never frustum-culled (treated as always visible).
    pub world_aabb: Option<Aabb>,
    frames_since_visible: u32,
    declared_this_frame: bool,
}

impl RenderInstance {
    /// Frames since this instance was last inside the frustum. `0` means
    /// visible *this frame*; positive values mean within the hysteresis
    /// window. Useful for fade-out shaders or pop-out diagnostics.
    pub fn frames_since_visible(&self) -> u32 {
        self.frames_since_visible
    }

    /// True if the instance was inside the frustum on the most recent
    /// [`RenderInstances::cull`] call.
    pub fn is_visible(&self) -> bool {
        self.frames_since_visible == 0
    }
}

/// Live instances of render-object templates, keyed by
/// `(WorldObjectRef, R)`. Generic over the render-id type `R`.
pub struct RenderInstances<R: Copy + Eq + Hash> {
    instances: HashMap<(WorldObjectRef, R), RenderInstance>,
    hysteresis_frames: u32,
}

impl<R: Copy + Eq + Hash> RenderInstances<R> {
    /// Create an empty reconciler. `hysteresis_frames` is the number of
    /// frames an instance is kept alive after it leaves the frustum;
    /// CLAUDE.md commits to ~30 as a starting point.
    pub fn new(hysteresis_frames: u32) -> Self {
        Self {
            instances: HashMap::new(),
            hysteresis_frames,
        }
    }

    pub fn hysteresis_frames(&self) -> u32 {
        self.hysteresis_frames
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Mark every existing instance as not-yet-seen this frame. Call at
    /// the start of each frame before [`declare`](Self::declare)ing.
    pub fn begin_frame(&mut self) {
        for inst in self.instances.values_mut() {
            inst.declared_this_frame = false;
        }
    }

    /// Declare an instance for this frame. Lazy-creates on first call for
    /// the `(parent, render_id)` key; otherwise refreshes its
    /// `world_xform` and `world_aabb` and marks it declared. New
    /// instances are initialised just outside the hysteresis window, so
    /// they're dropped on the next [`cull`](Self::cull) unless visible —
    /// matching CLAUDE.md's "created on first visibility" semantics.
    pub fn declare(
        &mut self,
        parent: WorldObjectRef,
        render_id: R,
        world_xform: Mat4,
        world_aabb: Option<Aabb>,
    ) {
        let init_frames = self.hysteresis_frames.saturating_add(1);
        let inst = self
            .instances
            .entry((parent, render_id))
            .or_insert(RenderInstance {
                world_xform,
                world_aabb,
                // Just past the window so a never-visible new instance is
                // dropped on the first cull, instead of lingering 30 frames.
                frames_since_visible: init_frames,
                declared_this_frame: true,
            });
        inst.world_xform = world_xform;
        inst.world_aabb = world_aabb;
        inst.declared_this_frame = true;
    }

    /// Test each declared instance against `frustum`, update its
    /// hysteresis counter, and drop instances that either are no longer
    /// declared (sim object gone) or have been outside the frustum for
    /// more than [`hysteresis_frames`](Self::hysteresis_frames) frames.
    ///
    /// Instances with no `world_aabb` are treated as always visible.
    pub fn cull(&mut self, frustum: &Frustum) {
        let hysteresis = self.hysteresis_frames;
        self.instances.retain(|_, inst| {
            if !inst.declared_this_frame {
                return false;
            }
            let visible = match &inst.world_aabb {
                Some(aabb) => frustum.contains_aabb(aabb),
                None => true,
            };
            if visible {
                inst.frames_since_visible = 0;
            } else {
                inst.frames_since_visible = inst.frames_since_visible.saturating_add(1);
            }
            inst.frames_since_visible <= hysteresis
        });
    }

    /// Iterate alive instances as `(parent, render_id, &instance)`.
    /// Alive means: declared this frame AND currently inside the frustum
    /// or within the hysteresis window.
    pub fn iter(&self) -> impl Iterator<Item = (WorldObjectRef, R, &RenderInstance)> + '_ {
        self.instances
            .iter()
            .map(|((parent, rid), inst)| (*parent, *rid, inst))
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
    fn declare_and_cull_inside_frustum_creates_instance() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: RenderInstances<Rid> = RenderInstances::new(30);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)));
        live.cull(&always_inside());

        assert_eq!(live.len(), 1);
        let (_, _, inst) = live.iter().next().unwrap();
        assert!(inst.is_visible());
    }

    #[test]
    fn new_instance_outside_frustum_is_dropped_immediately() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: RenderInstances<Rid> = RenderInstances::new(30);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)));
        live.cull(&always_outside());

        // CLAUDE.md: "Instances are created on first visibility." A new
        // instance that was never visible is dropped, not lingering.
        assert_eq!(live.len(), 0);
    }

    #[test]
    fn visible_then_invisible_lingers_for_hysteresis_window() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: RenderInstances<Rid> = RenderInstances::new(5);
        // Establish visibility.
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)));
        live.cull(&always_inside());
        assert_eq!(live.len(), 1);

        // Now invisible — should linger up to hysteresis frames.
        for f in 1..=5 {
            live.begin_frame();
            live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)));
            live.cull(&always_outside());
            assert_eq!(live.len(), 1, "instance dropped at frame {f} of hysteresis");
            let (_, _, inst) = live.iter().next().unwrap();
            assert_eq!(inst.frames_since_visible(), f);
            assert!(!inst.is_visible());
        }
        // One more frame past the window → drop.
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)));
        live.cull(&always_outside());
        assert_eq!(live.len(), 0);
    }

    #[test]
    fn returning_to_frustum_resets_counter() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: RenderInstances<Rid> = RenderInstances::new(5);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)));
        live.cull(&always_inside());

        // Three frames invisible.
        for _ in 0..3 {
            live.begin_frame();
            live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)));
            live.cull(&always_outside());
        }
        let (_, _, inst) = live.iter().next().unwrap();
        assert_eq!(inst.frames_since_visible(), 3);

        // Now visible again — counter should reset.
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)));
        live.cull(&always_inside());
        let (_, _, inst) = live.iter().next().unwrap();
        assert!(inst.is_visible());
    }

    #[test]
    fn undeclared_instance_is_dropped() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: RenderInstances<Rid> = RenderInstances::new(30);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)));
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

        let mut live: RenderInstances<Rid> = RenderInstances::new(2);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, None);
        live.cull(&always_outside()); // would normally drop on first cull

        assert_eq!(live.len(), 1);
        let (_, _, inst) = live.iter().next().unwrap();
        assert!(inst.is_visible());
    }

    #[test]
    fn multiple_render_ids_on_same_parent_are_distinct() {
        let mut zones = Zones::new();
        let parent = make_parent(&mut zones);

        let mut live: RenderInstances<Rid> = RenderInstances::new(30);
        live.begin_frame();
        live.declare(parent, Rid::A, Mat4::IDENTITY, Some(Aabb::cube(0.5)));
        live.declare(
            parent,
            Rid::B,
            Mat4::from_translation(Vec3::X),
            Some(Aabb::cube(0.5)),
        );
        live.cull(&always_inside());

        assert_eq!(live.len(), 2);
    }
}
