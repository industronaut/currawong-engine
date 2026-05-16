//! Geometric movement. Any object with a [`Move`] component advances toward
//! its target each tick; [`advance`] removes `Move` on arrival and returns
//! the arrived ids so intent modules can react in the same tick.
//!
//! Pure of intent — `Move` doesn't know *why* the object is moving, only
//! where to and how fast. The intent modules ([`super::chopping`],
//! [`super::hauling`]) own the lifecycle: they insert `Move` when they want
//! motion and read [`advance`]'s arrival list to react when the pawn gets
//! there.

use currawong::glam::Quat;
use currawong::{WorldObjectId, Zone};

/// Advance the parent's transform toward `target` at `speed` metres per
/// second each tick. Removed by [`advance`] on arrival.
pub struct Move {
    pub target: currawong::glam::Vec3,
    pub speed: f32,
}

/// Step every Move-bearing object toward its target. Returns the ids of
/// objects that arrived this tick — their `Move` has already been removed
/// by the time the call returns, so intent modules can call e.g.
/// [`super::chopping::on_arrival`] on the same slice.
///
/// Also faces the object along its velocity (local +X = "forward", rotating
/// around Z). Capsule pawns are rotationally symmetric so this isn't visible
/// today, but the per-instance rotation flows through `WorldTransform.rotation`
/// into the pawn template's model matrix end-to-end.
pub fn advance(zone: &mut Zone, dt: f32) -> Vec<WorldObjectId> {
    let mut arrived: Vec<WorldObjectId> = Vec::new();
    {
        let (mut objects, components) = zone.split_mut();
        for (pawn, m) in components.iter::<Move>() {
            let Some(transform) = objects.get_mut(pawn) else {
                continue;
            };
            let delta = m.target - transform.position;
            let dist = delta.length();
            let step = m.speed * dt;
            if dist <= step.max(1e-3) {
                transform.position = m.target;
                arrived.push(pawn);
            } else {
                let dir = delta / dist;
                transform.position += dir * step;
                transform.rotation = Quat::from_rotation_z(dir.y.atan2(dir.x));
            }
        }
    }
    for &pawn in &arrived {
        zone.components_mut().remove::<Move>(pawn);
    }
    arrived
}
