//! Geometric movement. Any object with a [`Move`] component advances toward
//! its target each tick; [`advance`] removes `Move` on arrival and returns
//! the arrived ids so intent modules can react in the same tick.
//!
//! Pure of intent — `Move` doesn't know *why* the object is moving, only
//! where to and how fast. The intent modules ([`super::chopping`],
//! [`super::hauling`]) own the lifecycle: they insert `Move` when they want
//! motion and read [`advance`]'s arrival list to react when the pawn gets
//! there.

use currawong::{Facing, SimPos, SimUnit, SimVec, WorldObjectId, Zone};

/// Advance the parent's transform toward `target` at `speed` Q16.16 metres
/// per second each tick. Removed by [`advance`] on arrival.
pub struct Move {
    pub target: SimPos,
    pub speed: SimUnit,
}

/// Step every Move-bearing object toward its target. Returns the ids of
/// objects that arrived this tick — their `Move` has already been removed
/// by the time the call returns, so intent modules can call e.g.
/// [`super::chopping::on_arrival`] on the same slice.
///
/// Also faces the object along its velocity (local +X = "forward", rotating
/// around Z) via the integer atan2 in [`Facing::from_direction`] — bit-exact
/// across architectures, no `f32::atan2` involved.
pub fn advance(zone: &mut Zone, dt: SimUnit) -> Vec<WorldObjectId> {
    let mut arrived: Vec<WorldObjectId> = Vec::new();
    {
        let (mut objects, components) = zone.split_mut();
        for (pawn, m) in components.iter::<Move>() {
            let Some(transform) = objects.get_mut(pawn) else {
                continue;
            };
            let delta: SimVec = m.target - transform.position;
            let dist = delta.length();
            let step = m.speed * dt;
            // 1e-3 tile threshold mirrors the float version. Q16.16
            // resolution makes anything smaller indistinguishable from
            // zero anyway, so this is a clarity threshold, not a fudge.
            let epsilon = SimUnit::from_num(0.001);
            if dist <= step.max(epsilon) {
                transform.position = m.target;
                arrived.push(pawn);
            } else {
                // dir = delta / dist (Q16.16 / Q16.16). Direction
                // magnitudes are small, so no widening tricks needed —
                // the `fixed` crate's divide handles it.
                let dir = SimVec {
                    x: delta.x / dist,
                    y: delta.y / dist,
                    z: delta.z / dist,
                };
                transform.position += dir * step;
                transform.facing = Facing::from_direction(dir);
            }
        }
    }
    for &pawn in &arrived {
        zone.components_mut().remove::<Move>(pawn);
    }
    arrived
}
