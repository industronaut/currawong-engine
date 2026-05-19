//! Idle marker: every pawn carries an [`Idle`] component whenever it has no
//! active job. The other intent modules ([`super::chopping`],
//! [`super::hauling`]) own the insert/remove edges — chopping removes `Idle`
//! when it assigns a new chop and re-adds it when a chop is cancelled;
//! hauling re-adds it when a delivery completes. This module just owns the
//! type and ticks the counter.

use currawong::{SimUnit, Zone};

/// Component on a pawn with no job. `seconds` resets to 0 each time the
/// pawn becomes idle and increments by `dt` every tick it remains idle.
pub struct Idle {
    pub seconds: SimUnit,
}

/// Increment every idle pawn's counter by `dt` sim-seconds.
pub fn tick(zone: &mut Zone, dt: SimUnit) {
    for (_, idle) in zone.components_mut().iter_mut::<Idle>() {
        idle.seconds += dt;
    }
}
