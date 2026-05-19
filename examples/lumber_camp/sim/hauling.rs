//! Hauling behaviour: Carrying pawns are routed to a stockpile, deposit
//! their log on arrival, and return to idle for the next chop assignment.
//!
//! Owns three components: the [`Carrying`] marker (set when a tree falls,
//! cleared on deposit), the per-pawn [`Hauling`] intent (which stockpile),
//! and the per-stockpile [`WoodStored`] count.

use currawong::{SimPos, SimUnit, WorldObjectId, Zone};

use super::{GameStats, Idle, Move};

/// Marker on a pawn that finished a chop and is hauling a single log back
/// to the stockpile. Zero-sized today; carrying capacity grows when the
/// game wants stack sizes (multiple logs per trip).
pub struct Carrying;

/// Intent component naming the stockpile a Carrying pawn is delivering to.
/// On arrival the log is deposited (incrementing [`WoodStored`]) and the
/// pawn drops both `Carrying` and `Hauling`, returning to idle.
pub struct Hauling {
    pub stockpile: WorldObjectId,
}

/// Accumulated logs delivered to a stockpile. Updated by [`on_arrival`];
/// read by the HUD and the win-condition check at end of tick.
pub struct WoodStored {
    pub count: u32,
}

/// Refresh `Move.target` from the current `Hauling` target's position.
/// Stockpiles are static today but this future-proofs against mobile depots.
pub fn refresh_move_targets(zone: &mut Zone) {
    let mut refresh: Vec<(WorldObjectId, SimPos)> = Vec::new();
    for (pawn, h) in zone.components().iter::<Hauling>() {
        if let Some(t) = zone.get(h.stockpile) {
            refresh.push((pawn, t.position));
        }
    }
    for (pawn, target) in refresh {
        if let Some(m) = zone.components_mut().get_mut::<Move>(pawn) {
            m.target = target;
        }
    }
}

/// React to arrivals: a pawn that just reached its stockpile deposits its
/// log, drops `Carrying` + `Hauling`, and becomes idle next tick.
pub fn on_arrival(zone: &mut Zone, arrived: &[WorldObjectId]) {
    for &pawn in arrived {
        let Some(stockpile) = zone.components().get::<Hauling>(pawn).map(|h| h.stockpile) else {
            continue;
        };
        zone.components_mut().remove::<Hauling>(pawn);
        zone.components_mut().remove::<Carrying>(pawn);
        zone.components_mut().insert(
            pawn,
            Idle {
                seconds: SimUnit::ZERO,
            },
        );
        if let Some(store) = zone.components_mut().get_mut::<WoodStored>(stockpile) {
            store.count += 1;
        }
    }
}

/// Route `Carrying` pawns with no `Move`/`Hauling` to the given stockpile.
/// Newly-felled choppers (promoted to `Carrying` in
/// [`super::chopping::tick_progress`]) land here on the same tick so the
/// round trip starts immediately. Walk speed comes from
/// [`GameStats::lumberjack_speed`] — the same speed `dispatch_idle` stamps
/// onto the outbound trip.
pub fn dispatch_carrying(zone: &mut Zone, stockpile: WorldObjectId, stats: &GameStats) {
    let Some(stockpile_pos) = zone.get(stockpile).map(|t| t.position) else {
        return;
    };
    let to_haul: Vec<WorldObjectId> = zone
        .components()
        .iter::<Carrying>()
        .filter_map(|(pawn, _)| {
            let has_move = zone.components().get::<Move>(pawn).is_some();
            let has_haul = zone.components().get::<Hauling>(pawn).is_some();
            (!has_move && !has_haul).then_some(pawn)
        })
        .collect();
    for pawn in to_haul {
        zone.components_mut().insert(pawn, Hauling { stockpile });
        zone.components_mut().insert(
            pawn,
            Move {
                target: stockpile_pos,
                speed: stats.lumberjack_speed,
            },
        );
    }
}
