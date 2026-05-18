//! Chopping behaviour: pawns walking to designated trees, the per-tree chop
//! progress, and the dispatch that hands idle pawns the nearest free
//! designation.
//!
//! Owns three components: the player-facing [`Designated`] marker, the
//! per-pawn [`Chopping`] intent (and its claim semantics), and the per-tree
//! [`ChopProgress`] countdown that runs once a chopper arrives.

use std::collections::HashSet;

use currawong::data::KindId;
use currawong::glam::Vec3;
use currawong::{WorldObjectId, Zone};

use super::{Carrying, GameStats, Move};

/// Marker on a tree the player has flagged for chopping. Carries no data
/// today; the planned `JobBoard` slice will grow this into `Designated { job:
/// JobId }` once jobs exist.
pub struct Designated;

/// Intent component naming the tree a pawn is walking to. Doubles as the
/// claim record: idle dispatch skips any tree already referenced by another
/// pawn's `Chopping`, so designations are taken by exactly one worker. On
/// arrival the pawn drops its [`Move`] but keeps `Chopping`; the tree gets
/// a [`ChopProgress`] that ticks down before it falls. If the named tree
/// disappears or is un-designated mid-walk, [`validate`] drops both `Move`
/// and `Chopping` and the pawn returns to idle.
pub struct Chopping {
    pub tree: WorldObjectId,
}

/// Countdown on a tree currently being chopped. Inserted on the tree (not
/// the pawn) when a pawn arrives, ticks down each sim tick, removes the tree
/// at zero. Lives only while a pawn is actively `Chopping` this tree —
/// [`validate`] clears orphans so progress doesn't persist into a
/// re-designation.
pub struct ChopProgress {
    pub ticks_remaining: u32,
}

/// Invalidate any `Chopping` whose tree is gone or no longer designated,
/// then clean up any `ChopProgress` whose chopper bailed (un-designation,
/// removed pawn). Runs first each tick so downstream systems see a
/// consistent state.
pub fn validate(zone: &mut Zone) {
    let mut cancel: Vec<WorldObjectId> = Vec::new();
    for (pawn, chopping) in zone.components().iter::<Chopping>() {
        let tree_alive = zone.contains(chopping.tree);
        let still_designated = zone.components().get::<Designated>(chopping.tree).is_some();
        if !tree_alive || !still_designated {
            cancel.push(pawn);
        }
    }
    for pawn in cancel {
        zone.components_mut().remove::<Chopping>(pawn);
        zone.components_mut().remove::<Move>(pawn);
    }
    let active_choppers: HashSet<WorldObjectId> = zone
        .components()
        .iter::<Chopping>()
        .map(|(_, c)| c.tree)
        .collect();
    let orphan_progress: Vec<WorldObjectId> = zone
        .components()
        .iter::<ChopProgress>()
        .filter_map(|(tree, _)| (!active_choppers.contains(&tree)).then_some(tree))
        .collect();
    for tree in orphan_progress {
        zone.components_mut().remove::<ChopProgress>(tree);
    }
}

/// Refresh `Move.target` from the current `Chopping` target's position.
/// Trees are static today but this future-proofs against them moving
/// (wind sway) without changing the arrival check.
pub fn refresh_move_targets(zone: &mut Zone) {
    let mut refresh: Vec<(WorldObjectId, Vec3)> = Vec::new();
    for (pawn, c) in zone.components().iter::<Chopping>() {
        if let Some(t) = zone.get(c.tree) {
            refresh.push((pawn, t.position));
        }
    }
    for (pawn, target) in refresh {
        if let Some(m) = zone.components_mut().get_mut::<Move>(pawn) {
            m.target = target;
        }
    }
}

/// React to arrivals: a pawn that just reached its chopping target starts
/// a `ChopProgress` on the tree (if not already counting). The pawn stays
/// stationary at the tree until [`tick_progress`] fells it.
///
/// `chop_ticks` is read off the tree's [`KindId`] component via
/// [`GameStats::tree_stats`], so pine takes longer than oak per its
/// `pine_tree.ron` body.
pub fn on_arrival(zone: &mut Zone, arrived: &[WorldObjectId], stats: &GameStats) {
    for &pawn in arrived {
        let Some(tree) = zone.components().get::<Chopping>(pawn).map(|c| c.tree) else {
            continue;
        };
        if zone.components().get::<ChopProgress>(tree).is_some() {
            continue;
        }
        let Some(tree_kind) = zone.components().get::<KindId>(tree) else {
            continue;
        };
        let Some(tree_stats) = stats.tree_stats(tree_kind) else {
            // Defensive — designations are limited to tree kinds, but if a
            // user-defined kind sneaks through dispatch we don't want to
            // panic mid-tick.
            continue;
        };
        let ticks_remaining = tree_stats.chop_ticks;
        zone.components_mut()
            .insert(tree, ChopProgress { ticks_remaining });
    }
}

/// Tick down every `ChopProgress`; fell trees at zero. `Zone::remove`
/// cascades through every component on the tree, so the view stops drawing
/// the tree and its marker next frame. Before removing the tree, promote
/// its chopper(s) from `Chopping` to `Carrying` — that's the "you got a log"
/// signal [`super::hauling::dispatch_carrying`] picks up.
pub fn tick_progress(zone: &mut Zone) {
    let mut felled: Vec<WorldObjectId> = Vec::new();
    for (tree, progress) in zone.components_mut().iter_mut::<ChopProgress>() {
        progress.ticks_remaining = progress.ticks_remaining.saturating_sub(1);
        if progress.ticks_remaining == 0 {
            felled.push(tree);
        }
    }
    for tree in felled {
        let choppers: Vec<WorldObjectId> = zone
            .components()
            .iter::<Chopping>()
            .filter_map(|(pawn, c)| (c.tree == tree).then_some(pawn))
            .collect();
        for pawn in choppers {
            zone.components_mut().remove::<Chopping>(pawn);
            zone.components_mut().insert(pawn, Carrying);
        }
        zone.remove(tree);
    }
}

/// Hand idle pawns (no `Move`, no `Chopping`, no `Carrying`) the nearest
/// unclaimed `Designated` tree. Claim semantics on trees prevent two pawns
/// racing the same designation.
///
/// "Pawn" is anything carrying the `lumberjack` kind id — extending to
/// other worker kinds means adding that kind id to a small allow-list (or
/// switching to a `Worker` marker component once a second worker kind
/// exists).
pub fn dispatch_idle(zone: &mut Zone, stats: &GameStats) {
    let lumberjack = &stats.kinds.lumberjack;
    let designated: Vec<(WorldObjectId, Vec3)> = zone
        .components()
        .iter::<Designated>()
        .filter_map(|(id, _)| zone.get(id).map(|t| (id, t.position)))
        .collect();
    if designated.is_empty() {
        return;
    }
    let mut claimed: HashSet<WorldObjectId> = zone
        .components()
        .iter::<Chopping>()
        .map(|(_, c)| c.tree)
        .collect();
    let idle: Vec<(WorldObjectId, Vec3)> = zone
        .iter()
        .filter(|(id, _)| {
            zone.components().get::<KindId>(*id) == Some(lumberjack)
                && zone.components().get::<Move>(*id).is_none()
                && zone.components().get::<Chopping>(*id).is_none()
                && zone.components().get::<Carrying>(*id).is_none()
        })
        .map(|(id, t)| (id, t.position))
        .collect();
    for (pawn, pawn_pos) in idle {
        let Some(&(tree_id, tree_pos)) = designated
            .iter()
            .filter(|(tree, _)| !claimed.contains(tree))
            .min_by(|a, b| {
                let da = (a.1 - pawn_pos).length_squared();
                let db = (b.1 - pawn_pos).length_squared();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
        else {
            continue;
        };
        claimed.insert(tree_id);
        zone.components_mut()
            .insert(pawn, Chopping { tree: tree_id });
        zone.components_mut().insert(
            pawn,
            Move {
                target: tree_pos,
                speed: stats.lumberjack_speed,
            },
        );
    }
}
