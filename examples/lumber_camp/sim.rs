//! Sim-side state for the lumber camp.
//!
//! Single zone, flat 16×16 square-grid terrain. The world holds three
//! pawns, five trees, and one stockpile. Pawns run a two-step loop:
//! 1. walk to nearest unclaimed `Designated` tree, chop it down;
//! 2. carry the resulting log back to the stockpile, drop it.
//!
//! `Chopping { tree }` and `Hauling { stockpile }` are the two "jobs"
//! the loop alternates between, plus the geometric [`Move`] component
//! that any in-flight pawn carries. Claim semantics on tree designations
//! live in `Chopping` (one tree = one chopper). A real `JobBoard`
//! generalises this when there's a third job kind to coordinate;
//! path-finding and stumps stay deferred.

use std::collections::HashSet;
use std::time::Duration;

use currawong::glam::{Quat, Vec3};
use currawong::{Simulation, TileCoord, WorldObjectId, WorldTransform, Zone, ZoneId, Zones};

/// Names a render template. Attached as a component on every sim object that
/// should appear in the world; the view resolves it to a concrete mesh +
/// material. Sim-side concept (it's how the sim labels its objects); the
/// view owns the actual GPU buffers behind each variant.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RenderId {
    Pawn,
    Tree,
    Stockpile,
}

/// Marker component on a tree the player has flagged for chopping. Carries no
/// data today; the `JobBoard` slice will grow this into `Designated { job:
/// JobId }` once jobs exist. Presence/absence is the toggle the click handler
/// manipulates and the view's marker draw reads.
pub struct Designated;

/// Geometric movement: advance the parent's transform toward `target` at
/// `speed` metres per second each tick, stopping on arrival (the component
/// is removed when the pawn closes the gap). Lives alongside whatever
/// component supplies the intent — [`Chopping`] today; later `Hauling`,
/// `Patrolling`, etc.
pub struct Move {
    pub target: Vec3,
    pub speed: f32,
}

/// Intent component naming the tree a pawn is walking to. Doubles as the
/// claim record: an idle pawn skips any tree already referenced by another
/// pawn's `Chopping`, so designations are taken by exactly one worker. On
/// arrival the pawn drops its [`Move`] but keeps `Chopping`; the tree gets
/// a [`ChopProgress`] that ticks down before it falls. If the named tree
/// disappears or is un-designated mid-walk, the pawn drops both [`Move`]
/// and `Chopping` and returns to idle.
pub struct Chopping {
    pub tree: WorldObjectId,
}

/// Countdown on a tree currently being chopped. Inserted on the tree (not
/// the pawn) when a pawn arrives, ticks down each sim tick, and removes the
/// tree at zero. Lives only while a pawn is actively `Chopping` this tree —
/// if the chopper bails (tree un-designated, pawn killed), Phase 0 clears
/// the orphan so progress doesn't persist into a re-designation.
pub struct ChopProgress {
    pub ticks_remaining: u32,
}

/// Marker on a pawn that finished a chop and is hauling a single log back
/// to the stockpile. Zero-sized today; carrying capacity grows when the
/// game wants stack sizes (multiple logs per trip).
pub struct Carrying;

/// Intent component naming the stockpile a Carrying pawn is delivering to.
/// On arrival, the log is deposited (incrementing [`WoodStored`]) and the
/// pawn drops both `Carrying` and `Hauling`, returning to idle.
pub struct Hauling {
    pub stockpile: WorldObjectId,
}

/// Accumulated logs delivered to a stockpile. Updated by Phase 3 (arrival)
/// when a `Hauling` pawn reaches its target; read by the HUD and by the
/// win-condition check at end of tick.
pub struct WoodStored {
    pub count: u32,
}

/// Overall game state. The sim ticks gameplay only while `Playing`; on
/// `Won`/`Lost` the world freezes in place and the HUD shows a banner.
/// No in-game restart for v1 — quit and rerun to play again.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameState {
    Playing,
    Won,
    Lost,
}

/// Tile size in metres. One world unit per tile.
pub const TILE_SIZE: f32 = 1.0;
/// Height step in metres. The terrain is flat for the skeleton, so this only
/// affects the meshed slab thickness.
pub const HEIGHT_UNIT: f32 = 0.1;
/// Pawn walk speed. Tuned so a cross-map walk is several seconds at 1×
/// sim speed; numerically separate from any rendering value.
pub const PAWN_SPEED: f32 = 2.2;
/// Ticks of `ChopProgress` per tree at default 60 Hz — 1.5 seconds of
/// "chopping" once a pawn reaches the tree.
pub const CHOP_TICKS: u32 = 90;
/// Logs the player needs to deliver to win. Five trees on the map; need
/// to clear most of them.
pub const WOOD_GOAL: u32 = 5;
/// Wall-time budget (sim seconds — same thing at 1× speed) before the
/// game is lost. Tuned so a fresh player who clicks all trees quickly
/// can win with a few seconds to spare.
pub const TIME_LIMIT_SECS: f32 = 60.0;

// Visual extents (used here only to compute the resting Z so the bottom of
// each primitive sits on the ground). The view picks the same numbers when
// building its meshes.
const PAWN_HEIGHT: f32 = 1.6;
const TREE_HEIGHT: f32 = 2.0;
const STOCKPILE_SIZE: f32 = 1.0;

/// Square-grid extent. Tiles span `[-HALF_EXTENT, HALF_EXTENT)` along X and Y.
const HALF_EXTENT: i32 = 8;

pub struct Game {
    pub zones: Zones,
    pub zone: ZoneId,
    /// The (only) stockpile in this PoC. Cached so Phase 5 can route
    /// `Hauling` jobs at it without walking the zone every tick.
    pub stockpile: WorldObjectId,
    /// Game-state machine. Gameplay phases run only while `Playing`; the
    /// HUD reads `Won`/`Lost` to swap in the end-of-game banner.
    pub state: GameState,
    /// Sim-seconds elapsed since the run started — drives the time-limit
    /// check and the HUD's countdown. Only advances while `Playing`.
    pub elapsed: f32,
}

impl Game {
    pub fn new() -> Self {
        let mut zones = Zones::new();
        let zone_id = zones.insert(Zone::new());
        let zone = zones.get_mut(zone_id).expect("just inserted");

        // Flat ground over the playable area. Tiles default to floor_height = 0
        // and walkable, so we only need to touch them to allocate the cells.
        let terrain = zone.terrain_mut();
        for ty in -HALF_EXTENT..HALF_EXTENT {
            for tx in -HALF_EXTENT..HALF_EXTENT {
                terrain.tile_mut(TileCoord::new(tx, ty)).floor_height = 0;
            }
        }

        // Stockpile in the +X +Y corner. Centre-of-cube sits half a unit
        // above the ground so the cube rests on it.
        let stockpile = zone.insert(WorldTransform {
            position: Vec3::new(6.0, 6.0, STOCKPILE_SIZE * 0.5),
            rotation: Quat::IDENTITY,
        });
        zone.components_mut().insert(stockpile, RenderId::Stockpile);
        zone.components_mut()
            .insert(stockpile, WoodStored { count: 0 });

        // Three pawns loitering near the stockpile. Capsule pivot is at its
        // centre so the bottom hemisphere clears the ground at z = height/2.
        for &(x, y) in &[(4.5, 5.5), (5.5, 4.5), (4.5, 6.5)] {
            let pawn = zone.insert(WorldTransform {
                position: Vec3::new(x, y, PAWN_HEIGHT * 0.5),
                rotation: Quat::IDENTITY,
            });
            zone.components_mut().insert(pawn, RenderId::Pawn);
        }

        // Five trees scattered across the -X / -Y half of the map.
        for &(x, y) in &[
            (-5.5, -4.5),
            (-3.0, -6.0),
            (-1.0, -3.5),
            (-6.5, -1.5),
            (1.5, -5.5),
        ] {
            let tree = zone.insert(WorldTransform {
                position: Vec3::new(x, y, TREE_HEIGHT * 0.5),
                rotation: Quat::IDENTITY,
            });
            zone.components_mut().insert(tree, RenderId::Tree);
        }

        Self {
            zones,
            zone: zone_id,
            stockpile,
            state: GameState::Playing,
            elapsed: 0.0,
        }
    }

    /// Logs delivered so far — read by the HUD and the win check. Returns 0
    /// if the stockpile was somehow destroyed (can't happen today, but the
    /// stockpile is a `WorldObjectId` like any other and could be removed
    /// by a future feature).
    pub fn wood_count(&self) -> u32 {
        self.zones
            .get(self.zone)
            .and_then(|z| z.components().get::<WoodStored>(self.stockpile))
            .map(|s| s.count)
            .unwrap_or(0)
    }
}

impl Simulation for Game {
    fn tick(&mut self, dt: Duration) {
        if self.state != GameState::Playing {
            // Freeze gameplay on win/lose: no pawn motion, no chop ticks,
            // no timer advancement. The HUD keeps drawing the banner.
            return;
        }
        let dt = dt.as_secs_f32();
        self.elapsed += dt;
        let Some(zone) = self.zones.get_mut(self.zone) else {
            return;
        };

        // Phase 0 — invalidate any Chopping whose tree is gone or no longer
        // designated, then clean up any ChopProgress whose chopper bailed
        // (un-designation, removed pawn). Necessary because the user can
        // un-designate mid-chop and we don't want stale progress carrying
        // over into a future re-designation.
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

        // Phase 1 — refresh Move.target from the current `Chopping`/`Hauling`
        // target's position. Trees and stockpile are static today but this
        // future-proofs against either moving (wind sway, mobile depots)
        // without changing the arrival check.
        let mut refresh: Vec<(WorldObjectId, Vec3)> = Vec::new();
        for (pawn, c) in zone.components().iter::<Chopping>() {
            if let Some(t) = zone.get(c.tree) {
                refresh.push((pawn, t.position));
            }
        }
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

        // Phase 2 — advance every Move toward target; collect arrivals. Also
        // face the pawn along its velocity: local +X = "forward" in the
        // pawn's frame (rotating around Z, the up axis). Capsule pawns are
        // rotationally symmetric so this isn't visible yet, but the per-
        // instance rotation flows through `WorldTransform.rotation` into the
        // pawn template's model matrix end-to-end — a non-symmetric pawn
        // mesh will reveal it without further plumbing.
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

        // Phase 3 — arrival handling. Drop Move (the pawn is stationary
        // now), then dispatch on whichever intent component the pawn was
        // carrying:
        // - `Chopping` → start ChopProgress on the tree if not already
        //   counting; pawn stays here until Phase 4 fells the tree.
        // - `Hauling`  → deposit the log: drop Carrying + Hauling and
        //   increment the stockpile's WoodStored. Pawn is idle next tick.
        for pawn in arrived {
            zone.components_mut().remove::<Move>(pawn);
            let chop_tree = zone.components().get::<Chopping>(pawn).map(|c| c.tree);
            if let Some(tree) = chop_tree
                && zone.components().get::<ChopProgress>(tree).is_none()
            {
                zone.components_mut().insert(
                    tree,
                    ChopProgress {
                        ticks_remaining: CHOP_TICKS,
                    },
                );
            }
            let haul_stockpile = zone.components().get::<Hauling>(pawn).map(|h| h.stockpile);
            if let Some(stockpile) = haul_stockpile {
                zone.components_mut().remove::<Hauling>(pawn);
                zone.components_mut().remove::<Carrying>(pawn);
                if let Some(store) = zone.components_mut().get_mut::<WoodStored>(stockpile) {
                    store.count += 1;
                }
            }
        }

        // Phase 4 — tick down every ChopProgress; fell trees that reach
        // zero. `Zone::remove` cascades through every component on the tree
        // (Designated, RenderId, ChopProgress), so the view stops drawing
        // the tree and its marker on the next frame. Before removing the
        // tree, promote its chopper(s) from `Chopping` to `Carrying` —
        // that's the "you got a log" signal the next phase routes to the
        // stockpile.
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

        // Phase 5 — assign work. Two queues, in order:
        // (a) Carrying pawns with no Move/Hauling → route to the stockpile
        //     (newly-felled choppers from Phase 4 land here on the same
        //     tick they finish, so the round trip starts immediately).
        // (b) Truly idle pawns (no Move, no Chopping, no Carrying) → nearest
        //     unclaimed `Designated` tree. Claim semantics on trees prevent
        //     two pawns racing the same designation; stockpiles have no
        //     analogous claim (one stockpile, unbounded capacity).
        let stockpile_pos = self
            .zones
            .get(self.zone)
            .and_then(|z| z.get(self.stockpile))
            .map(|t| t.position);
        let zone = self.zones.get_mut(self.zone).expect("zone exists");
        if let Some(stockpile_pos) = stockpile_pos {
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
                zone.components_mut().insert(
                    pawn,
                    Hauling {
                        stockpile: self.stockpile,
                    },
                );
                zone.components_mut().insert(
                    pawn,
                    Move {
                        target: stockpile_pos,
                        speed: PAWN_SPEED,
                    },
                );
            }
        }

        let designated: Vec<(WorldObjectId, Vec3)> = zone
            .components()
            .iter::<Designated>()
            .filter_map(|(id, _)| zone.get(id).map(|t| (id, t.position)))
            .collect();
        if !designated.is_empty() {
            let mut claimed: HashSet<WorldObjectId> = zone
                .components()
                .iter::<Chopping>()
                .map(|(_, c)| c.tree)
                .collect();
            let idle: Vec<(WorldObjectId, Vec3)> = zone
                .iter()
                .filter(|(id, _)| {
                    zone.components().get::<RenderId>(*id) == Some(&RenderId::Pawn)
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
                        speed: PAWN_SPEED,
                    },
                );
            }
        }

        // Phase 6 — win/lose check. Win has priority: if a delivery during
        // this same tick crossed the goal, that beats the timer expiring on
        // the same tick. State transition freezes the world on the next
        // call to `tick` via the early-return at the top.
        if self.wood_count() >= WOOD_GOAL {
            self.state = GameState::Won;
        } else if self.elapsed >= TIME_LIMIT_SECS {
            self.state = GameState::Lost;
        }
    }
}
