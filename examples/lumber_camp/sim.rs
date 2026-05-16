//! Sim-side state for the lumber camp.
//!
//! Single zone, flat 16×16 square-grid terrain. The world holds three
//! pawns, five trees, and one stockpile. Pawns are autonomous: every tick,
//! any pawn without a [`Move`] looks for the nearest [`Designated`] tree and
//! walks to it; on arrival the tree is removed and the pawn becomes idle
//! again.
//!
//! The job-assignment policy is "idle pawn pulls the nearest unclaimed
//! designated tree". Claim semantics live in the `Chopping` component:
//! a tree referenced by any pawn's `Chopping` is off-limits to other
//! idlers. A real [`JobBoard`] generalises this when there's a second job
//! kind to coordinate. Path-finding, chop timer, and stumps are also
//! deferred.

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
        }
    }
}

impl Simulation for Game {
    fn tick(&mut self, dt: Duration) {
        let dt = dt.as_secs_f32();
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

        // Phase 1 — refresh Move.target from the chopping tree's current
        // position. Trees are static today but this future-proofs against
        // tree-moving (e.g. wind sway pulling collisions) without changing
        // the arrival check.
        let refresh: Vec<(WorldObjectId, Vec3)> = zone
            .components()
            .iter::<Chopping>()
            .filter_map(|(pawn, c)| zone.get(c.tree).map(|t| (pawn, t.position)))
            .collect();
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

        // Phase 3 — arrival handling: drop Move (the pawn is now stationed
        // at the tree) but keep Chopping (the claim) and start ChopProgress
        // on the tree if it isn't already counting. The pawn remains "at
        // work" — not idle — until the tree falls in Phase 4.
        for pawn in arrived {
            zone.components_mut().remove::<Move>(pawn);
            let tree = zone.components().get::<Chopping>(pawn).map(|c| c.tree);
            if let Some(tree) = tree
                && zone.components().get::<ChopProgress>(tree).is_none()
            {
                zone.components_mut().insert(
                    tree,
                    ChopProgress {
                        ticks_remaining: CHOP_TICKS,
                    },
                );
            }
        }

        // Phase 4 — tick down every ChopProgress; fell trees that reach
        // zero. `Zone::remove` cascades through every component on the tree
        // (Designated, RenderId, ChopProgress), so the view stops drawing
        // the tree and its marker on the next frame. The pawn keeps its
        // Chopping for one more tick, then Phase 0 invalidates it and the
        // pawn returns to idle.
        let mut felled: Vec<WorldObjectId> = Vec::new();
        for (tree, progress) in zone.components_mut().iter_mut::<ChopProgress>() {
            progress.ticks_remaining = progress.ticks_remaining.saturating_sub(1);
            if progress.ticks_remaining == 0 {
                felled.push(tree);
            }
        }
        for tree in felled {
            zone.remove(tree);
        }

        // Phase 5 — assign work to idle pawns. Idle = `RenderId::Pawn` with
        // no Move and no Chopping; assignment policy is "nearest *unclaimed*
        // designated tree". A tree is claimed if any pawn's `Chopping`
        // references it (either from a prior tick, or assigned earlier in
        // this same pass).
        let mut claimed: HashSet<WorldObjectId> = zone
            .components()
            .iter::<Chopping>()
            .map(|(_, c)| c.tree)
            .collect();
        let designated: Vec<(WorldObjectId, Vec3)> = zone
            .components()
            .iter::<Designated>()
            .filter_map(|(id, _)| zone.get(id).map(|t| (id, t.position)))
            .collect();
        if designated.is_empty() {
            return;
        }
        let idle: Vec<(WorldObjectId, Vec3)> = zone
            .iter()
            .filter(|(id, _)| {
                zone.components().get::<RenderId>(*id) == Some(&RenderId::Pawn)
                    && zone.components().get::<Move>(*id).is_none()
                    && zone.components().get::<Chopping>(*id).is_none()
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
}
