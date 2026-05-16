//! Sim-side state for the lumber camp.
//!
//! Single zone, flat 16×16 square-grid terrain. The world holds three
//! pawns, five trees, and one stockpile. Pawns are autonomous: every tick,
//! any pawn without a [`Move`] looks for the nearest [`Designated`] tree and
//! walks to it; on arrival the tree is removed and the pawn becomes idle
//! again.
//!
//! The job-assignment policy here is deliberately the smallest thing that
//! exercises [`Move`] end-to-end: one pawn → one tree, no exclusion (two
//! pawns may race for the same tree), no path-finding, no chop timer, no
//! stump. A real [`JobBoard`] takes over when there's a second job kind to
//! coordinate.

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

/// Intent component naming the tree a pawn is walking to. When the pawn
/// arrives, the tree is removed (instant chop — `ChopProgress` will replace
/// this with a tick-counted timer). If the named tree disappears or is
/// un-designated mid-walk, the pawn drops both [`Move`] and `Chopping` and
/// returns to idle.
pub struct Chopping {
    pub tree: WorldObjectId,
}

/// Tile size in metres. One world unit per tile.
pub const TILE_SIZE: f32 = 1.0;
/// Height step in metres. The terrain is flat for the skeleton, so this only
/// affects the meshed slab thickness.
pub const HEIGHT_UNIT: f32 = 0.1;
/// Pawn walk speed. Tuned so a cross-map walk is several seconds at 1×
/// sim speed; numerically separate from any rendering value.
pub const PAWN_SPEED: f32 = 2.2;

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
        // designated. Necessary because two idle pawns can race for the same
        // tree, and because the user can un-designate mid-walk.
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

        // Phase 3 — arrival handling: drop Move; if the pawn was Chopping,
        // fell the named tree. `Zone::remove` cascades to every component on
        // the tree (including Designated, RenderId), so the view stops
        // drawing the tree and its marker on the next frame.
        for pawn in arrived {
            zone.components_mut().remove::<Move>(pawn);
            if let Some(chopping) = zone.components_mut().remove::<Chopping>(pawn) {
                zone.remove(chopping.tree);
            }
        }

        // Phase 4 — assign work to idle pawns. Idle = `RenderId::Pawn` with
        // no Move and no Chopping; assignment policy is "nearest designated
        // tree", duplicates allowed. The first arriving pawn wins the chop;
        // Phase 0 returns the loser to idle next tick.
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
            let Some(&(tree_id, tree_pos)) = designated.iter().min_by(|a, b| {
                let da = (a.1 - pawn_pos).length_squared();
                let db = (b.1 - pawn_pos).length_squared();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            }) else {
                continue;
            };
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
