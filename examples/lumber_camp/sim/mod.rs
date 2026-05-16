//! Sim-side state for the lumber camp.
//!
//! Each behaviour lives in its own component-keyed submodule:
//! - [`motion`] — geometric movement ([`Move`] + advance)
//! - [`chopping`] — designations, chopping intent, per-tree progress
//! - [`hauling`] — Carrying marker, Hauling intent, stockpile deposit
//!
//! [`Game`] is the thin orchestrator: it owns the [`Zones`], the cached
//! stockpile id, and the win/lose state, and its `tick` is an ordered call
//! list into the submodule systems. The order graph lives in one place so
//! it's the only thing that has to know who-runs-before-whom.

pub mod chopping;
pub mod hauling;
pub mod motion;

use std::time::Duration;

use currawong::glam::{Quat, Vec3};
use currawong::{Simulation, TileCoord, WorldObjectId, WorldTransform, Zone, ZoneId, Zones};

// Re-export only what the view consumes today; everything else stays
// behind its submodule path. Adding to this list is deliberate, not
// automatic.
pub use chopping::Designated;
pub use hauling::{Carrying, WoodStored};
pub use motion::Move;

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
/// Pawn walk speed. Tuned so a cross-map walk is several seconds at 1× sim
/// speed. Stays in the orchestrator because both intent modules
/// ([`chopping`], [`hauling`]) read it when they create a [`Move`] — hoist
/// rather than duplicate.
pub const PAWN_SPEED: f32 = 2.2;
/// Logs the player needs to deliver to win. Five trees on the map; need to
/// clear most of them.
pub const WOOD_GOAL: u32 = 5;
/// Wall-time budget (sim seconds — same thing at 1× speed) before the game
/// is lost.
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
    /// The (only) stockpile in this PoC. Cached so
    /// [`hauling::dispatch_carrying`] can route Carrying pawns without
    /// walking the zone every tick.
    pub stockpile: WorldObjectId,
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

        // Flat ground over the playable area.
        let terrain = zone.terrain_mut();
        for ty in -HALF_EXTENT..HALF_EXTENT {
            for tx in -HALF_EXTENT..HALF_EXTENT {
                terrain.tile_mut(TileCoord::new(tx, ty)).floor_height = 0;
            }
        }

        // Stockpile in the +X +Y corner.
        let stockpile = zone.insert(WorldTransform {
            position: Vec3::new(6.0, 6.0, STOCKPILE_SIZE * 0.5),
            rotation: Quat::IDENTITY,
        });
        zone.components_mut().insert(stockpile, RenderId::Stockpile);
        zone.components_mut()
            .insert(stockpile, WoodStored { count: 0 });

        // Three pawns loitering near the stockpile.
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
    /// if the stockpile was somehow destroyed.
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

        // The order graph. Reading top-to-bottom: cancel stale work,
        // refresh in-flight targets, step motion (collecting arrivals),
        // dispatch arrival side-effects per intent kind, tick chop
        // countdowns, then assign new work to the now-idle.
        chopping::validate(zone);
        chopping::refresh_move_targets(zone);
        hauling::refresh_move_targets(zone);
        let arrived = motion::advance(zone, dt);
        chopping::on_arrival(zone, &arrived);
        hauling::on_arrival(zone, &arrived);
        chopping::tick_progress(zone);
        hauling::dispatch_carrying(zone, self.stockpile);
        chopping::dispatch_idle(zone);

        // Win has priority: if a delivery during this same tick crossed
        // the goal, that beats the timer expiring on the same tick. The
        // state transition freezes the world on the next call via the
        // early-return at the top.
        if self.wood_count() >= WOOD_GOAL {
            self.state = GameState::Won;
        } else if self.elapsed >= TIME_LIMIT_SECS {
            self.state = GameState::Lost;
        }
    }
}
