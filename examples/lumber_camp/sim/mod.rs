//! Sim-side state for the lumber camp.
//!
//! Each behaviour lives in its own component-keyed submodule:
//! - [`motion`] — geometric movement ([`Move`] + advance)
//! - [`chopping`] — designations, chopping intent, per-tree progress
//! - [`hauling`] — Carrying marker, Hauling intent, stockpile deposit
//!
//! [`Game`] is the thin orchestrator: it owns the [`Zones`], the cached
//! stockpile id, the win/lose state, and the resolved [`GameStats`] read
//! out of the [`Definitions`] at construction. Its `tick` is an ordered
//! call list into the submodule systems. The order graph lives in one
//! place so it's the only thing that has to know who-runs-before-whom.
//!
//! ## Kind identification
//!
//! Sim objects carry a [`KindId`] component (namespaced string —
//! `currawong:oak_tree`, `currawong:lumberjack`, …). Comparison against the
//! cached `stats.kinds` ids is what distinguishes "this is a pawn" from
//! "this is a tree" — no engine enum, no view leakage. The numeric
//! per-species stats (chop ticks, walk speed) are pulled out of the def
//! files at construction so the per-tick path is a plain `&GameStats`
//! lookup rather than a re-deserialise.

pub mod chopping;
pub mod hauling;
pub mod idle;
pub mod motion;

use std::time::Duration;

use currawong::data::{Definitions, KindId};
use currawong::{
    Facing, SimEnvironment, SimPos, SimUnit, Simulation, TileCoord, WorldObjectId, WorldObjectRef,
    WorldTransform, Zone, ZoneId, Zones,
};
use serde::Deserialize;

// Re-export only what the view consumes today; everything else stays
// behind its submodule path. Adding to this list is deliberate, not
// automatic.
pub use chopping::Designated;
pub use hauling::{Carrying, WoodStored};
pub use idle::Idle;
pub use motion::Move;

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
/// Logs the player needs to deliver to win. Five trees on the map; need to
/// clear most of them.
pub const WOOD_GOAL: u32 = 5;
/// Wall-time budget (sim seconds — same thing at 1× speed) before the game
/// is lost. Stored as a plain integer since the value has no fractional
/// part; compared against [`Game::elapsed`] (Q16.16 sim-seconds).
pub const TIME_LIMIT_SECS: i32 = 60;

/// Square-grid extent. Tiles span `[-HALF_EXTENT, HALF_EXTENT)` along X and Y.
const HALF_EXTENT: i32 = 8;

/// Resolved kind ids — pulled out of the defs once at startup so per-tick
/// comparisons don't have to re-parse strings.
pub struct KindIds {
    pub oak_tree: KindId,
    pub pine_tree: KindId,
    pub lumber_camp: KindId,
    pub lumberjack: KindId,
}

/// Per-tree-species sim stats. Read out of each tree def's body fields.
pub struct TreeStats {
    /// Sim ticks (default 60 Hz) of [`chopping::ChopProgress`] before the
    /// tree falls. Per-species so pine is meaningfully different from oak.
    pub chop_ticks: u32,
}

/// Resolved sim-side stats for every kind the lumber camp uses. Held on
/// [`Game::stats`]; systems read what they need by reference.
pub struct GameStats {
    pub kinds: KindIds,
    pub oak: TreeStats,
    pub pine: TreeStats,
    /// Walk speed in metres per sim-second for [`KindIds::lumberjack`].
    /// Stamped into every [`Move`] component at insertion time. Defs
    /// authored as floats (RON files speak `f32`); converted to fixed-
    /// point once at load.
    pub lumberjack_speed: SimUnit,
}

impl GameStats {
    /// Pull the expected lumber-camp kinds out of `defs` and bind their
    /// stats. Panics with a kind-id-bearing message when a required def is
    /// missing — the example's content is in the repo, so a missing kind is
    /// a deployment/build error, not a runtime condition we should swallow.
    pub fn from_definitions(defs: &Definitions) -> Self {
        let oak_id = kind("currawong:oak_tree");
        let pine_id = kind("currawong:pine_tree");
        let camp_id = kind("currawong:lumber_camp");
        let jack_id = kind("currawong:lumberjack");

        let oak: TreeBody = body(defs, &oak_id);
        let pine: TreeBody = body(defs, &pine_id);
        // Confirm the building exists; we don't read any sim-side stats off
        // it (the wood-stored count + win goal live on the example, not on
        // the def). The lookup still surfaces typos like a missing
        // `lumber_camp.ron` at startup instead of mid-run.
        let _: LumberCampBody = body(defs, &camp_id);
        let walker: WalkerBody = body(defs, &jack_id);

        Self {
            kinds: KindIds {
                oak_tree: oak_id,
                pine_tree: pine_id,
                lumber_camp: camp_id,
                lumberjack: jack_id,
            },
            oak: TreeStats {
                chop_ticks: oak.chop_ticks,
            },
            pine: TreeStats {
                chop_ticks: pine.chop_ticks,
            },
            lumberjack_speed: SimUnit::from_num(walker.walk_speed),
        }
    }

    /// Look up the [`TreeStats`] for a kind id. Returns `None` if the id
    /// isn't a registered tree species — callers handling already-validated
    /// objects can `.unwrap_or_else(...)` to a panic; speculative callers
    /// (e.g. dispatching on whatever kind happens to be designated) can
    /// gracefully bail.
    pub fn tree_stats(&self, kind: &KindId) -> Option<&TreeStats> {
        if kind == &self.kinds.oak_tree {
            Some(&self.oak)
        } else if kind == &self.kinds.pine_tree {
            Some(&self.pine)
        } else {
            None
        }
    }
}

fn kind(s: &'static str) -> KindId {
    KindId::new(s).expect("hard-coded kind id is well-formed")
}

fn body<T: serde::de::DeserializeOwned>(defs: &Definitions, id: &KindId) -> T {
    let def = defs
        .get(id)
        .unwrap_or_else(|| panic!("missing required kind definition: {id}"));
    def.value
        .clone()
        .into_rust::<T>()
        .unwrap_or_else(|e| panic!("kind {id} does not match expected body: {e}"))
}

#[derive(Deserialize)]
struct TreeBody {
    chop_ticks: u32,
}

#[derive(Deserialize)]
struct WalkerBody {
    walk_speed: f32,
}

/// The lumber camp's body shape today is just the `render` block (sim has
/// nothing tunable about it yet); structurally validating the def is enough
/// to surface typos at startup.
#[derive(Deserialize)]
struct LumberCampBody {}

pub struct Game {
    pub zones: Zones,
    pub zone: ZoneId,
    /// The (only) stockpile in this PoC. Cached so
    /// [`hauling::dispatch_carrying`] can route Carrying pawns without
    /// walking the zone every tick.
    pub stockpile: WorldObjectId,
    pub state: GameState,
    /// Sim-seconds elapsed since the run started — drives the time-limit
    /// check and the HUD's countdown. Only advances while `Playing`. Q16.16
    /// for determinism; the HUD converts to `f32` at render time.
    pub elapsed: SimUnit,
    /// Resolved kind ids + per-species stats from the [`Definitions`].
    pub stats: GameStats,
    /// Sim-side environment — drives the moving sun the view extracts each
    /// frame. Advances every tick (also while `Won`/`Lost` so the world
    /// keeps animating under the game-over banner). Day length is tuned to
    /// match the time limit so the sun visibly travels during one run.
    pub env: SimEnvironment,
}

impl Game {
    pub fn new(defs: Definitions) -> Self {
        let stats = GameStats::from_definitions(&defs);

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

        // Lumber camp building in the +X +Y corner. Origin-at-base on the
        // glTF means Z = 0 is the ground.
        let stockpile = zone.insert(WorldTransform {
            position: SimPos::tile(6, 6, 0),
            facing: Facing::ZERO,
        });
        zone.components_mut()
            .insert(stockpile, stats.kinds.lumber_camp.clone());
        zone.components_mut()
            .insert(stockpile, WoodStored { count: 0 });

        // Three lumberjacks loitering near the camp. Tile centres
        // (`x.5`, `y.5`) — half-tile offset is the example's convention.
        for &(x, y) in &[(4, 5), (5, 4), (4, 6)] {
            let pawn = zone.insert(WorldTransform {
                position: SimPos {
                    x: SimUnit::from_num(x) + SimUnit::from_num(0.5),
                    y: SimUnit::from_num(y) + SimUnit::from_num(0.5),
                    z: SimUnit::ZERO,
                },
                facing: Facing::ZERO,
            });
            zone.components_mut()
                .insert(pawn, stats.kinds.lumberjack.clone());
            zone.components_mut().insert(
                pawn,
                Idle {
                    seconds: SimUnit::ZERO,
                },
            );
        }

        // Three oaks + two pines scattered across the -X / -Y half of the
        // map. Two species so the per-kind asset binding (oak.glb vs
        // pine.glb, different bark textures, different chop times) is
        // visibly load-bearing rather than incidental. Half-tile-centred
        // like the pawns.
        let oaks: &[(i32, i32)] = &[(-6, -5), (-3, -6), (-7, -2)];
        let pines: &[(i32, i32)] = &[(-1, -4), (1, -6)];
        for &(x, y) in oaks {
            spawn_tree(zone, x, y, &stats.kinds.oak_tree);
        }
        for &(x, y) in pines {
            spawn_tree(zone, x, y, &stats.kinds.pine_tree);
        }

        let mut env = SimEnvironment::new();
        // One in-world day per real-time game length: the sun rises at start
        // (default `time_of_day = 0.25`), reaches noon halfway through, sets
        // as the timer expires. Short enough that the panel's time-of-day
        // slider has visible work to do even without dragging it.
        env.seconds_per_day = SimUnit::from_num(TIME_LIMIT_SECS);

        Self {
            zones,
            zone: zone_id,
            stockpile,
            state: GameState::Playing,
            elapsed: SimUnit::ZERO,
            stats,
            env,
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

fn spawn_tree(zone: &mut Zone, x: i32, y: i32, kind: &KindId) {
    let tree = zone.insert(WorldTransform {
        position: SimPos {
            x: SimUnit::from_num(x) + SimUnit::from_num(0.5),
            y: SimUnit::from_num(y) + SimUnit::from_num(0.5),
            z: SimUnit::ZERO,
        },
        facing: Facing::ZERO,
    });
    zone.components_mut().insert(tree, kind.clone());
}

/// External sim mutations the player can request. Today there's one —
/// toggling a chop designation on a tree by left-clicking it — but this is
/// the place new player verbs land (order rally, pause-individual-pawn,
/// cancel-order, …). Commands carry only the ids they need so they
/// serialise as plain data; resolution against current sim state happens
/// inside [`Game::apply_command`].
#[derive(Debug, Clone)]
pub enum Command {
    /// Toggle a chop designation on the targeted object. No-op if the
    /// target isn't a tree (the view filters before pushing, but
    /// `apply_command` re-validates so a recorded command stream replays
    /// safely against a possibly-altered sim).
    ToggleDesignation { target: WorldObjectRef },
    /// Set [`SimEnvironment::time_of_day`] to the given `[0, 1)` value.
    /// Pushed by the bottom-right debug panel's slider. Float at the
    /// command boundary (egui speaks `f32`); the sim converts to fixed-
    /// point on apply.
    SetTimeOfDay(f32),
    /// Set [`SimEnvironment::seconds_per_day`] — the real-time length of
    /// one in-world day. Same float-at-boundary shape as `SetTimeOfDay`.
    SetSecondsPerDay(f32),
}

impl Simulation for Game {
    type Command = Command;

    fn apply_command(&mut self, cmd: &Command) {
        match cmd {
            Command::ToggleDesignation { target } => {
                let WorldObjectRef { zone, id } = *target;
                // Snapshot the kind id from the immutable side before
                // taking a mutable borrow — same shape the original
                // toggle helper used.
                let is_tree = self
                    .zones
                    .get(zone)
                    .and_then(|z| z.components().get::<KindId>(id))
                    .map(|k| self.stats.tree_stats(k).is_some())
                    .unwrap_or(false);
                if !is_tree {
                    return;
                }
                let Some(zone) = self.zones.get_mut(zone) else {
                    return;
                };
                if zone.components().get::<Designated>(id).is_some() {
                    zone.components_mut().remove::<Designated>(id);
                } else {
                    zone.components_mut().insert(id, Designated);
                }
            }
            Command::SetTimeOfDay(t) => {
                self.env.time_of_day = SimUnit::from_num(t.rem_euclid(1.0));
            }
            Command::SetSecondsPerDay(s) => {
                // Clamp at the boundary — `SimEnvironment::advance` early-
                // returns on non-positive periods, so this just guards the
                // slider from making the panel display a paused-sun state
                // by accident.
                self.env.seconds_per_day = SimUnit::from_num(s.max(1.0));
            }
        }
    }

    fn tick(&mut self, dt: Duration) {
        if self.state != GameState::Playing {
            // Freeze gameplay on win/lose: no pawn motion, no chop ticks,
            // no timer advancement, no sun motion. The HUD keeps drawing
            // the banner. Time-of-day can still be dragged via Command.
            return;
        }
        self.env.advance(dt);
        // Convert wall-clock dt to Q16.16 sim-seconds via integer nanos
        // — bit-exact, no float involved.
        let nanos = dt.as_nanos();
        let secs_q16: u128 = (nanos << 16) / 1_000_000_000;
        let dt = SimUnit::from_bits(secs_q16.min(i32::MAX as u128) as i32);
        self.elapsed += dt;
        let stockpile = self.stockpile;
        let stats = &self.stats;
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
        chopping::on_arrival(zone, &arrived, stats);
        hauling::on_arrival(zone, &arrived);
        chopping::tick_progress(zone);
        hauling::dispatch_carrying(zone, stockpile, stats);
        chopping::dispatch_idle(zone, stats);
        idle::tick(zone, dt);

        // Win has priority: if a delivery during this same tick crossed
        // the goal, that beats the timer expiring on the same tick. The
        // state transition freezes the world on the next call via the
        // early-return at the top.
        if self.wood_count() >= WOOD_GOAL {
            self.state = GameState::Won;
        } else if self.elapsed >= SimUnit::from_num(TIME_LIMIT_SECS) {
            self.state = GameState::Lost;
        }
    }
}
