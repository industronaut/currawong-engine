//! Simulation side of the engine.
//!
//! Types in this module do not depend on `wgpu` or `winit`. The simulation
//! ticks regardless of whether anything is rendering, supports headless
//! execution, and is the authoritative state for the world.
//!
//! ## Hierarchy
//!
//! [`Simulation`] → [`Zones`] → [`Zone`] → [`WorldTransform`].
//!
//! Zones are coordinate-isolated: each zone has its own local frame and the
//! engine does not provide cross-zone positional math. Movement between zones
//! is a storage operation (remove from one zone, insert into another) — not a
//! position update.
//!
//! ## Module layout
//!
//! - [`slot_map`] — generic generational slot-map (the storage primitive).
//! - [`grid`] — tile-grid topology trait + [`SquareGrid`] / (future) `HexGrid`.
//! - [`zone`] — [`WorldTransform`], [`Zone`], [`Zones`], cross-zone refs.
//! - [`components`] — sparse, type-erased per-object data.
//! - [`terrain`] — tile-grid terrain with optional liquids per tile.
//! - [`clock`] — fixed-tick driver with speed scaling.
//! - [`command`] — [`CommandQueue`]: the one seam through which external
//!   mutations reach the simulation (input handlers, UI, scripts).
//! - [`environment`] — sim-side environment state (time of day) + the trivial
//!   sun-direction model.
//!
//! Submodules are private; their public types are re-exported here so callers
//! see a flat `sim::*` surface.

use std::time::Duration;

mod clock;
mod command;
mod components;
mod environment;
mod grid;
mod slot_map;
mod terrain;
mod zone;

pub use clock::SimClock;
pub use command::CommandQueue;
pub use components::Components;
pub use environment::{SimEnvironment, sun_direction_for};
pub use grid::{Grid, HexGrid, SquareGrid};
pub use slot_map::{SlotKey, SlotMap};
pub use terrain::{CHUNK_SIZE, Chunk, ChunkCoord, Liquid, LiquidId, Terrain, Tile, TileCoord};
pub use zone::{
    WorldObjectId, WorldObjectRef, WorldObjectsMut, WorldTransform, Zone, ZoneId, Zones,
};

/// A simulation of the world.
///
/// Implement this on your game's state to define per-tick logic. The same
/// `Simulation` is the authoritative state whether the world is being viewed
/// in a window, ticked headless on a server, or replayed from a recording.
///
/// External mutations (player input, UI widgets, scripted tests) flow through
/// [`Command`](Self::Command) values; views push them into a
/// [`CommandQueue`] and the engine drains the queue at tick boundaries,
/// calling [`apply_command`](Self::apply_command) once per command before
/// [`tick`](Self::tick). No code outside the engine ever holds `&mut Self`.
pub trait Simulation: 'static {
    /// Closed enumerable set of external mutations. The user's enum
    /// (`PlayerMove`, `ToggleDesignation`, `SetSpeed`, …) — primitive
    /// serialisable data only, no closures, no borrows, no `dyn Trait`. If
    /// a variant needs to reach a sim object it carries the [`WorldObjectId`]
    /// (or other id) and [`apply_command`](Self::apply_command) resolves it
    /// against current sim state.
    ///
    /// Simulations with no external mutations (view-only demos, headless
    /// fixtures) can default this to `()`.
    type Command;

    /// Advance the simulation by `dt`.
    fn tick(&mut self, dt: Duration);

    /// Apply a single drained [`Command`](Self::Command). The single
    /// mutation seam. Called by the engine — once per command, in insertion
    /// order — at the tick boundary the command was scheduled for.
    fn apply_command(&mut self, cmd: &Self::Command) {
        let _ = cmd;
    }
}

/// Trivial simulation; useful for view-only examples that have no world to
/// tick yet.
impl Simulation for () {
    type Command = ();
    fn tick(&mut self, _: Duration) {}
}
