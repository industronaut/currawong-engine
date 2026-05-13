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
//! - [`environment`] — sim-side environment state (time of day) + the trivial
//!   sun-direction model.
//!
//! Submodules are private; their public types are re-exported here so callers
//! see a flat `sim::*` surface.

use std::time::Duration;

mod clock;
mod components;
mod environment;
mod grid;
mod slot_map;
mod terrain;
mod zone;

pub use clock::SimClock;
pub use components::Components;
pub use environment::{SimEnvironment, sun_direction_for};
pub use grid::{Grid, SquareGrid};
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
pub trait Simulation: 'static {
    /// Advance the simulation by `dt`.
    fn tick(&mut self, dt: Duration);
}

/// Trivial simulation; useful for view-only examples that have no world to
/// tick yet.
impl Simulation for () {
    fn tick(&mut self, _: Duration) {}
}
