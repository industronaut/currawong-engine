//! Simulation side of the engine.
//!
//! Types in this module do not depend on `wgpu` or `winit`. The simulation
//! ticks regardless of whether anything is rendering, supports headless
//! execution, and is the authoritative state for the world.

use std::time::Duration;

use glam::{Quat, Vec3};

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

/// A region of the world. Owns a collection of [`WorldObject`]s.
#[derive(Default)]
pub struct Zone {
    pub objects: Vec<WorldObject>,
}

/// An entity in the simulation.
///
/// Carries position and rotation only. Richer payloads (kind, behaviour,
/// attached data) are added by the caller in their own [`Simulation`] impl —
/// usually as parallel storage indexed by `WorldObject` position in the
/// owning [`Zone`], or as data carried in a wrapping container.
#[derive(Clone, Copy)]
pub struct WorldObject {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Default for WorldObject {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}
