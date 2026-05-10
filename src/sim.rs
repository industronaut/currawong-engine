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

/// An entity in the simulation.
///
/// Carries position and rotation only. Richer payloads (kind, behaviour,
/// attached data) are added by the caller in their own [`Simulation`] impl —
/// usually as parallel storage keyed by [`WorldObjectId`].
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

/// Stable handle to a [`WorldObject`] in a [`Zone`].
///
/// Generational: when an object is removed and its slot reused, the slot's
/// generation increments, which invalidates any old ID pointing at that
/// slot. Looking up a stale ID returns `None` rather than the new occupant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldObjectId {
    index: u32,
    generation: u32,
}

enum Slot {
    Occupied {
        generation: u32,
        object: WorldObject,
    },
    Vacant {
        generation: u32,
        next_free: Option<u32>,
    },
}

/// A region of the world.
///
/// Owns its [`WorldObject`]s in a generational slot-map. Insertion returns a
/// stable [`WorldObjectId`]; removal invalidates that ID. Iteration is O(n)
/// over live objects regardless of how many have been removed.
pub struct Zone {
    slots: Vec<Slot>,
    free_head: Option<u32>,
    len: usize,
}

impl Default for Zone {
    fn default() -> Self {
        Self::new()
    }
}

impl Zone {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert an object and return a handle to it.
    pub fn insert(&mut self, object: WorldObject) -> WorldObjectId {
        if let Some(idx) = self.free_head {
            let slot = &mut self.slots[idx as usize];
            let (generation, next_free) = match *slot {
                Slot::Vacant { generation, next_free } => (generation, next_free),
                Slot::Occupied { .. } => unreachable!("free_head pointed at occupied slot"),
            };
            self.free_head = next_free;
            *slot = Slot::Occupied { generation, object };
            self.len += 1;
            WorldObjectId { index: idx, generation }
        } else {
            let idx = u32::try_from(self.slots.len()).expect("zone capacity exceeded u32::MAX");
            self.slots.push(Slot::Occupied { generation: 0, object });
            self.len += 1;
            WorldObjectId { index: idx, generation: 0 }
        }
    }

    /// Remove the object referenced by `id`. Returns the removed object, or
    /// `None` if the id is stale or refers to an empty slot.
    pub fn remove(&mut self, id: WorldObjectId) -> Option<WorldObject> {
        let idx = id.index as usize;
        let slot = self.slots.get(idx)?;
        let new_gen = match slot {
            Slot::Occupied { generation, .. } if *generation == id.generation => {
                generation.wrapping_add(1)
            }
            _ => return None,
        };

        let next_free = self.free_head;
        let prev = std::mem::replace(
            &mut self.slots[idx],
            Slot::Vacant { generation: new_gen, next_free },
        );
        self.free_head = Some(id.index);
        self.len -= 1;

        match prev {
            Slot::Occupied { object, .. } => Some(object),
            Slot::Vacant { .. } => unreachable!("validated occupancy above"),
        }
    }

    pub fn get(&self, id: WorldObjectId) -> Option<&WorldObject> {
        match self.slots.get(id.index as usize)? {
            Slot::Occupied { generation, object } if *generation == id.generation => Some(object),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, id: WorldObjectId) -> Option<&mut WorldObject> {
        match self.slots.get_mut(id.index as usize)? {
            Slot::Occupied { generation, object } if *generation == id.generation => Some(object),
            _ => None,
        }
    }

    pub fn contains(&self, id: WorldObjectId) -> bool {
        self.get(id).is_some()
    }

    pub fn iter(&self) -> impl Iterator<Item = (WorldObjectId, &WorldObject)> + '_ {
        self.slots.iter().enumerate().filter_map(|(i, slot)| match slot {
            Slot::Occupied { generation, object } => Some((
                WorldObjectId { index: i as u32, generation: *generation },
                object,
            )),
            Slot::Vacant { .. } => None,
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (WorldObjectId, &mut WorldObject)> + '_ {
        self.slots.iter_mut().enumerate().filter_map(|(i, slot)| match slot {
            Slot::Occupied { generation, object } => Some((
                WorldObjectId { index: i as u32, generation: *generation },
                object,
            )),
            Slot::Vacant { .. } => None,
        })
    }
}

/// Maximum simulation ticks consumed per render frame. Caps speed-driven
/// catch-up to prevent spiral-of-death when the sim falls behind real time.
const MAX_TICKS_PER_FRAME: u32 = 16;

/// Drives a fixed-tick simulation from wall-clock time, with speed scaling.
///
/// The simulation always sees a constant [`tick_period`](Self::tick_period)
/// per call to [`Simulation::tick`]; varying `speed` only changes how many
/// ticks fire per wall-clock second. This keeps sim logic deterministic
/// regardless of playback speed.
///
/// Pause is `speed = 0.0`. Reverse playback (negative speed) is not supported.
pub struct SimClock {
    speed: f32,
    tick_period: Duration,
    accumulator: Duration,
    sim_time: Duration,
    total_ticks: u64,
}

impl SimClock {
    /// Default tick rate when none is specified.
    pub const DEFAULT_TICK_HZ: u32 = 60;

    /// Create a clock at 60 Hz, speed 1.0.
    pub fn new() -> Self {
        Self::with_tick_rate(Self::DEFAULT_TICK_HZ)
    }

    /// Create a clock at the given tick rate, speed 1.0.
    pub fn with_tick_rate(hz: u32) -> Self {
        assert!(hz > 0, "tick rate must be > 0");
        Self {
            speed: 1.0,
            tick_period: Duration::from_secs_f64(1.0 / hz as f64),
            accumulator: Duration::ZERO,
            sim_time: Duration::ZERO,
            total_ticks: 0,
        }
    }

    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Set the speed multiplier. `1.0` is real-time; `2.0` is 2x; `0.5` is
    /// half-speed; `0.0` is paused. Negative values are clamped to zero.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.max(0.0);
    }

    pub fn is_paused(&self) -> bool {
        self.speed == 0.0
    }

    pub fn tick_period(&self) -> Duration {
        self.tick_period
    }

    /// Number of simulation ticks elapsed since the clock was created.
    pub fn total_ticks(&self) -> u64 {
        self.total_ticks
    }

    /// Total simulated time elapsed (`total_ticks * tick_period`). Differs
    /// from wall time when speed is not 1.0.
    pub fn sim_time(&self) -> Duration {
        self.sim_time
    }

    /// Interpolation factor in `[0, 1]` between the most recent tick and the
    /// next pending tick. Pass to [`View::render`](crate::View::render) for
    /// smooth animation when tick rate is below refresh rate.
    pub fn alpha(&self) -> f32 {
        let acc = self.accumulator.as_secs_f64();
        let period = self.tick_period.as_secs_f64();
        (acc / period).clamp(0.0, 1.0) as f32
    }

    /// Advance the clock by `wall_dt` and return the number of sim ticks the
    /// caller should run. Caps at [`MAX_TICKS_PER_FRAME`] to prevent
    /// spiral-of-death; remaining accumulator is dropped at the cap.
    pub fn advance(&mut self, wall_dt: Duration) -> u32 {
        if self.speed <= 0.0 {
            return 0;
        }
        self.accumulator += wall_dt.mul_f32(self.speed);
        let mut ticks = 0;
        while self.accumulator >= self.tick_period && ticks < MAX_TICKS_PER_FRAME {
            self.accumulator -= self.tick_period;
            self.sim_time += self.tick_period;
            self.total_ticks += 1;
            ticks += 1;
        }
        if ticks == MAX_TICKS_PER_FRAME {
            self.accumulator = Duration::ZERO;
        }
        ticks
    }
}

impl Default for SimClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(x: f32) -> WorldObject {
        WorldObject {
            position: Vec3::new(x, 0.0, 0.0),
            rotation: Quat::IDENTITY,
        }
    }

    #[test]
    fn insert_then_get() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        assert_eq!(z.len(), 1);
        assert!(z.contains(a));
        assert_eq!(z.get(a).unwrap().position.x, 1.0);
    }

    #[test]
    fn remove_invalidates_id() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        let removed = z.remove(a).unwrap();
        assert_eq!(removed.position.x, 1.0);
        assert!(!z.contains(a));
        assert!(z.get(a).is_none());
        assert_eq!(z.len(), 0);
        // Removing a stale id is a no-op.
        assert!(z.remove(a).is_none());
    }

    #[test]
    fn slot_reuse_bumps_generation() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        z.remove(a);
        let b = z.insert(obj(2.0));
        assert_eq!(a.index, b.index, "slot index reused");
        assert_ne!(a.generation, b.generation, "generation bumped");
        assert!(!z.contains(a), "old id is stale");
        assert!(z.contains(b), "new id is valid");
        assert_eq!(z.get(b).unwrap().position.x, 2.0);
    }

    #[test]
    fn iter_visits_only_live_objects() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        let b = z.insert(obj(2.0));
        let c = z.insert(obj(3.0));
        z.remove(b);
        let ids: Vec<_> = z.iter().map(|(id, _)| id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&a));
        assert!(ids.contains(&c));
        assert!(!ids.contains(&b));
    }

    #[test]
    fn iter_mut_can_modify_in_place() {
        let mut z = Zone::new();
        let a = z.insert(obj(0.0));
        for (_, o) in z.iter_mut() {
            o.position.x += 5.0;
        }
        assert_eq!(z.get(a).unwrap().position.x, 5.0);
    }

    #[test]
    fn free_list_chains_multiple_removes() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        let b = z.insert(obj(2.0));
        let c = z.insert(obj(3.0));
        z.remove(a);
        z.remove(b);
        // Two reuses; both should land on the freed slots.
        let d = z.insert(obj(4.0));
        let e = z.insert(obj(5.0));
        // Third insert needs a fresh slot.
        let f = z.insert(obj(6.0));
        assert_eq!(z.len(), 4);
        for id in [c, d, e, f] {
            assert!(z.contains(id));
        }
        assert!(!z.contains(a));
        assert!(!z.contains(b));
    }
}
