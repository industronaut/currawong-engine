//! Simulation side of the engine.
//!
//! Types in this module do not depend on `wgpu` or `winit`. The simulation
//! ticks regardless of whether anything is rendering, supports headless
//! execution, and is the authoritative state for the world.
//!
//! ## Hierarchy
//!
//! [`Simulation`] → [`Zones`] → [`Zone`] → [`WorldObject`].
//!
//! Zones are coordinate-isolated: each zone has its own local frame and the
//! engine does not provide cross-zone positional math. Movement between zones
//! is a storage operation (remove from one zone, insert into another) — not a
//! position update.

use std::hash::Hash;
use std::marker::PhantomData;
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

// --- Generic generational slot-map ---------------------------------------

/// A key into a [`SlotMap`]. Implementors are typed newtypes (e.g.
/// [`WorldObjectId`], [`ZoneId`]) so that an id from one map can't be used
/// to query a different one — the type system rejects it.
pub trait SlotKey: Copy + Eq + Hash + std::fmt::Debug + 'static {
    fn from_raw(index: u32, generation: u32) -> Self;
    fn index(&self) -> u32;
    fn generation(&self) -> u32;
}

enum Slot<V> {
    Occupied { generation: u32, value: V },
    Vacant { generation: u32, next_free: Option<u32> },
}

/// A generational slot-map.
///
/// Insertion returns a stable [`SlotKey`]; removal invalidates that key by
/// bumping the slot's generation, so a stale key returns `None` from `get`
/// even after the slot is reused for a different value. Iteration is O(n)
/// over live entries regardless of how many have been removed.
pub struct SlotMap<K: SlotKey, V> {
    slots: Vec<Slot<V>>,
    free_head: Option<u32>,
    len: usize,
    _phantom: PhantomData<fn() -> K>,
}

impl<K: SlotKey, V> Default for SlotMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: SlotKey, V> SlotMap<K, V> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            len: 0,
            _phantom: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert a value and return a handle to it.
    pub fn insert(&mut self, value: V) -> K {
        if let Some(idx) = self.free_head {
            let slot = &mut self.slots[idx as usize];
            let (generation, next_free) = match *slot {
                Slot::Vacant { generation, next_free } => (generation, next_free),
                Slot::Occupied { .. } => unreachable!("free_head pointed at occupied slot"),
            };
            self.free_head = next_free;
            *slot = Slot::Occupied { generation, value };
            self.len += 1;
            K::from_raw(idx, generation)
        } else {
            let idx =
                u32::try_from(self.slots.len()).expect("slot map capacity exceeded u32::MAX");
            self.slots.push(Slot::Occupied { generation: 0, value });
            self.len += 1;
            K::from_raw(idx, 0)
        }
    }

    /// Remove the value referenced by `key`. Returns the removed value, or
    /// `None` if the key is stale or out of range.
    pub fn remove(&mut self, key: K) -> Option<V> {
        let idx = key.index() as usize;
        let slot = self.slots.get(idx)?;
        let new_gen = match slot {
            Slot::Occupied { generation, .. } if *generation == key.generation() => {
                generation.wrapping_add(1)
            }
            _ => return None,
        };

        let next_free = self.free_head;
        let prev = std::mem::replace(
            &mut self.slots[idx],
            Slot::Vacant { generation: new_gen, next_free },
        );
        self.free_head = Some(key.index());
        self.len -= 1;

        match prev {
            Slot::Occupied { value, .. } => Some(value),
            Slot::Vacant { .. } => unreachable!("validated occupancy above"),
        }
    }

    pub fn get(&self, key: K) -> Option<&V> {
        match self.slots.get(key.index() as usize)? {
            Slot::Occupied { generation, value } if *generation == key.generation() => Some(value),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        match self.slots.get_mut(key.index() as usize)? {
            Slot::Occupied { generation, value } if *generation == key.generation() => Some(value),
            _ => None,
        }
    }

    pub fn contains(&self, key: K) -> bool {
        self.get(key).is_some()
    }

    pub fn iter(&self) -> impl Iterator<Item = (K, &V)> + '_ {
        self.slots.iter().enumerate().filter_map(|(i, slot)| match slot {
            Slot::Occupied { generation, value } => {
                Some((K::from_raw(i as u32, *generation), value))
            }
            Slot::Vacant { .. } => None,
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (K, &mut V)> + '_ {
        self.slots.iter_mut().enumerate().filter_map(|(i, slot)| match slot {
            Slot::Occupied { generation, value } => {
                Some((K::from_raw(i as u32, *generation), value))
            }
            Slot::Vacant { .. } => None,
        })
    }
}

// --- Concrete keys + aliases --------------------------------------------

/// Stable handle to a [`WorldObject`] within its owning [`Zone`].
///
/// Generational, so a key whose slot has been reused returns `None` from
/// [`Zone::get`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldObjectId {
    index: u32,
    generation: u32,
}

impl SlotKey for WorldObjectId {
    fn from_raw(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }
    fn index(&self) -> u32 {
        self.index
    }
    fn generation(&self) -> u32 {
        self.generation
    }
}

/// Stable handle to a [`Zone`] within a [`Zones`] collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZoneId {
    index: u32,
    generation: u32,
}

impl SlotKey for ZoneId {
    fn from_raw(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }
    fn index(&self) -> u32 {
        self.index
    }
    fn generation(&self) -> u32 {
        self.generation
    }
}

/// A region of the world. Owns a generational slot-map of [`WorldObject`]s.
pub type Zone = SlotMap<WorldObjectId, WorldObject>;

/// The simulation's collection of [`Zone`]s. Owned by the user's
/// [`Simulation`] impl.
pub type Zones = SlotMap<ZoneId, Zone>;

/// Fully-qualified handle to a [`WorldObject`] across zones. Use this for
/// references that outlive a single zone's scope — camera targets, AI
/// memory, save-game pointers, network messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldObjectRef {
    pub zone: ZoneId,
    pub id: WorldObjectId,
}

impl WorldObjectRef {
    /// Look up the object this ref points at. Returns `None` if either the
    /// zone or the object has been removed since the ref was created.
    pub fn resolve<'a>(&self, zones: &'a Zones) -> Option<&'a WorldObject> {
        zones.get(self.zone)?.get(self.id)
    }

    pub fn resolve_mut<'a>(&self, zones: &'a mut Zones) -> Option<&'a mut WorldObject> {
        zones.get_mut(self.zone)?.get_mut(self.id)
    }
}

// --- Simulation clock ---------------------------------------------------

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

    // --- Zone (SlotMap<WorldObjectId, WorldObject>) ----------------------

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
        assert!(z.remove(a).is_none());
    }

    #[test]
    fn slot_reuse_bumps_generation() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        z.remove(a);
        let b = z.insert(obj(2.0));
        assert_eq!(a.index(), b.index(), "slot index reused");
        assert_ne!(a.generation(), b.generation(), "generation bumped");
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
        let d = z.insert(obj(4.0));
        let e = z.insert(obj(5.0));
        let f = z.insert(obj(6.0));
        assert_eq!(z.len(), 4);
        for id in [c, d, e, f] {
            assert!(z.contains(id));
        }
        assert!(!z.contains(a));
        assert!(!z.contains(b));
    }

    // --- Zones (SlotMap<ZoneId, Zone>) + WorldObjectRef ------------------

    #[test]
    fn nested_slot_maps_compose() {
        let mut zones = Zones::new();
        let zone_id = zones.insert(Zone::new());
        let obj_id = zones.get_mut(zone_id).unwrap().insert(obj(7.0));
        assert_eq!(zones.get(zone_id).unwrap().get(obj_id).unwrap().position.x, 7.0);
    }

    #[test]
    fn world_object_ref_resolves() {
        let mut zones = Zones::new();
        let zone = zones.insert(Zone::new());
        let id = zones.get_mut(zone).unwrap().insert(obj(3.5));
        let r = WorldObjectRef { zone, id };
        assert_eq!(r.resolve(&zones).unwrap().position.x, 3.5);
    }

    #[test]
    fn world_object_ref_resolve_mut() {
        let mut zones = Zones::new();
        let zone = zones.insert(Zone::new());
        let id = zones.get_mut(zone).unwrap().insert(obj(0.0));
        let r = WorldObjectRef { zone, id };
        r.resolve_mut(&mut zones).unwrap().position.x = 9.0;
        assert_eq!(r.resolve(&zones).unwrap().position.x, 9.0);
    }

    #[test]
    fn ref_with_stale_zone_returns_none() {
        let mut zones = Zones::new();
        let zone = zones.insert(Zone::new());
        let id = zones.get_mut(zone).unwrap().insert(obj(1.0));
        let r = WorldObjectRef { zone, id };
        zones.remove(zone);
        assert!(r.resolve(&zones).is_none());
    }

    #[test]
    fn ref_with_stale_object_id_returns_none() {
        let mut zones = Zones::new();
        let zone = zones.insert(Zone::new());
        let id = zones.get_mut(zone).unwrap().insert(obj(1.0));
        let r = WorldObjectRef { zone, id };
        zones.get_mut(zone).unwrap().remove(id);
        assert!(r.resolve(&zones).is_none());
    }
}
