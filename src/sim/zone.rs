//! World objects, zones, and cross-zone references.
//!
//! Zones are coordinate-isolated: each owns a local frame and a [`SlotMap`]
//! of [`WorldObject`]s plus a [`Components`] registry for sparse per-object
//! data. Movement between zones is a storage operation — the engine provides
//! no cross-zone positional math.

use glam::{Quat, Vec3};

use super::components::Components;
use super::slot_map::{SlotKey, SlotMap};

/// An entity in the simulation.
///
/// Carries position and rotation only. Richer payloads (kind, behaviour,
/// attached data) are added by the caller in their own
/// [`Simulation`](crate::Simulation) impl — usually as parallel storage keyed
/// by [`WorldObjectId`], or as [`Components`] entries.
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

/// A region of the world. Owns the [`WorldObject`]s within it and a
/// [`Components`] registry for sparse, optional per-object data (health,
/// faction, AI state, etc.).
///
/// Use [`Zone::remove`] rather than reaching for the inner storage: it's the
/// only path that keeps the [`Components`] registry in sync.
pub struct Zone {
    objects: SlotMap<WorldObjectId, WorldObject>,
    components: Components,
}

impl Zone {
    pub fn new() -> Self {
        Self {
            objects: SlotMap::new(),
            components: Components::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn insert(&mut self, obj: WorldObject) -> WorldObjectId {
        self.objects.insert(obj)
    }

    /// Remove `id` and every component attached to it. Returns the removed
    /// [`WorldObject`], or `None` if the id is stale.
    pub fn remove(&mut self, id: WorldObjectId) -> Option<WorldObject> {
        let obj = self.objects.remove(id)?;
        self.components.remove_all(id);
        Some(obj)
    }

    pub fn get(&self, id: WorldObjectId) -> Option<&WorldObject> {
        self.objects.get(id)
    }

    pub fn get_mut(&mut self, id: WorldObjectId) -> Option<&mut WorldObject> {
        self.objects.get_mut(id)
    }

    pub fn contains(&self, id: WorldObjectId) -> bool {
        self.objects.contains(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (WorldObjectId, &WorldObject)> + '_ {
        self.objects.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (WorldObjectId, &mut WorldObject)> + '_ {
        self.objects.iter_mut()
    }

    pub fn components(&self) -> &Components {
        &self.components
    }

    pub fn components_mut(&mut self) -> &mut Components {
        &mut self.components
    }

    /// Borrow the object slot-map and the component registry independently —
    /// useful when iterating components and mutating objects (or vice versa)
    /// in the same pass, where `&mut self` would borrow the whole zone.
    ///
    /// Removing through the returned [`SlotMap`] bypasses
    /// [`Components::remove_all`] and will leak components; use [`Zone::remove`]
    /// for normal removal.
    pub fn split_mut(
        &mut self,
    ) -> (&mut SlotMap<WorldObjectId, WorldObject>, &mut Components) {
        (&mut self.objects, &mut self.components)
    }
}

impl Default for Zone {
    fn default() -> Self {
        Self::new()
    }
}

/// The simulation's collection of [`Zone`]s. Owned by the user's
/// [`Simulation`](crate::Simulation) impl.
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

    #[test]
    fn zone_split_mut_allows_concurrent_iter_and_object_mutation() {
        #[derive(Debug, PartialEq)]
        struct Health(u32);

        let mut z = Zone::new();
        let a = z.insert(obj(0.0));
        z.components_mut().insert(a, Health(7));

        let (objects, components) = z.split_mut();
        for (id, h) in components.iter::<Health>() {
            if let Some(o) = objects.get_mut(id) {
                o.position.x = h.0 as f32;
            }
        }
        assert_eq!(z.get(a).unwrap().position.x, 7.0);
    }
}
