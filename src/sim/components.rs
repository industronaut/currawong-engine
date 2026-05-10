//! Heterogeneous, type-erased component storage for sparse per-object state.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use super::zone::WorldObjectId;

/// Heterogeneous component storage keyed by [`WorldObjectId`].
///
/// One sparse map per registered component type, looked up by [`TypeId`].
/// Insertion is lazy — the first `insert::<T>` for a new type allocates the
/// inner map. Components are removed automatically when their owning
/// [`WorldObject`](crate::WorldObject) is removed via
/// [`Zone::remove`](crate::Zone::remove); that is the only path that keeps
/// component lifecycle in sync, so prefer it over reaching into internals.
///
/// Inspired by the `ThingComp` pattern in RimWorld and creature-fact tables in
/// Dwarf Fortress: most sim-game state is sparse, optional, and per-entity, so
/// a hand-rolled per-type map fits better than an archetype ECS.
///
/// # Determinism
///
/// `HashMap` iteration order is randomly seeded — component iteration is
/// **not** deterministic across runs. This is fine for prototyping but will
/// break sim replay / lockstep networking once those exist. Swap the inner
/// `HashMap<WorldObjectId, T>` for a sparse-set (deterministic by insertion
/// order) or a fixed-seed hasher when that becomes a constraint.
pub struct Components {
    maps: HashMap<TypeId, Box<dyn ComponentStorage>>,
}

trait ComponentStorage: Any {
    fn remove(&mut self, id: WorldObjectId);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

struct TypedStore<T> {
    map: HashMap<WorldObjectId, T>,
}

impl<T: 'static> ComponentStorage for TypedStore<T> {
    fn remove(&mut self, id: WorldObjectId) {
        self.map.remove(&id);
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Components {
    pub fn new() -> Self {
        Self { maps: HashMap::new() }
    }

    /// Attach a `T` to `id`, returning the previous value if one was already set.
    pub fn insert<T: 'static>(&mut self, id: WorldObjectId, value: T) -> Option<T> {
        let store = self
            .maps
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(TypedStore::<T> { map: HashMap::new() }));
        store
            .as_any_mut()
            .downcast_mut::<TypedStore<T>>()
            .expect("TypeId keys its concrete TypedStore")
            .map
            .insert(id, value)
    }

    pub fn get<T: 'static>(&self, id: WorldObjectId) -> Option<&T> {
        let store = self.maps.get(&TypeId::of::<T>())?;
        store
            .as_any()
            .downcast_ref::<TypedStore<T>>()?
            .map
            .get(&id)
    }

    pub fn get_mut<T: 'static>(&mut self, id: WorldObjectId) -> Option<&mut T> {
        let store = self.maps.get_mut(&TypeId::of::<T>())?;
        store
            .as_any_mut()
            .downcast_mut::<TypedStore<T>>()?
            .map
            .get_mut(&id)
    }

    /// Remove the `T` attached to `id`, returning it if present.
    pub fn remove<T: 'static>(&mut self, id: WorldObjectId) -> Option<T> {
        let store = self.maps.get_mut(&TypeId::of::<T>())?;
        store
            .as_any_mut()
            .downcast_mut::<TypedStore<T>>()?
            .map
            .remove(&id)
    }

    /// Drop every component attached to `id`. Called by
    /// [`Zone::remove`](crate::Zone::remove) so component lifecycle tracks
    /// object lifecycle.
    pub fn remove_all(&mut self, id: WorldObjectId) {
        for store in self.maps.values_mut() {
            store.remove(id);
        }
    }

    pub fn iter<T: 'static>(&self) -> impl Iterator<Item = (WorldObjectId, &T)> + '_ {
        self.maps
            .get(&TypeId::of::<T>())
            .and_then(|store| store.as_any().downcast_ref::<TypedStore<T>>())
            .into_iter()
            .flat_map(|store| store.map.iter().map(|(id, v)| (*id, v)))
    }

    pub fn iter_mut<T: 'static>(
        &mut self,
    ) -> impl Iterator<Item = (WorldObjectId, &mut T)> + '_ {
        self.maps
            .get_mut(&TypeId::of::<T>())
            .and_then(|store| store.as_any_mut().downcast_mut::<TypedStore<T>>())
            .into_iter()
            .flat_map(|store| store.map.iter_mut().map(|(id, v)| (*id, v)))
    }
}

impl Default for Components {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{WorldObject, Zone};
    use glam::{Quat, Vec3};

    fn obj(x: f32) -> WorldObject {
        WorldObject {
            position: Vec3::new(x, 0.0, 0.0),
            rotation: Quat::IDENTITY,
        }
    }

    #[derive(Debug, PartialEq)]
    struct Health(u32);

    #[derive(Debug, PartialEq)]
    struct Faction(&'static str);

    #[test]
    fn components_insert_and_get() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        z.components_mut().insert(a, Health(100));
        assert_eq!(z.components().get::<Health>(a), Some(&Health(100)));
    }

    #[test]
    fn components_get_returns_none_when_absent() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        assert_eq!(z.components().get::<Health>(a), None);
    }

    #[test]
    fn components_insert_replaces_returning_old_value() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        assert_eq!(z.components_mut().insert(a, Health(100)), None);
        assert_eq!(z.components_mut().insert(a, Health(50)), Some(Health(100)));
        assert_eq!(z.components().get::<Health>(a), Some(&Health(50)));
    }

    #[test]
    fn components_get_mut_allows_modification() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        z.components_mut().insert(a, Health(100));
        if let Some(h) = z.components_mut().get_mut::<Health>(a) {
            h.0 -= 25;
        }
        assert_eq!(z.components().get::<Health>(a), Some(&Health(75)));
    }

    #[test]
    fn components_remove_returns_value() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        z.components_mut().insert(a, Health(42));
        assert_eq!(z.components_mut().remove::<Health>(a), Some(Health(42)));
        assert_eq!(z.components().get::<Health>(a), None);
        assert_eq!(z.components_mut().remove::<Health>(a), None);
    }

    #[test]
    fn components_iter_visits_all_entries() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        let b = z.insert(obj(2.0));
        z.components_mut().insert(a, Health(10));
        z.components_mut().insert(b, Health(20));
        // HashMap iteration order is non-deterministic — sort before comparing.
        let mut got: Vec<(WorldObjectId, u32)> =
            z.components().iter::<Health>().map(|(id, h)| (id, h.0)).collect();
        got.sort_by_key(|(_, h)| *h);
        assert_eq!(got, vec![(a, 10), (b, 20)]);
    }

    #[test]
    fn components_iter_mut_modifies_in_place() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        let b = z.insert(obj(2.0));
        z.components_mut().insert(a, Health(10));
        z.components_mut().insert(b, Health(20));
        for (_, h) in z.components_mut().iter_mut::<Health>() {
            h.0 += 5;
        }
        assert_eq!(z.components().get::<Health>(a), Some(&Health(15)));
        assert_eq!(z.components().get::<Health>(b), Some(&Health(25)));
    }

    #[test]
    fn components_different_types_coexist() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        z.components_mut().insert(a, Health(100));
        z.components_mut().insert(a, Faction("blue"));
        assert_eq!(z.components().get::<Health>(a), Some(&Health(100)));
        assert_eq!(z.components().get::<Faction>(a), Some(&Faction("blue")));
    }

    #[test]
    fn components_iter_for_unregistered_type_is_empty() {
        let z = Zone::new();
        assert_eq!(z.components().iter::<Health>().count(), 0);
    }

    #[test]
    fn zone_remove_cascades_to_components() {
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        let b = z.insert(obj(2.0));
        z.components_mut().insert(a, Health(100));
        z.components_mut().insert(a, Faction("blue"));
        z.components_mut().insert(b, Health(50));

        z.remove(a);

        // a's components are gone across all types.
        assert_eq!(z.components().get::<Health>(a), None);
        assert_eq!(z.components().get::<Faction>(a), None);
        // b's components are untouched.
        assert_eq!(z.components().get::<Health>(b), Some(&Health(50)));
    }
}
