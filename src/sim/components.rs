//! Heterogeneous, type-erased component storage for sparse per-object state.

use std::any::{Any, TypeId};

use indexmap::IndexMap;

use super::zone::WorldObjectId;

/// Heterogeneous component storage keyed by [`WorldObjectId`].
///
/// One sparse map per registered component type, looked up by [`TypeId`].
/// Insertion is lazy — the first `insert::<T>` for a new type allocates the
/// inner map. Components are removed automatically when their owning object
/// is removed via [`Zone::remove`](crate::Zone::remove); that is the only
/// path that keeps component lifecycle in sync, so prefer it over reaching
/// into internals.
///
/// Inspired by the `ThingComp` pattern in RimWorld and creature-fact tables in
/// Dwarf Fortress: most sim-game state is sparse, optional, and per-entity, so
/// a hand-rolled per-type map fits better than an archetype ECS.
///
/// # Determinism
///
/// Both the inner per-type map and the outer type-id map are [`IndexMap`]s,
/// so [`Self::iter`] / [`Self::iter_mut`] visit objects in insertion order.
/// This is bit-stable across runs and across library versions (IndexMap has
/// no random seed). Removal uses `swap_remove` so it stays O(1); the
/// last-inserted entry takes the freed slot, which is still deterministic
/// for a given sequence of insertions + removals.
///
/// Outer-map iteration order over `TypeId`s is not part of the API; `TypeId`
/// values themselves can shift across recompilations, so cross-build
/// save-file portability is a separate concern (a stable component-id
/// registry — see #87's out-of-scope notes).
pub struct Components {
    maps: IndexMap<TypeId, Entry>,
}

/// One type-erased per-component map, plus the function pointer needed to
/// drop a `WorldObjectId` from it without knowing `T` (used by
/// [`Components::remove_all`]). The trait-object indirection that previously
/// hosted these has collapsed into the function pointer.
struct Entry {
    map: Box<dyn Any>,
    remove: fn(&mut dyn Any, WorldObjectId),
}

impl Entry {
    fn new<T: 'static>() -> Self {
        Self {
            map: Box::new(IndexMap::<WorldObjectId, T>::new()),
            remove: |map, id| {
                map.downcast_mut::<IndexMap<WorldObjectId, T>>()
                    .expect("TypeId keys its concrete map")
                    .swap_remove(&id);
            },
        }
    }
}

fn map_of<T: 'static>(maps: &IndexMap<TypeId, Entry>) -> Option<&IndexMap<WorldObjectId, T>> {
    maps.get(&TypeId::of::<T>())?.map.downcast_ref()
}

fn map_of_mut<T: 'static>(
    maps: &mut IndexMap<TypeId, Entry>,
) -> Option<&mut IndexMap<WorldObjectId, T>> {
    maps.get_mut(&TypeId::of::<T>())?.map.downcast_mut()
}

impl Components {
    pub fn new() -> Self {
        Self {
            maps: IndexMap::new(),
        }
    }

    /// Attach a `T` to `id`, returning the previous value if one was already set.
    pub fn insert<T: 'static>(&mut self, id: WorldObjectId, value: T) -> Option<T> {
        self.maps
            .entry(TypeId::of::<T>())
            .or_insert_with(Entry::new::<T>)
            .map
            .downcast_mut::<IndexMap<WorldObjectId, T>>()
            .expect("TypeId keys its concrete map")
            .insert(id, value)
    }

    pub fn get<T: 'static>(&self, id: WorldObjectId) -> Option<&T> {
        map_of::<T>(&self.maps)?.get(&id)
    }

    pub fn get_mut<T: 'static>(&mut self, id: WorldObjectId) -> Option<&mut T> {
        map_of_mut::<T>(&mut self.maps)?.get_mut(&id)
    }

    /// Remove the `T` attached to `id`, returning it if present. O(1) via
    /// `swap_remove`; the entry that previously held the last slot moves
    /// into the freed position.
    pub fn remove<T: 'static>(&mut self, id: WorldObjectId) -> Option<T> {
        map_of_mut::<T>(&mut self.maps)?.swap_remove(&id)
    }

    /// Drop every component attached to `id`. Called by
    /// [`Zone::remove`](crate::Zone::remove) so component lifecycle tracks
    /// object lifecycle.
    pub fn remove_all(&mut self, id: WorldObjectId) {
        for entry in self.maps.values_mut() {
            (entry.remove)(&mut *entry.map, id);
        }
    }

    pub fn iter<T: 'static>(&self) -> impl Iterator<Item = (WorldObjectId, &T)> + '_ {
        map_of::<T>(&self.maps)
            .into_iter()
            .flat_map(|m| m.iter().map(|(id, v)| (*id, v)))
    }

    pub fn iter_mut<T: 'static>(&mut self) -> impl Iterator<Item = (WorldObjectId, &mut T)> + '_ {
        map_of_mut::<T>(&mut self.maps)
            .into_iter()
            .flat_map(|m| m.iter_mut().map(|(id, v)| (*id, v)))
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
    use crate::sim::{Facing, SimPos, WorldTransform, Zone};

    fn obj(x: f32) -> WorldTransform {
        // Tests don't care about exact position; tile-flavoured + facing
        // zero is enough. Cast f32 → integer tile via truncation.
        WorldTransform {
            position: SimPos::tile(x as i32, 0, 0),
            facing: Facing::ZERO,
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
    fn components_iter_visits_in_insertion_order() {
        // Determinism guarantee post-#87: IndexMap iterates in insertion
        // order, no sort required.
        let mut z = Zone::new();
        let a = z.insert(obj(1.0));
        let b = z.insert(obj(2.0));
        z.components_mut().insert(a, Health(10));
        z.components_mut().insert(b, Health(20));
        let got: Vec<(WorldObjectId, u32)> = z
            .components()
            .iter::<Health>()
            .map(|(id, h)| (id, h.0))
            .collect();
        assert_eq!(got, vec![(a, 10), (b, 20)]);
    }

    #[test]
    fn components_iter_order_is_stable_across_runs() {
        // The determinism invariant. Build the same component soup twice
        // and assert iteration order matches exactly. A `HashMap`-backed
        // version of this test would have ~50% odds of failing on any
        // given run.
        fn build() -> Vec<u32> {
            let mut z = Zone::new();
            let ids: Vec<_> = (0..32).map(|_| z.insert(obj(0.0))).collect();
            for (i, &id) in ids.iter().enumerate() {
                z.components_mut().insert(id, Health(i as u32));
            }
            z.components().iter::<Health>().map(|(_, h)| h.0).collect()
        }
        let a = build();
        let b = build();
        assert_eq!(a, b);
        assert_eq!(a, (0..32).collect::<Vec<_>>());
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
