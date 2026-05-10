//! Generic generational slot-map.
//!
//! A handle returned from [`SlotMap::insert`] is invalidated by removal:
//! after [`SlotMap::remove`], the slot's generation is bumped, so a stale key
//! returns `None` from [`SlotMap::get`] even after the slot is reused for a
//! different value.

use std::hash::Hash;
use std::marker::PhantomData;

/// A key into a [`SlotMap`]. Implementors are typed newtypes (e.g.
/// [`WorldObjectId`](crate::WorldObjectId), [`ZoneId`](crate::ZoneId)) so that
/// an id from one map can't be used to query a different one — the type
/// system rejects it.
pub trait SlotKey: Copy + Eq + Hash + std::fmt::Debug + 'static {
    fn from_raw(index: u32, generation: u32) -> Self;
    fn index(&self) -> u32;
    fn generation(&self) -> u32;
}

enum Slot<V> {
    Occupied {
        generation: u32,
        value: V,
    },
    Vacant {
        generation: u32,
        next_free: Option<u32>,
    },
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
                Slot::Vacant {
                    generation,
                    next_free,
                } => (generation, next_free),
                Slot::Occupied { .. } => unreachable!("free_head pointed at occupied slot"),
            };
            self.free_head = next_free;
            *slot = Slot::Occupied { generation, value };
            self.len += 1;
            K::from_raw(idx, generation)
        } else {
            let idx = u32::try_from(self.slots.len()).expect("slot map capacity exceeded u32::MAX");
            self.slots.push(Slot::Occupied {
                generation: 0,
                value,
            });
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
            Slot::Vacant {
                generation: new_gen,
                next_free,
            },
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
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied { generation, value } => {
                    Some((K::from_raw(i as u32, *generation), value))
                }
                Slot::Vacant { .. } => None,
            })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (K, &mut V)> + '_ {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied { generation, value } => {
                    Some((K::from_raw(i as u32, *generation), value))
                }
                Slot::Vacant { .. } => None,
            })
    }
}
