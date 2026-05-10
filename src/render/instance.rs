//! Per-key instance bucketing for batched instanced rendering.
//!
//! Subsumes the clear-and-rebuild loop most views need to expose their visible
//! parts as instance buffers: register one bucket per key (typically a mesh
//! handle) at init, then each frame `begin_frame` → `push` for each visible
//! part → `upload` → drive draws from `iter_filled`.
//!
//! ```text
//! init:    for k in keys: instances.register(device, k);
//! frame:   instances.begin_frame();
//!          extract_visuals(...) { instances.push(k, instance) }
//!          instances.upload(queue);
//!          for (k, buffer, count) in instances.iter_filled() { draw }
//! ```
//!
//! The user retains ownership of mesh resources and the pipeline; the helper
//! only handles per-key CPU scratch buffers and their GPU instance buffers.

use std::collections::HashMap;
use std::hash::Hash;

use bytemuck::Pod;

/// Manages a CPU scratch `Vec` and a GPU instance buffer per key. Generic
/// over the bucket key `K` (typically a mesh handle) and the per-instance
/// data `I` (typically a `Mat4` model matrix or a struct of attributes).
///
/// Iteration order over filled buckets is unspecified — fine for opaque
/// depth-tested rendering. Transparent passes that need back-to-front order
/// must sort the iter_filled result themselves.
pub struct InstanceBuckets<K, I> {
    buckets: HashMap<K, Bucket<I>>,
    label: String,
    capacity: u32,
}

struct Bucket<I> {
    scratch: Vec<I>,
    buffer: wgpu::Buffer,
}

impl<K, I> InstanceBuckets<K, I>
where
    K: Copy + Eq + Hash,
    I: Pod,
{
    /// Create an empty bucket group. Buckets are allocated lazily by
    /// [`register`](Self::register). `capacity` is the per-bucket instance
    /// limit — pushes past it are silently dropped.
    pub fn new(label: impl Into<String>, capacity: u32) -> Self {
        assert!(capacity > 0, "InstanceBuckets capacity must be > 0");
        Self {
            buckets: HashMap::new(),
            label: label.into(),
            capacity,
        }
    }

    /// Per-bucket capacity in instances.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Allocate a bucket for `key`. The GPU buffer is sized for `capacity`
    /// instances. Re-registering an already-known key replaces the bucket
    /// and releases the prior buffer.
    pub fn register(&mut self, device: &wgpu::Device, key: K) {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&self.label),
            size: (self.capacity as u64) * std::mem::size_of::<I>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.buckets.insert(
            key,
            Bucket {
                scratch: Vec::with_capacity(self.capacity as usize),
                buffer,
            },
        );
    }

    /// Clear all bucket scratch. Call at the start of each frame before
    /// [`push`](Self::push)ing.
    pub fn begin_frame(&mut self) {
        for b in self.buckets.values_mut() {
            b.scratch.clear();
        }
    }

    /// Append `instance` to `key`'s bucket.
    ///
    /// Panics if `key` was not registered — that's a programming error,
    /// not a runtime condition. Pushes past the configured capacity are
    /// silently dropped.
    pub fn push(&mut self, key: K, instance: I) {
        let bucket = self
            .buckets
            .get_mut(&key)
            .expect("InstanceBuckets::push called with unregistered key — register at init");
        if (bucket.scratch.len() as u32) < self.capacity {
            bucket.scratch.push(instance);
        }
    }

    /// Upload all non-empty buckets to their GPU buffers. Call once per
    /// frame after all pushes and before drawing.
    pub fn upload(&self, queue: &wgpu::Queue) {
        for b in self.buckets.values() {
            if !b.scratch.is_empty() {
                queue.write_buffer(&b.buffer, 0, bytemuck::cast_slice(&b.scratch));
            }
        }
    }

    /// Iterate non-empty buckets as `(key, instance buffer, instance count)`.
    /// Bind the buffer as a vertex slot and issue an instanced draw against
    /// the mesh associated with `key`.
    pub fn iter_filled(&self) -> impl Iterator<Item = (K, &wgpu::Buffer, u32)> + '_ {
        self.buckets.iter().filter_map(|(k, b)| {
            let count = b.scratch.len() as u32;
            (count > 0).then_some((*k, &b.buffer, count))
        })
    }
}
