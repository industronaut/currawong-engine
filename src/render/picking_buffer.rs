//! GPU hit-ID buffer plumbing (PR 2 of #56).
//!
//! Three pieces live here:
//!
//! - [`HitTarget`] — what a hit ID resolves to. PR 2 only emits
//!   [`HitTarget::TerrainCell`]; mesh objects join in PR 3.
//! - [`FrameIdTable`] — the per-frame indirection table. Things that draw
//!   into the ID attachment ask it for an ID and the table remembers what
//!   that ID stood for. Lives inside the engine's [`SceneResources`] behind
//!   a `Mutex` so the engine can build it up during `View::render` (which
//!   only sees `&Renderer`) and consult it after readback completes.
//! - [`IdReadback`] — the ring of staging buffers. Each frame, if the
//!   cursor is in bounds, one slot copies a 1×1 region of the ID texture
//!   under the cursor into a `MAP_READ`-able buffer; a `map_async` callback
//!   resolves the result through the table snapshot that frame committed,
//!   and stores it in [`IdReadback::latest_target`] for picker consumers.
//!
//! Readback has 1–3 frames of latency (depends on adapter pipelining) —
//! short enough that hover feels real-time over coarse targets like terrain
//! cells. The [`TerrainPicker`](super::TerrainPicker) keeps its old ray-vs-
//! plane result around as the zero-latency fallback that wallpapers over
//! the first frame and any frame where every slot is in flight.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use glam::IVec2;

use crate::sim::{CHUNK_SIZE, ChunkCoord, ZoneId};

/// What a GPU hit ID stands for. The picker resolves a sampled u32 to one
/// of these (or `None` for the no-hit sentinel `0`) by consulting the
/// [`FrameIdTable`] that was current when the readback was submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HitTarget {
    /// A terrain tile within a specific zone.
    TerrainCell { zone: ZoneId, cell: IVec2 },
    // PR 3 will add Object { zone: ZoneId, id: WorldObjectId } here.
}

/// One terrain chunk's reservation in a [`FrameIdTable`]. A chunk reserves
/// `CHUNK_SIZE²` sequential IDs starting at `base_id`; the cell at local
/// position `(lx, ly)` has ID `base_id + ly * CHUNK_SIZE + lx`.
#[derive(Clone, Copy, Debug)]
struct TerrainChunkEntry {
    base_id: u32,
    zone: ZoneId,
    chunk_coord: ChunkCoord,
}

/// Cells per chunk — the size of one chunk's reservation in the ID space.
const CELLS_PER_CHUNK: u32 = CHUNK_SIZE * CHUNK_SIZE;

/// Per-frame indirection from GPU hit IDs back to [`HitTarget`]s.
///
/// Built up during the frame: callers (engine-side terrain/object renderers)
/// reserve IDs as they record draws; the rendered ID buffer holds those IDs
/// per pixel. When readback completes (1–3 frames later) the captured
/// snapshot of the table that was current when *that* frame was submitted
/// is what resolves the sampled u32 — so the table is cloned into each
/// readback slot rather than mutated in place across frames.
#[derive(Clone, Default)]
pub struct FrameIdTable {
    terrain_chunks: Vec<TerrainChunkEntry>,
}

impl FrameIdTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.terrain_chunks.clear();
    }

    /// Reserve `CELLS_PER_CHUNK` IDs for the given chunk's cells. Returns
    /// the base ID; cells at local `(lx, ly)` (0..`CHUNK_SIZE`) resolve to
    /// `base_id + ly * CHUNK_SIZE + lx`.
    ///
    /// Allocations begin at 1; ID 0 is reserved as the no-hit sentinel
    /// (the clear value of the ID attachment).
    pub fn reserve_terrain_chunk(&mut self, zone: ZoneId, chunk_coord: ChunkCoord) -> u32 {
        let next = self
            .terrain_chunks
            .last()
            .map(|e| e.base_id + CELLS_PER_CHUNK)
            .unwrap_or(1);
        self.terrain_chunks.push(TerrainChunkEntry {
            base_id: next,
            zone,
            chunk_coord,
        });
        next
    }

    /// Resolve a u32 sampled from the ID buffer. Returns `None` for the
    /// no-hit sentinel (`0`) or any ID outside reserved ranges (defensive —
    /// shouldn't happen if the writers stay consistent with the readers).
    pub fn resolve(&self, id: u32) -> Option<HitTarget> {
        if id == 0 {
            return None;
        }
        for entry in &self.terrain_chunks {
            let local = id.checked_sub(entry.base_id)?;
            if local < CELLS_PER_CHUNK {
                let lx = (local % CHUNK_SIZE) as i32;
                let ly = (local / CHUNK_SIZE) as i32;
                let cell = entry.chunk_coord * (CHUNK_SIZE as i32) + IVec2::new(lx, ly);
                return Some(HitTarget::TerrainCell {
                    zone: entry.zone,
                    cell,
                });
            }
        }
        None
    }
}

/// Number of in-flight readback slots. 3 is the standard "deeper than the
/// driver's typical 2-frame pipeline" choice — any of them might be busy
/// in any given frame, so 3 leaves one likely free without paying for more.
const READBACK_SLOTS: usize = 3;

/// Bytes wgpu allocates per row in a `copy_texture_to_buffer` destination.
/// Spec requires `bytes_per_row` to be a multiple of
/// `COPY_BYTES_PER_ROW_ALIGNMENT` (256); for a 1×1 R32Uint copy that's the
/// full row, even though we only read the first 4 bytes.
const READBACK_ROW_BYTES: u64 = 256;

struct ReadbackSlot {
    buffer: wgpu::Buffer,
    in_flight: Arc<AtomicBool>,
}

/// Ring of staging buffers + a `map_async`-fed latest-result slot. One
/// instance per renderer; lives on `SceneResources`.
///
/// Per-frame use is a two-step dance because wgpu requires `map_async` to
/// be called *after* the submission that writes the staging buffer:
///
/// 1. While encoding, [`Self::enqueue_copy`] records the
///    `copy_texture_to_buffer` and marks the chosen slot in flight.
/// 2. After `queue.submit`, [`Self::schedule_readback`] calls `map_async`
///    on that slot with a snapshot of the frame's [`FrameIdTable`]. The
///    callback fires on a later `device.poll` and resolves the sampled
///    `u32` into [`Self::latest`].
pub(super) struct IdReadback {
    slots: [ReadbackSlot; READBACK_SLOTS],
    latest: Arc<Mutex<Option<HitTarget>>>,
    /// Slot index whose copy was recorded into the *most recent* encoder
    /// and which is now awaiting [`Self::schedule_readback`]. `None`
    /// between frames or when no copy was enqueued (cursor out of bounds
    /// / all slots in flight).
    pending: Mutex<Option<usize>>,
}

impl IdReadback {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let make_slot = |i: usize| ReadbackSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("currawong hit-id readback slot {i}")),
                size: READBACK_ROW_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            in_flight: Arc::new(AtomicBool::new(false)),
        };
        Self {
            slots: [make_slot(0), make_slot(1), make_slot(2)],
            latest: Arc::new(Mutex::new(None)),
            pending: Mutex::new(None),
        }
    }

    /// Latest hit target delivered by a completed readback. `None` either
    /// because no readback has landed yet, the most recent sampled pixel
    /// was the no-hit sentinel, or the cursor left the window.
    pub(super) fn latest(&self) -> Option<HitTarget> {
        *self.latest.lock().expect("hit-id mutex poisoned")
    }

    /// Force the latest result to `None` — call when the cursor leaves the
    /// window so stale hits don't persist after the input source stops
    /// updating.
    pub(super) fn clear_latest(&self) {
        *self.latest.lock().expect("hit-id mutex poisoned") = None;
    }

    /// Step 1: pick a free slot and record a 1×1 copy of the ID texture
    /// under `(cursor_x, cursor_y)` into it. Silently no-ops if every slot
    /// is in flight — picker fallback covers the gap. Must be paired with
    /// a [`Self::schedule_readback`] call *after* the encoder is
    /// submitted; the pending-slot record is consumed there.
    pub(super) fn enqueue_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        id_texture: &wgpu::Texture,
        cursor_x: u32,
        cursor_y: u32,
    ) {
        let Some(slot_idx) = self.acquire_slot_idx() else {
            return;
        };
        let slot = &self.slots[slot_idx];
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: id_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: cursor_x,
                    y: cursor_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(READBACK_ROW_BYTES as u32),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        *self.pending.lock().expect("pending poisoned") = Some(slot_idx);
    }

    /// Step 2: after the encoder containing the [`Self::enqueue_copy`]
    /// command has been submitted, schedule a `map_async` on the pending
    /// slot. The callback resolves the sampled u32 through
    /// `table_snapshot` and updates [`Self::latest`]. No-ops when no copy
    /// was enqueued this frame.
    pub(super) fn schedule_readback(&self, table_snapshot: FrameIdTable) {
        let slot_idx = match self.pending.lock().expect("pending poisoned").take() {
            Some(i) => i,
            None => return,
        };
        let slot = &self.slots[slot_idx];
        // Capture clones for the callback. wgpu::Buffer is Arc-internally,
        // so this is one refcount bump per submission.
        let buffer = slot.buffer.clone();
        let in_flight = slot.in_flight.clone();
        let latest = self.latest.clone();
        slot.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    let view = buffer.slice(..).get_mapped_range();
                    // First 4 bytes of a R32Uint copy at (0, 0) are the
                    // pixel value little-endian on every target wgpu
                    // supports.
                    let bytes: [u8; 4] = view[0..4]
                        .try_into()
                        .expect("readback buffer has at least 4 bytes");
                    drop(view);
                    buffer.unmap();
                    let id = u32::from_le_bytes(bytes);
                    let resolved = table_snapshot.resolve(id);
                    if let Ok(mut latest) = latest.lock() {
                        *latest = resolved;
                    }
                }
                in_flight.store(false, Ordering::Release);
            });
    }

    fn acquire_slot_idx(&self) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| !s.in_flight.swap(true, Ordering::AcqRel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{SlotKey, ZoneId};

    fn zone(idx: u32) -> ZoneId {
        ZoneId::from_raw(idx, 0)
    }

    #[test]
    fn resolve_no_hit_returns_none() {
        let table = FrameIdTable::new();
        assert_eq!(table.resolve(0), None);
    }

    #[test]
    fn first_reservation_starts_at_one() {
        let mut table = FrameIdTable::new();
        let base = table.reserve_terrain_chunk(zone(0), ChunkCoord::new(0, 0));
        assert_eq!(base, 1);
    }

    #[test]
    fn reservations_pack_sequentially() {
        let mut table = FrameIdTable::new();
        let a = table.reserve_terrain_chunk(zone(0), ChunkCoord::new(0, 0));
        let b = table.reserve_terrain_chunk(zone(0), ChunkCoord::new(1, 0));
        assert_eq!(a, 1);
        assert_eq!(b, 1 + CELLS_PER_CHUNK);
    }

    fn unwrap_cell(target: Option<HitTarget>) -> (ZoneId, IVec2) {
        match target.expect("expected a hit target") {
            HitTarget::TerrainCell { zone, cell } => (zone, cell),
        }
    }

    #[test]
    fn resolve_finds_cells_within_a_chunk() {
        // Reserve one chunk at world origin; resolving the chunk's base ID
        // should give the (0, 0) cell of that chunk; base + 1 should give
        // (1, 0); base + CHUNK_SIZE should give (0, 1).
        let mut table = FrameIdTable::new();
        let base = table.reserve_terrain_chunk(zone(7), ChunkCoord::new(0, 0));

        let (z, cell) = unwrap_cell(table.resolve(base));
        assert_eq!(z, zone(7));
        assert_eq!(cell, IVec2::new(0, 0));

        let (_, cell) = unwrap_cell(table.resolve(base + 1));
        assert_eq!(cell, IVec2::new(1, 0));

        let (_, cell) = unwrap_cell(table.resolve(base + CHUNK_SIZE));
        assert_eq!(cell, IVec2::new(0, 1));
    }

    #[test]
    fn resolve_respects_chunk_coord_offset() {
        // A chunk at chunk_coord = (2, -1) should map its (0, 0) local cell
        // to world cell (2 * CHUNK_SIZE, -1 * CHUNK_SIZE).
        let mut table = FrameIdTable::new();
        let base = table.reserve_terrain_chunk(zone(0), ChunkCoord::new(2, -1));
        let (_, cell) = unwrap_cell(table.resolve(base));
        assert_eq!(
            cell,
            IVec2::new(2 * CHUNK_SIZE as i32, -(CHUNK_SIZE as i32)),
        );
    }

    #[test]
    fn resolve_returns_none_for_ids_past_the_end() {
        let mut table = FrameIdTable::new();
        let base = table.reserve_terrain_chunk(zone(0), ChunkCoord::ZERO);
        assert!(table.resolve(base + CELLS_PER_CHUNK).is_none());
        assert!(table.resolve(u32::MAX).is_none());
    }

    #[test]
    fn clear_resets_to_empty() {
        let mut table = FrameIdTable::new();
        table.reserve_terrain_chunk(zone(0), ChunkCoord::ZERO);
        table.clear();
        assert_eq!(table.resolve(1), None);
        // After clear, the next reservation should restart at 1 again.
        let base = table.reserve_terrain_chunk(zone(0), ChunkCoord::new(5, 5));
        assert_eq!(base, 1);
    }
}
