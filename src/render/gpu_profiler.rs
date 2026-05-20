//! GPU timestamp queries broken out into named segments.
//!
//! Each segment is a `(begin, end)` pair of timestamps in a shared
//! [`wgpu::QuerySet`]. The world segment uses *pass* timestamp writes
//! (`RenderPassTimestampWrites` on the main pass descriptor) and is
//! available whenever the adapter exposes
//! [`wgpu::Features::TIMESTAMP_QUERY`]. The overlay segment uses
//! *encoder-level* `write_timestamp` calls around the egui + yakui passes
//! and needs [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`] on top —
//! when missing, `overlay` stays `None` and the rest still works.
//!
//! Shape mirrors [`super::picking_buffer::IdReadback`]: a small ring of
//! resolve/readback buffer pairs, a one-frame-pending handoff between
//! encoding and submit, and `map_async` callbacks resolving into a shared
//! latest-result slot. 1–3 frames of latency between submit and the value
//! showing up — fine for a debug overlay.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use super::frame_timings::GpuSegments;

const SLOT_COUNT: usize = 3;
/// Per-segment slot indices. Two timestamps per segment (begin + end).
const WORLD_BEGIN: u32 = 0;
const WORLD_END: u32 = 1;
const OVERLAY_BEGIN: u32 = 2;
const OVERLAY_END: u32 = 3;
const TIMESTAMP_COUNT: u32 = 4;
const RESOLVE_BYTES: u64 = TIMESTAMP_COUNT as u64 * 8;

/// One slot of the ring — `resolve` is the `QUERY_RESOLVE` target the
/// `resolve_query_set` writes into; `readback` is the `MAP_READ` copy.
struct Slot {
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    in_flight: Arc<AtomicBool>,
}

/// Latest readback values as raw query ticks. `Option` per segment because
/// the overlay segment isn't populated when `INSIDE_ENCODERS` is absent.
#[derive(Clone, Copy, Default)]
struct LatestTicks {
    world: Option<u64>,
    overlay: Option<u64>,
}

/// Multi-segment GPU timing.
///
/// Constructed via [`Self::new`] when the adapter has `TIMESTAMP_QUERY`;
/// pass `overlay_supported = true` only if `TIMESTAMP_QUERY_INSIDE_ENCODERS`
/// is also available. The runner queries [`Self::overlay_supported`] before
/// trying to write overlay timestamps.
pub(super) struct GpuProfiler {
    query_set: wgpu::QuerySet,
    slots: [Slot; SLOT_COUNT],
    /// Nanoseconds per query tick — read once at construction from the queue.
    period_ns: f32,
    /// Last completed measurement in raw query ticks. `map_async` callbacks
    /// write here; readers convert to a [`Duration`] in [`Self::latest`].
    latest_ticks: Arc<Mutex<LatestTicks>>,
    /// Slot index whose resolve was recorded into the most recent encoder
    /// and which is now awaiting [`Self::schedule_readback`].
    pending: Mutex<Option<usize>>,
    /// Whether the overlay segment's encoder-level timestamps are usable.
    overlay_supported: bool,
}

impl GpuProfiler {
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue, overlay_supported: bool) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("currawong gpu-profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: TIMESTAMP_COUNT,
        });
        let make_slot = |i: usize| Slot {
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("currawong gpu-profiler resolve {i}")),
                size: RESOLVE_BYTES,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("currawong gpu-profiler readback {i}")),
                size: RESOLVE_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            in_flight: Arc::new(AtomicBool::new(false)),
        };
        Self {
            query_set,
            slots: [make_slot(0), make_slot(1), make_slot(2)],
            period_ns: queue.get_timestamp_period(),
            latest_ticks: Arc::new(Mutex::new(LatestTicks::default())),
            pending: Mutex::new(None),
            overlay_supported,
        }
    }

    /// Timestamp-writes block to plug into the world pass's
    /// [`wgpu::RenderPassDescriptor`].
    pub(super) fn world_pass_timestamps(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(WORLD_BEGIN),
            end_of_pass_write_index: Some(WORLD_END),
        }
    }

    /// Write the overlay-begin timestamp directly onto the encoder. No-op
    /// when `INSIDE_ENCODERS` wasn't supported at device creation.
    pub(super) fn write_overlay_begin(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.overlay_supported {
            encoder.write_timestamp(&self.query_set, OVERLAY_BEGIN);
        }
    }

    /// Write the overlay-end timestamp directly onto the encoder. No-op
    /// when `INSIDE_ENCODERS` wasn't supported at device creation.
    pub(super) fn write_overlay_end(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.overlay_supported {
            encoder.write_timestamp(&self.query_set, OVERLAY_END);
        }
    }

    /// After all the timestamp writes for the frame are recorded (the world
    /// pass has dropped and the overlays are done), resolve every timestamp
    /// into a free slot and stage them for readback. Silently skipped when
    /// every slot is in flight — the previous frame's values stay in
    /// `latest_ticks`.
    pub(super) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(slot_idx) = self.acquire_slot_idx() else {
            return;
        };
        let slot = &self.slots[slot_idx];
        encoder.resolve_query_set(&self.query_set, 0..TIMESTAMP_COUNT, &slot.resolve, 0);
        encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, RESOLVE_BYTES);
        *self.pending.lock().expect("gpu-profiler pending poisoned") = Some(slot_idx);
    }

    /// After `queue.submit`, kick off the `map_async` on the pending slot.
    /// The callback fires on a later `device.poll` and updates
    /// `latest_ticks`.
    pub(super) fn schedule_readback(&self) {
        let slot_idx = match self
            .pending
            .lock()
            .expect("gpu-profiler pending poisoned")
            .take()
        {
            Some(i) => i,
            None => return,
        };
        let slot = &self.slots[slot_idx];
        let buffer = slot.readback.clone();
        let in_flight = slot.in_flight.clone();
        let latest = self.latest_ticks.clone();
        let overlay_supported = self.overlay_supported;
        slot.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    let view = buffer.slice(..).get_mapped_range();
                    let read_u64 = |start: usize| {
                        u64::from_le_bytes(
                            view[start..start + 8]
                                .try_into()
                                .expect("gpu-profiler readback has 32 bytes"),
                        )
                    };
                    let world_begin = read_u64(0);
                    let world_end = read_u64(8);
                    let overlay_begin = read_u64(16);
                    let overlay_end = read_u64(24);
                    drop(view);
                    buffer.unmap();
                    let world_delta = world_end.saturating_sub(world_begin);
                    let overlay_delta = if overlay_supported {
                        // Encoder timestamps overlap with the world pass's
                        // pass-level timestamps on the same query set;
                        // a spurious zero (or a wrapping subtract) shows up
                        // as `None` rather than a misleading 0 ns.
                        let d = overlay_end.saturating_sub(overlay_begin);
                        (d > 0).then_some(d)
                    } else {
                        None
                    };
                    if let Ok(mut latest) = latest.lock() {
                        latest.world = Some(world_delta);
                        latest.overlay = overlay_delta;
                    }
                }
                in_flight.store(false, Ordering::Release);
            });
    }

    /// Latest per-segment GPU durations.
    pub(super) fn latest(&self) -> GpuSegments {
        let ticks = *self
            .latest_ticks
            .lock()
            .expect("gpu-profiler latest poisoned");
        let to_duration = |t: u64| {
            let ns = (t as f64) * (self.period_ns as f64);
            Duration::from_nanos(ns as u64)
        };
        GpuSegments {
            world: ticks.world.map(to_duration),
            overlay: ticks.overlay.map(to_duration),
        }
    }

    fn acquire_slot_idx(&self) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| !s.in_flight.swap(true, Ordering::AcqRel))
    }
}
