//! Per-frame CPU + GPU timing breakdown exposed to View callbacks via
//! [`EngineCtx::timings`](super::EngineCtx).
//!
//! The engine measures each segment of the per-frame work and snapshots the
//! result onto [`EngineCtx`](super::EngineCtx) before calling user callbacks.
//! The intended consumer is a debug overlay (see the lumber_camp example);
//! gameplay code should ignore these numbers.

use std::time::Duration;

/// Per-frame CPU + GPU timing breakdown.
///
/// All measurements reflect the *previous* completed frame. CPU figures
/// are wall-clock spans recorded by the engine around the sim tick loop
/// and the main render pass; [`gpu`](Self::gpu) carries per-segment GPU
/// durations from wgpu timestamp queries.
#[derive(Clone, Copy, Default, Debug)]
pub struct FrameTimings {
    /// Wall-clock time spent in `Simulation::apply_command` + `Simulation::tick`
    /// during the previous frame's tick loop. Zero on frames that fired no
    /// ticks (paused sim, sub-tick frame intervals).
    pub sim_cpu: Duration,
    /// Wall-clock time spent in [`View::render`](super::View::render) on the
    /// previous frame.
    pub render_cpu: Duration,
    /// GPU pass durations from the timestamp-query ring. Per-segment fields
    /// are independently `Option` — `world` populates when the adapter
    /// exposes [`wgpu::Features::TIMESTAMP_QUERY`], `overlay` additionally
    /// needs [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`].
    pub gpu: GpuSegments,
}

/// Per-segment GPU pass durations.
///
/// `None` per segment means "the adapter / device doesn't support the
/// timestamp variant this segment relies on" *or* "no readback has landed
/// yet on the first few frames." The two cases aren't distinguished — a
/// debug overlay should render them the same way (e.g. "— ms").
#[derive(Clone, Copy, Default, Debug)]
pub struct GpuSegments {
    /// Duration of the main world render pass (terrain + meshes,
    /// everything inside the engine's `currawong frame pass`).
    pub world: Option<Duration>,
    /// Combined duration of the yakui and egui overlay passes. Needs
    /// [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`] because yakui's
    /// pass is owned by `yakui_wgpu::paint_with_encoder` and can't take
    /// pass-level timestamp writes — the engine brackets it with
    /// encoder-level `write_timestamp` calls instead.
    pub overlay: Option<Duration>,
}

impl GpuSegments {
    /// Sum of populated segments, or `None` if every segment is `None`.
    pub fn total(&self) -> Option<Duration> {
        match (self.world, self.overlay) {
            (Some(w), Some(o)) => Some(w + o),
            (Some(d), None) | (None, Some(d)) => Some(d),
            (None, None) => None,
        }
    }
}
