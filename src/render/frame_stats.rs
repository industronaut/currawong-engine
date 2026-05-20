//! Per-frame CPU-side counters surfaced on [`EngineCtx::stats`](super::EngineCtx).
//!
//! Numbers that explain frametime variance — draw calls, instance counts,
//! live render-object proxies — without doing any GPU work to collect them.
//! Populated by the view as it records draws (via
//! [`Renderer::record_draw`](super::Renderer::record_draw)) and snapshotted
//! by the engine at the end of each frame for the next frame's callbacks.
//!
//! Unlike [`FrameTimings`](super::FrameTimings) which is engine-driven, these
//! counters require explicit cooperation from view code: the engine has no
//! way to intercept `pass.draw_indexed` calls, so the view bumps the counter
//! once per draw. The trade-off is engine simplicity at the cost of "if the
//! view forgets to call `record_draw`, the number is wrong" — same shape as
//! every renderer's stats overlay.

/// Per-frame CPU-side render stats.
///
/// All fields reflect the *previous* completed frame. Zero on the very first
/// frame and on frames where the view never called the corresponding
/// `record_*` method.
#[derive(Clone, Copy, Default, Debug)]
pub struct FrameStats {
    /// Number of `pass.draw*` calls the view issued during
    /// [`View::render`](super::View::render). View code reports each draw
    /// through [`Renderer::record_draw`](super::Renderer::record_draw).
    pub draw_calls: u32,
    /// Sum of the per-draw instance counts — i.e. total mesh copies sent
    /// to the GPU. Reported alongside `draw_calls`.
    pub instances: u32,
    /// Live render-object proxies inside the camera frustum this frame.
    /// Reported by the view through
    /// [`Renderer::record_proxies`](super::Renderer::record_proxies); the
    /// numbers usually come from
    /// [`LiveRenderObjects::cull_counts`](super::LiveRenderObjects::cull_counts).
    pub proxies_visible: u32,
    /// Live render-object proxies *outside* the frustum but still inside
    /// the hysteresis grace window — drawn this frame on borrowed time.
    /// Same source as [`Self::proxies_visible`].
    pub proxies_hysteresis: u32,
}
