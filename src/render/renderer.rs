//! Window + GPU surface, device, queue, and engine-managed scene resources.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use winit::window::Window;

use crate::sim::{ChunkCoord, WorldObjectId, ZoneId};

use super::environment::ViewEnvironment;
use super::frame_stats::FrameStats;
use super::gpu_profiler::GpuProfiler;
use super::picking_buffer::HitTarget;
use super::scene_resources::SceneResources;

/// Owns the window and the wgpu device/queue/surface, plus engine-managed
/// per-scene GPU resources (depth attachment, scene-environment binding,
/// future shadow maps, IBL probes, …) via [`SceneResources`].
pub struct Renderer {
    pub window: Arc<Window>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub(super) surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    scene: SceneResources,
    /// `Some` when the adapter exposed
    /// [`wgpu::Features::TIMESTAMP_QUERY`] at device-creation time. Drives
    /// the GPU column of the debug overlay's
    /// [`FrameTimings`](super::FrameTimings).
    gpu_profiler: Option<GpuProfiler>,
    /// Snapshot of [`wgpu::Adapter::get_info`] from device creation —
    /// backend, GPU name, vendor / driver strings. Surfaced through
    /// [`Self::adapter_info`] for debug overlays and crash reports.
    adapter_info: wgpu::AdapterInfo,
    /// Atomic counters the view bumps via [`Self::record_draw`] /
    /// [`Self::record_proxies`] during `render`. Snapshotted + reset by
    /// the runner at frame boundaries; surfaces through
    /// [`EngineCtx::stats`](super::EngineCtx).
    frame_stats: FrameStatsCollector,
}

/// Atomic backing for [`Renderer::record_draw`] / [`Renderer::record_proxies`].
/// Interior mutability so the `&Renderer` the view sees during `render` can
/// still update the counters; the runner takes a snapshot and resets at the
/// frame boundary.
struct FrameStatsCollector {
    draw_calls: AtomicU32,
    instances: AtomicU32,
    proxies_visible: AtomicU32,
    proxies_hysteresis: AtomicU32,
}

impl FrameStatsCollector {
    fn new() -> Self {
        Self {
            draw_calls: AtomicU32::new(0),
            instances: AtomicU32::new(0),
            proxies_visible: AtomicU32::new(0),
            proxies_hysteresis: AtomicU32::new(0),
        }
    }

    fn record_draw(&self, instances: u32) {
        self.draw_calls.fetch_add(1, Ordering::Relaxed);
        self.instances.fetch_add(instances, Ordering::Relaxed);
    }

    fn record_proxies(&self, visible: u32, hysteresis: u32) {
        self.proxies_visible.store(visible, Ordering::Relaxed);
        self.proxies_hysteresis.store(hysteresis, Ordering::Relaxed);
    }

    /// Read all four counters and zero `draw_calls` / `instances` for the
    /// next frame. Proxy counts are *overwritten* by the next
    /// `record_proxies` so they stay at their last value if the view skips
    /// a call (e.g. a frame where the proxy table didn't update).
    fn take_and_reset(&self) -> FrameStats {
        FrameStats {
            draw_calls: self.draw_calls.swap(0, Ordering::Relaxed),
            instances: self.instances.swap(0, Ordering::Relaxed),
            proxies_visible: self.proxies_visible.load(Ordering::Relaxed),
            proxies_hysteresis: self.proxies_hysteresis.load(Ordering::Relaxed),
        }
    }
}

impl Renderer {
    /// Format pipelines targeting the swapchain should declare.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// GPU adapter snapshot from device creation — backend, device name,
    /// vendor / driver strings. Useful for debug overlays and crash
    /// reports; remains stable for the lifetime of the [`Renderer`].
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Record one draw call rendering `instances` instances. Bumps the
    /// per-frame counters surfaced through
    /// [`EngineCtx::stats`](super::EngineCtx). Call once per
    /// `pass.draw*`; cost is two relaxed atomic increments.
    pub fn record_draw(&self, instances: u32) {
        self.frame_stats.record_draw(instances);
    }

    /// Record live render-object proxy counts for this frame. `visible` is
    /// the count inside the frustum; `hysteresis` is the count outside but
    /// still inside the grace window. Conventional source is
    /// [`RenderProxies::cull_counts`](super::RenderProxies::cull_counts).
    /// Counts are stored as-written (not accumulated), so call exactly
    /// once per frame after culling.
    pub fn record_proxies(&self, visible: u32, hysteresis: u32) {
        self.frame_stats.record_proxies(visible, hysteresis);
    }

    /// Snapshot the per-frame counters and zero the per-draw fields. Called
    /// by the runner at frame boundaries; not normally needed by views.
    pub(super) fn take_frame_stats(&self) -> FrameStats {
        self.frame_stats.take_and_reset()
    }

    /// Current swapchain size in pixels. Useful for screen-space rendering
    /// (e.g. egui's `ScreenDescriptor`).
    pub fn surface_size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Depth format the engine has allocated for this view, if any. Pipelines
    /// using depth must declare this format in their `DepthStencilState`.
    pub fn depth_format(&self) -> Option<wgpu::TextureFormat> {
        self.scene.depth_format()
    }

    /// Depth texture view, if a depth attachment was requested. Used by the
    /// engine's frame loop to attach to the render pass — not normally
    /// needed by views.
    pub(super) fn depth_view(&self) -> Option<&wgpu::TextureView> {
        self.scene.depth_view()
    }

    /// Format of the engine's hit-ID attachment. Pipelines drawn in the
    /// opaque pass must declare a target at `targets[1]` — either writing
    /// per-pixel hit IDs in this format, or opting out via the no-write
    /// pattern returned by [`id_target_opt_out`](Self::id_target_opt_out).
    /// The buffer is cleared to `0` at the start of every frame; `0` is the
    /// no-hit sentinel. See issue #56 for the full picking design.
    pub fn id_format(&self) -> wgpu::TextureFormat {
        self.scene.id_format()
    }

    /// `ColorTargetState` for the hit-ID slot that declares the format but
    /// disables writes — the right value for any pipeline drawn in the
    /// opaque pass that doesn't participate in picking (debug overlays,
    /// decorative geometry, transparent particles, …).
    ///
    /// wgpu requires `targets[i]` formats to match the render pass's
    /// `color_attachments[i]` formats; setting the write mask to empty is
    /// how a pipeline declares "leave whatever's there alone" so existing
    /// IDs from earlier opaque draws survive under decorative pixels.
    pub fn id_target_opt_out(&self) -> Option<wgpu::ColorTargetState> {
        Some(wgpu::ColorTargetState {
            format: self.id_format(),
            blend: None,
            write_mask: wgpu::ColorWrites::empty(),
        })
    }

    /// `ColorTargetState` for pipelines that *write* to the hit-ID slot —
    /// terrain in PR 2, opt-in mesh materials in PR 3. R32Uint format,
    /// write-mask ALL, no blend (integer formats can't be blended).
    pub fn id_target_writer(&self) -> Option<wgpu::ColorTargetState> {
        Some(wgpu::ColorTargetState {
            format: self.id_format(),
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })
    }

    /// Bind-group layout for the per-draw "ID base" uniform that pipelines
    /// participating in picking declare alongside their normal bind groups.
    /// Holds a single `u32`; terrain uses one of these per chunk, written
    /// to the bound buffer at draw time with the chunk's base hit ID. Pass
    /// to [`PipelineLayoutDescriptor`](wgpu::PipelineLayoutDescriptor) when
    /// building any pipeline that writes to the hit-ID attachment via a
    /// per-draw uniform.
    pub fn id_base_layout(&self) -> &wgpu::BindGroupLayout {
        self.scene.id_base_layout()
    }

    /// Reserve a chunk's worth of hit IDs in the current frame's table.
    /// Engine renderers (and, eventually, user-side mesh renderers) call
    /// this once per drawn chunk; the returned `base_id` is written into
    /// the chunk's per-frame uniform and added to the per-vertex
    /// chunk-local cell index in the shader so each pixel emits a unique
    /// frame-scoped hit ID. See [`Self::hit_id_hover`] for the readback
    /// side.
    pub fn reserve_terrain_chunk(&self, zone: ZoneId, chunk: ChunkCoord) -> u32 {
        self.scene.reserve_terrain_chunk(zone, chunk)
    }

    /// Reserve a single hit ID for a mesh object. Returns the ID, which
    /// callers write into the per-instance attributes of every pickable
    /// draw for that object — typically the `hit_id` field on
    /// [`MeshInstanceAttribs`](super::MeshInstanceAttribs) via
    /// [`MeshInstanceAttribs::with_hit_id`](super::MeshInstanceAttribs::with_hit_id).
    /// The engine's `R32Uint` attachment then holds this ID at every pixel
    /// the object covers.
    ///
    /// All mesh parts of one sim object should share a single hit ID — see
    /// [`RenderObjectTraversal::for_each_alive_with_hit_id`](super::RenderObjectTraversal::for_each_alive_with_hit_id)
    /// for the engine-driven path that reserves once per parent so a click
    /// on any part resolves to the same `WorldObjectId`. Users walking the
    /// sim by hand should reserve once per parent themselves.
    pub fn reserve_object(&self, zone: ZoneId, id: WorldObjectId) -> u32 {
        self.scene.reserve_object(zone, id)
    }

    /// Latest hit target delivered by the readback ring, or `None` if the
    /// most recent sampled pixel was the no-hit sentinel, no readback has
    /// landed yet, or the cursor has left the window. `View::update` is
    /// the typical reader — call it once per frame and feed the result
    /// into [`TerrainPicker::set_id_hover`](super::TerrainPicker::set_id_hover).
    pub fn hit_id_hover(&self) -> Option<HitTarget> {
        self.scene.hit_id_hover()
    }

    /// Hit-ID texture handle — used by the engine's frame loop for the
    /// per-frame cursor-pixel `copy_texture_to_buffer`. Not normally needed
    /// by views.
    pub(super) fn id_texture(&self) -> &wgpu::Texture {
        self.scene.id_texture()
    }

    pub(super) fn reset_frame_id_table(&self) {
        self.scene.reset_frame_id_table();
    }

    pub(super) fn snapshot_frame_id_table(&self) -> super::picking_buffer::FrameIdTable {
        self.scene.snapshot_frame_id_table()
    }

    pub(super) fn id_readback(&self) -> &super::picking_buffer::IdReadback {
        self.scene.id_readback()
    }

    /// Clear the latest hit-id result — runner calls this on `CursorLeft`
    /// so a stale hover doesn't persist after the input source stops
    /// updating.
    pub(super) fn clear_hit_id_hover(&self) {
        self.scene.id_readback().clear_latest();
    }

    /// Hit-ID texture view. Used by the engine's frame loop to attach to the
    /// opaque pass — not normally needed by views.
    pub(super) fn id_view(&self) -> &wgpu::TextureView {
        self.scene.id_view()
    }

    /// Bind-group layout for the engine-managed scene environment uniform.
    /// Pass to [`PipelineLayoutDescriptor`](wgpu::PipelineLayoutDescriptor)
    /// when building any pipeline that reads scene lighting (sun direction,
    /// ambient, etc.). The corresponding bind group is bound by the engine
    /// before [`View::render`](crate::View::render) runs — your shader just
    /// needs to declare a `@group(N) @binding(0)` for it.
    pub fn scene_layout(&self) -> &wgpu::BindGroupLayout {
        self.scene.scene_layout()
    }

    /// Bind group for the engine-managed scene environment uniform. Bind at
    /// the slot your pipeline reserved for [`scene_layout`](Self::scene_layout).
    /// The engine writes fresh values into it each frame from
    /// [`View::extract_environment`](crate::View::extract_environment).
    pub fn scene_bind_group(&self) -> &wgpu::BindGroup {
        self.scene.scene_bind_group()
    }

    pub(super) fn write_scene(&self, env: &ViewEnvironment) {
        self.scene.write_scene(&self.queue, env);
    }

    /// Timestamp-writes block for the main world pass, or `None` when the
    /// GPU profiler is disabled. Used by the runner; not normally needed by
    /// views.
    pub(super) fn gpu_profiler_world_pass_timestamps(
        &self,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.gpu_profiler
            .as_ref()
            .map(|p| p.world_pass_timestamps())
    }

    /// Write the overlay-begin timestamp on the encoder, just before yakui
    /// + egui run. No-op when `TIMESTAMP_QUERY_INSIDE_ENCODERS` is absent.
    pub(super) fn gpu_profiler_overlay_begin(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(profiler) = self.gpu_profiler.as_ref() {
            profiler.write_overlay_begin(encoder);
        }
    }

    /// Write the overlay-end timestamp on the encoder, just after egui
    /// finishes. Pairs with [`Self::gpu_profiler_overlay_begin`].
    pub(super) fn gpu_profiler_overlay_end(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(profiler) = self.gpu_profiler.as_ref() {
            profiler.write_overlay_end(encoder);
        }
    }

    /// Resolve the frame's timestamp pairs into a readback slot. Called
    /// by the runner after all timestamps have been written (world pass
    /// closed, overlays done).
    pub(super) fn gpu_profiler_resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(profiler) = self.gpu_profiler.as_ref() {
            profiler.resolve(encoder);
        }
    }

    /// Kick off `map_async` on the slot just resolved. Must run after the
    /// encoder has been submitted.
    pub(super) fn gpu_profiler_schedule_readback(&self) {
        if let Some(profiler) = self.gpu_profiler.as_ref() {
            profiler.schedule_readback();
        }
    }

    /// Latest per-segment GPU durations from the readback ring. All-`None`
    /// when the profiler is disabled or no readback has landed yet.
    pub(super) fn gpu_profiler_latest(&self) -> super::frame_timings::GpuSegments {
        self.gpu_profiler
            .as_ref()
            .map(|p| p.latest())
            .unwrap_or_default()
    }

    pub(super) async fn new(
        window: Arc<Window>,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let size = window.inner_size();

        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");

        // GPU profiler is currently disabled. Enabling `TIMESTAMP_QUERY` +
        // `TIMESTAMP_QUERY_INSIDE_ENCODERS` and wiring `RenderPassTimestampWrites`
        // / encoder-level `write_timestamp` (the shape implemented in
        // `super::gpu_profiler::GpuProfiler`) makes the lumber_camp frame
        // abort every submit under `METAL_DEVICE_WRAPPER_TYPE=1` with
        // `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Splitting world
        // and overlay into separate `QuerySet`s (so each MTLCounterSampleBuffer
        // is single-source) did *not* fix it, ruling out the pass-vs-encoder
        // sampling-mix hypothesis. Without the device wrapper Metal runs
        // fine. Until we understand what API Validation is rejecting we
        // skip the feature request and force `gpu_profiler = None` below,
        // so the overlay's GPU column reads "—" and no QuerySet is ever
        // allocated. For one-off profiling, use Xcode's GPU capture.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("currawong device"),
                ..Default::default()
            })
            .await
            .expect("failed to request device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let scene = SceneResources::new(&device, config.width, config.height, depth_format);
        // Force-disabled; see the comment on the device-request block above.
        let gpu_profiler: Option<GpuProfiler> = None;
        let adapter_info = adapter.get_info();

        Self {
            window,
            device,
            queue,
            surface,
            config,
            scene,
            gpu_profiler,
            adapter_info,
            frame_stats: FrameStatsCollector::new(),
        }
    }

    pub(super) fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        self.scene
            .resize(&self.device, self.config.width, self.config.height);
    }
}
