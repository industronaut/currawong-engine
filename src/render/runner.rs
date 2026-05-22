//! Event-loop integration: `run` and `run_with_clock` bind a [`View`] to a
//! `winit` window and a [`SimClock`].

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::sim::{CommandQueue, SimClock, Simulation};

#[cfg(feature = "egui")]
use super::debug_ui::DebugUi;
use super::environment::ViewEnvironment;
use super::frame_stats::FrameStats;
use super::frame_timings::FrameTimings;
#[cfg(feature = "yakui")]
use super::game_ui::GameUi;
use super::renderer::Renderer;
use super::screenshot::ScreenshotRequest;
use super::view::{EngineCtx, View};

/// Run an application with the given simulation. Uses [`SimClock::new`] —
/// 60 Hz fixed tick at speed 1.0.
pub fn run<V: View>(sim: V::Sim) {
    run_with_clock::<V>(sim, SimClock::new());
}

/// Run an application with a custom [`SimClock`].
pub fn run_with_clock<V: View>(sim: V::Sim, clock: SimClock) {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut handler = Handler::<V> {
        sim: Some(sim),
        clock: Some(clock),
        state: None,
        cmds: Some(CommandQueue::new()),
    };
    event_loop
        .run_app(&mut handler)
        .expect("event loop terminated unexpectedly");
}

struct Handler<V: View> {
    sim: Option<V::Sim>,
    clock: Option<SimClock>,
    cmds: Option<CommandQueue<<V::Sim as Simulation>::Command>>,
    state: Option<RunState<V>>,
}

struct RunState<V: View> {
    renderer: Renderer,
    view: V,
    sim: V::Sim,
    clock: SimClock,
    cmds: CommandQueue<<V::Sim as Simulation>::Command>,
    last_redraw: Instant,
    /// Latest cursor position in physical pixels; `None` while the cursor
    /// is outside the window. Drives the per-frame hit-ID readback
    /// (`copy_texture_to_buffer` at this pixel). Tracked independently from
    /// any view-side picker because the readback is engine-managed.
    cursor_px: Option<(u32, u32)>,
    /// Previous frame's per-segment timing. Refreshed by `render_frame` and
    /// snapshotted onto every [`EngineCtx`] the runner hands to user
    /// callbacks; debug overlays read it via `ctx.timings`. CPU figures
    /// are wall-clock around the sim tick loop and the main render pass;
    /// `gpu` reflects whatever the wgpu timestamp-query ring has finished
    /// reading back (1–3 frames behind the latest submit).
    timings: FrameTimings,
    /// Previous frame's CPU counters — draws, instances, proxies — taken
    /// from [`Renderer::take_frame_stats`] at the end of `render_frame`.
    /// Mirrored onto every [`EngineCtx`] alongside [`Self::timings`].
    stats: FrameStats,
    /// F12 was pressed since the last frame finished — capture this frame's
    /// swapchain image into a screenshot. Cleared after the capture is
    /// scheduled (regardless of whether the save itself succeeds). The
    /// engine intercepts F12 before [`View::input`] sees it, so View code
    /// can't accidentally suppress or duplicate the trigger.
    pending_screenshot: bool,
    #[cfg(feature = "egui")]
    debug_ui: DebugUi,
    #[cfg(feature = "yakui")]
    game_ui: GameUi,
}

impl<V: View> ApplicationHandler for Handler<V> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title(V::CONFIG.title);
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );
        let renderer = pollster::block_on(Renderer::new(
            window,
            V::CONFIG.depth_format,
            V::CONFIG.shadow_map_resolution,
        ));
        #[cfg(feature = "egui")]
        let debug_ui = DebugUi::new(&renderer);
        #[cfg(feature = "yakui")]
        let game_ui = GameUi::new(&renderer);
        let view = V::init(&renderer);
        let sim = self.sim.take().expect("simulation already taken");
        let clock = self.clock.take().expect("clock already taken");
        let mut cmds = self.cmds.take().expect("command queue already taken");
        cmds.set_current_tick(clock.tick());
        self.state = Some(RunState {
            renderer,
            view,
            sim,
            clock,
            cmds,
            last_redraw: Instant::now(),
            cursor_px: None,
            timings: FrameTimings::default(),
            stats: FrameStats::default(),
            pending_screenshot: false,
            #[cfg(feature = "egui")]
            debug_ui,
            #[cfg(feature = "yakui")]
            game_ui,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        // Engine-owned F12 → screenshot. Consumed before either overlay or
        // View::input sees it so a user holding F12 in a text field doesn't
        // accidentally suppress capture, and no example has to opt in.
        if let WindowEvent::KeyboardInput { event: key, .. } = &event
            && key.state == ElementState::Pressed
            && key.physical_key == PhysicalKey::Code(KeyCode::F12)
            && !key.repeat
        {
            state.pending_screenshot = true;
            return;
        }
        #[cfg(feature = "egui")]
        let egui_consumed = state
            .debug_ui
            .on_window_event(&state.renderer.window, &event)
            .consumed;
        #[cfg(not(feature = "egui"))]
        let egui_consumed = false;
        // yakui sees every event regardless of egui consumption so its internal
        // hover/layout state stays consistent across resizes etc; only the
        // application-level dispatch to View::input is suppressed when either
        // overlay claims the event.
        #[cfg(feature = "yakui")]
        let yakui_consumed = state.game_ui.on_window_event(&event);
        #[cfg(not(feature = "yakui"))]
        let yakui_consumed = false;
        if !egui_consumed && !yakui_consumed {
            let mut ctx = EngineCtx {
                event_loop,
                clock: &mut state.clock,
                timings: state.timings,
                stats: state.stats,
            };
            state
                .view
                .input(&state.sim, &mut ctx, &mut state.cmds, &event);
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.renderer.resize(size.width, size.height);
                state.renderer.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Cursor positions are in physical pixels matching the
                // swapchain extent; clamp into-bounds and drop sub-pixel
                // precision so it can be passed to copy_texture_to_buffer.
                let (w, h) = state.renderer.surface_size();
                let x = (position.x as i64).clamp(0, w.saturating_sub(1) as i64) as u32;
                let y = (position.y as i64).clamp(0, h.saturating_sub(1) as i64) as u32;
                state.cursor_px = Some((x, y));
            }
            WindowEvent::CursorLeft { .. } => {
                state.cursor_px = None;
                state.renderer.clear_hit_id_hover();
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let wall_dt = now - state.last_redraw;
                state.last_redraw = now;

                let pending = state.clock.advance(wall_dt);
                let tick_period = state.clock.tick_period();
                // After `advance`, the clock counts the ticks the engine
                // *should* have fired; the sim hasn't actually ticked yet,
                // so we replay them here. Per tick: bump the queue's
                // current tick, drain commands that ready'd at or before
                // it, then `sim.tick`. The drain seam is the single point
                // where external mutations land on `Sim`, mirroring
                // `Simulation::apply_command`'s contract.
                let total_after_advance = state.clock.tick();
                let sim_start = Instant::now();
                for i in 0..pending {
                    let sim_tick = total_after_advance - pending as u64 + (i + 1) as u64;
                    let sim = &mut state.sim;
                    state.cmds.set_current_tick(sim_tick);
                    state
                        .cmds
                        .drain_ready(sim_tick, |cmd| sim.apply_command(&cmd));
                    sim.tick(tick_period);
                }
                state.timings.sim_cpu = if pending > 0 {
                    sim_start.elapsed()
                } else {
                    Duration::ZERO
                };
                // Keep the queue's notion of "now" up to date even on frames
                // that fired zero ticks, so `push_now` during input events
                // between frames stamps with the latest known tick.
                state.cmds.set_current_tick(state.clock.tick());
                // View-side per-frame update, driven by wall-clock dt rather
                // than sim time so view animation (camera rigs, UI tweens, …)
                // keeps moving when the sim is paused.
                {
                    let mut ctx = EngineCtx {
                        event_loop,
                        clock: &mut state.clock,
                        timings: state.timings,
                        stats: state.stats,
                    };
                    state
                        .view
                        .update(&state.sim, &mut ctx, &mut state.cmds, wall_dt);
                }
                let alpha = state.clock.alpha();
                render_frame::<V>(state, event_loop, alpha);
                state.renderer.window.request_redraw();
            }
            _ => {}
        }
    }
}

/// Per-frame mutable resources passed between phases of [`render_frame`].
///
/// `Frame` exists for the lifetime of a single redraw: [`begin_frame`]
/// constructs it from a freshly-acquired surface texture and a new command
/// encoder, the main and overlay phases record draws into it, and
/// [`end_frame`] consumes it via `submit` + `present`. Splitting the per-frame
/// state out of `RunState` keeps the phase functions composable — each takes
/// `&mut Frame` plus only the long-lived state it actually needs.
struct Frame {
    surface_texture: wgpu::SurfaceTexture,
    view_tex: wgpu::TextureView,
    encoder: wgpu::CommandEncoder,
    /// Command buffers produced by overlay phases that must be submitted
    /// *before* the frame's main encoder (egui's texture-upload staging).
    pre_submit: Vec<wgpu::CommandBuffer>,
}

fn render_frame<V: View>(state: &mut RunState<V>, event_loop: &ActiveEventLoop, alpha: f32) {
    // Drain any hit-ID readbacks whose GPU writes finished since the last
    // frame. `Poll` is non-blocking — callbacks for not-yet-done buffers
    // simply don't fire this tick. The GPU profiler's map_async callbacks
    // ride the same poll — pull the latest result *after* polling so this
    // frame's overlay sees the freshest available measurement.
    let _ = state.renderer.device.poll(wgpu::PollType::Poll);
    state.timings.gpu = state.renderer.gpu_profiler_latest();
    // Per-frame indirection table is built up fresh inside main_pass via
    // engine-side renderers calling Renderer::reserve_terrain_chunk.
    state.renderer.reset_frame_id_table();

    let Some(mut frame) = begin_frame(&mut state.renderer) else {
        return;
    };
    extract_scene::<V>(&state.view, &state.sim, &state.renderer);
    if V::CONFIG.shadow_map_resolution.is_some() {
        for cascade in 0..4u32 {
            shadow_pass::<V>(
                &mut state.view,
                &state.sim,
                alpha,
                cascade,
                &state.renderer,
                &mut frame,
            );
        }
    }
    let render_start = Instant::now();
    main_pass::<V>(
        &mut state.view,
        &state.sim,
        alpha,
        &state.renderer,
        &mut frame,
    );
    state.timings.render_cpu = render_start.elapsed();
    // Record a 1×1 cursor-pixel copy into the next free readback slot.
    // No-ops if the cursor is outside the window or every slot is in
    // flight; the TerrainPicker's ray-plane path covers the gap. The
    // matching `schedule_readback` runs after `queue.submit` in
    // `end_frame` because wgpu requires `map_async` to be called *after*
    // the submission that writes the staging buffer.
    if let Some((cx, cy)) = state.cursor_px {
        state.renderer.id_readback().enqueue_copy(
            &mut frame.encoder,
            state.renderer.id_texture(),
            cx,
            cy,
        );
    }
    // Start the overlay-timing bracket *after* the world pass and picking
    // copy so it only covers yakui + egui. No-op when the device doesn't
    // have TIMESTAMP_QUERY_INSIDE_ENCODERS.
    state
        .renderer
        .gpu_profiler_overlay_begin(&mut frame.encoder);
    // Build the per-frame EngineCtx once and share &mut across both overlay
    // dispatch sites — keeps the construction in one place so new fields on
    // EngineCtx don't have to be threaded through each callback by hand.
    #[cfg(any(feature = "egui", feature = "yakui"))]
    let mut ctx = EngineCtx {
        event_loop,
        clock: &mut state.clock,
        timings: state.timings,
        stats: state.stats,
    };
    #[cfg(not(any(feature = "egui", feature = "yakui")))]
    let _ = event_loop;
    #[cfg(feature = "yakui")]
    yakui_overlay::<V>(
        &mut state.view,
        &state.sim,
        &mut ctx,
        &mut state.cmds,
        &state.renderer,
        &mut state.game_ui,
        &mut frame,
    );
    #[cfg(feature = "egui")]
    egui_overlay::<V>(
        &mut state.view,
        &state.sim,
        &mut ctx,
        &mut state.cmds,
        &state.renderer,
        &mut state.debug_ui,
        &mut frame,
    );
    // Both overlays have recorded their work — close the overlay-timing
    // bracket and then resolve all timestamps for the frame in one shot.
    // Doing the resolve here (rather than right after main_pass) means the
    // overlay timestamps land in the same readback buffer as the world ones.
    state.renderer.gpu_profiler_overlay_end(&mut frame.encoder);
    state.renderer.gpu_profiler_resolve(&mut frame.encoder);
    // Screenshot copy must be recorded last so it captures the final
    // composited image — world pass plus both overlays. Built here rather
    // than inside `end_frame` so the post-submit blocking save sits in
    // this function with the rest of the per-frame engine bookkeeping.
    let screenshot = if state.pending_screenshot {
        state.pending_screenshot = false;
        Some(ScreenshotRequest::record(
            &state.renderer.device,
            &mut frame.encoder,
            &frame.surface_texture.texture,
            state.renderer.surface_format(),
        ))
    } else {
        None
    };
    end_frame(&state.renderer, frame);
    if let Some(request) = screenshot {
        let dir = PathBuf::from("screenshots");
        match request.save_blocking(&state.renderer.device, &dir) {
            Ok(path) => println!("screenshot: {}", path.display()),
            Err(err) => eprintln!("screenshot failed: {err}"),
        }
    }
    // After submit, register the map_async on the readback slot we
    // copied into. Captures a snapshot of this frame's ID table so the
    // eventual callback resolves the sampled u32 through *this* frame's
    // mapping even though it fires 1–3 frames later.
    let table_snapshot = state.renderer.snapshot_frame_id_table();
    state
        .renderer
        .id_readback()
        .schedule_readback(table_snapshot);
    // Same shape for the GPU profiler — schedule the readback for the
    // timestamp pair we resolved earlier; a future `device.poll` fires
    // the callback that writes into `latest_duration`.
    state.renderer.gpu_profiler_schedule_readback();
    // Snapshot the per-frame CPU counters into `state.stats` so the next
    // frame's callbacks see them through `ctx.stats`. `take_and_reset`
    // zeroes the per-draw fields and keeps proxy counts at their last
    // value, so a view that records proxies only on tick boundaries still
    // reports the latest snapshot every frame.
    state.stats = state.renderer.take_frame_stats();
}

/// Phase 1: acquire the swapchain image and create the frame encoder.
///
/// Returns `None` on recoverable surface errors — the caller should skip the
/// frame and try again next redraw. Outdated / Lost trigger a resize so the
/// next acquire sees a fresh configuration.
fn begin_frame(renderer: &mut Renderer) -> Option<Frame> {
    let surface_texture = match renderer.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
            let size = renderer.window.inner_size();
            renderer.resize(size.width, size.height);
            return None;
        }
        wgpu::CurrentSurfaceTexture::Timeout
        | wgpu::CurrentSurfaceTexture::Occluded
        | wgpu::CurrentSurfaceTexture::Validation => return None,
    };

    let view_tex = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("currawong frame encoder"),
        });

    Some(Frame {
        surface_texture,
        view_tex,
        encoder,
        pre_submit: Vec::new(),
    })
}

/// Phase 2: extract per-frame scene state from sim → engine-managed uniforms.
///
/// Currently just the directional-light environment for the active zone.
/// Shadow/IBL probe extraction would land here when added.
///
/// When `active_zone` returns `None` (typical for UI/2D views) the neutral
/// environment is written instead of leaving the previous frame's values in
/// the buffer — otherwise any pipeline declaring `scene_layout` would sample
/// stale (or zero-initialised on the first frame) lighting.
fn extract_scene<V: View>(view: &V, sim: &V::Sim, renderer: &Renderer) {
    let env = match view.active_zone(sim) {
        Some(zone) => view.extract_environment(sim, zone),
        None => ViewEnvironment::neutral(),
    };
    renderer.write_scene(&env);
}

/// Phase 2.5: directional-light shadow cascade pass. Engine-driven; runs
/// `ViewConfig::shadow_map_resolution` is `Some` and `View::shadow_pass`
/// records depth-only occluder draws into the cascade's array layer. The
/// per-cascade light view-projection is pre-bound at `@group(0)` so the
/// View just needs to set its depth-only pipeline + vertex/instance buffers
/// and call `draw_indexed`.
fn shadow_pass<V: View>(
    view: &mut V,
    sim: &V::Sim,
    alpha: f32,
    cascade: u32,
    renderer: &Renderer,
    frame: &mut Frame,
) {
    let mut pass = frame
        .encoder
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("currawong shadow cascade pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: renderer.shadow_layer_view(cascade),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
    pass.set_bind_group(0, renderer.shadow_cascade_bind_group(cascade), &[]);
    view.shadow_pass(sim, alpha, cascade, renderer, &mut pass);
}

/// Phase 3: main world pass — clear, optional depth attach, `View::render`.
fn main_pass<V: View>(
    view: &mut V,
    sim: &V::Sim,
    alpha: f32,
    renderer: &Renderer,
    frame: &mut Frame,
) {
    let depth_attachment =
        renderer
            .depth_view()
            .map(|depth_view| wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            });
    let mut pass = frame
        .encoder
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("currawong frame pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &frame.view_tex,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(V::CONFIG.clear_colour),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                // Hit-ID attachment (#56). Cleared to 0 (no-hit sentinel);
                // PR 1 stores the result but has no consumers — readback and
                // per-pipeline opt-in land in later PRs. Pipelines drawn in
                // this pass either write `R32Uint` here or declare
                // `targets[1] = None` to leave existing IDs untouched.
                Some(wgpu::RenderPassColorAttachment {
                    view: renderer.id_view(),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: depth_attachment,
            // World-segment timestamps bracket only this pass. `None` when
            // the adapter lacks TIMESTAMP_QUERY. The overlay segment uses
            // encoder-level timestamps after this pass closes.
            timestamp_writes: renderer.gpu_profiler_world_pass_timestamps(),
            ..Default::default()
        });
    view.render(sim, alpha, renderer, &mut pass);
}

/// Phase 4a: yakui (game UI) overlay. Paints between the world and any
/// debug overlay.
#[cfg(feature = "yakui")]
fn yakui_overlay<V: View>(
    view: &mut V,
    sim: &V::Sim,
    ctx: &mut EngineCtx,
    cmds: &mut CommandQueue<<V::Sim as Simulation>::Command>,
    renderer: &Renderer,
    game_ui: &mut GameUi,
    frame: &mut Frame,
) {
    game_ui.run_and_render(renderer, &mut frame.encoder, &frame.view_tex, |yakui_ctx| {
        view.game_ui(sim, ctx, cmds, yakui_ctx);
    });
}

/// Phase 4b: egui (debug overlay). Sits visually on top of everything.
///
/// Returns staging command buffers (texture uploads) via [`Frame::pre_submit`]
/// — they must be submitted before the frame's main encoder.
#[cfg(feature = "egui")]
fn egui_overlay<V: View>(
    view: &mut V,
    sim: &V::Sim,
    ctx: &mut EngineCtx,
    cmds: &mut CommandQueue<<V::Sim as Simulation>::Command>,
    renderer: &Renderer,
    debug_ui: &mut DebugUi,
    frame: &mut Frame,
) {
    let staging =
        debug_ui.run_and_render(renderer, &mut frame.encoder, &frame.view_tex, |egui_ctx| {
            view.ui(sim, ctx, cmds, egui_ctx);
        });
    frame.pre_submit.extend(staging);
}

/// Phase 5: submit recorded work and present the swapchain image.
fn end_frame(renderer: &Renderer, frame: Frame) {
    let Frame {
        surface_texture,
        encoder,
        pre_submit,
        ..
    } = frame;
    renderer.queue.submit(
        pre_submit
            .into_iter()
            .chain(std::iter::once(encoder.finish())),
    );
    surface_texture.present();
}
