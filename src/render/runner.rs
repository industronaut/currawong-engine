//! Event-loop integration: `run` and `run_with_clock` bind a [`View`] to a
//! `winit` window and a [`SimClock`].

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::sim::{SimClock, Simulation};

#[cfg(feature = "egui")]
use super::debug_ui::DebugUi;
#[cfg(feature = "yakui")]
use super::game_ui::GameUi;
use super::renderer::Renderer;
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
    };
    event_loop
        .run_app(&mut handler)
        .expect("event loop terminated unexpectedly");
}

struct Handler<V: View> {
    sim: Option<V::Sim>,
    clock: Option<SimClock>,
    state: Option<RunState<V>>,
}

struct RunState<V: View> {
    renderer: Renderer,
    view: V,
    sim: V::Sim,
    clock: SimClock,
    last_redraw: Instant,
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
        let renderer = pollster::block_on(Renderer::new(window, V::CONFIG.depth_format));
        #[cfg(feature = "egui")]
        let debug_ui = DebugUi::new(&renderer);
        #[cfg(feature = "yakui")]
        let game_ui = GameUi::new(&renderer);
        let view = V::init(&renderer);
        let sim = self.sim.take().expect("simulation already taken");
        let clock = self.clock.take().expect("clock already taken");
        self.state = Some(RunState {
            renderer,
            view,
            sim,
            clock,
            last_redraw: Instant::now(),
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
            };
            state.view.input(&mut state.sim, &mut ctx, &event);
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.renderer.resize(size.width, size.height);
                state.renderer.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let wall_dt = now - state.last_redraw;
                state.last_redraw = now;

                let ticks = state.clock.advance(wall_dt);
                let tick_period = state.clock.tick_period();
                for _ in 0..ticks {
                    state.sim.tick(tick_period);
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
    let Some(mut frame) = begin_frame(&mut state.renderer) else {
        return;
    };
    extract_scene::<V>(&state.view, &state.sim, &state.renderer);
    main_pass::<V>(
        &mut state.view,
        &state.sim,
        alpha,
        &state.renderer,
        &mut frame,
    );
    #[cfg(feature = "yakui")]
    yakui_overlay::<V>(state, event_loop, &mut frame);
    #[cfg(feature = "egui")]
    egui_overlay::<V>(state, event_loop, &mut frame);
    #[cfg(not(any(feature = "egui", feature = "yakui")))]
    let _ = event_loop;
    end_frame(&state.renderer, frame);
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
fn extract_scene<V: View>(view: &V, sim: &V::Sim, renderer: &Renderer) {
    if let Some(zone) = view.active_zone(sim) {
        let env = view.extract_environment(sim, zone);
        renderer.write_scene(&env);
    }
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
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.view_tex,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(V::CONFIG.clear_colour),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth_attachment,
            ..Default::default()
        });
    view.render(sim, alpha, renderer, &mut pass);
}

/// Phase 4a: yakui (game UI) overlay. Paints between the world and any
/// debug overlay.
#[cfg(feature = "yakui")]
fn yakui_overlay<V: View>(
    state: &mut RunState<V>,
    event_loop: &ActiveEventLoop,
    frame: &mut Frame,
) {
    state
        .game_ui
        .run_and_render(&state.renderer, &mut frame.encoder, &frame.view_tex, || {
            let mut ctx = EngineCtx {
                event_loop,
                clock: &mut state.clock,
            };
            state.view.game_ui(&mut state.sim, &mut ctx);
        });
}

/// Phase 4b: egui (debug overlay). Sits visually on top of everything.
///
/// Returns staging command buffers (texture uploads) via [`Frame::pre_submit`]
/// — they must be submitted before the frame's main encoder.
#[cfg(feature = "egui")]
fn egui_overlay<V: View>(state: &mut RunState<V>, event_loop: &ActiveEventLoop, frame: &mut Frame) {
    let staging = state.debug_ui.run_and_render(
        &state.renderer,
        &mut frame.encoder,
        &frame.view_tex,
        |egui_ctx| {
            let mut ctx = EngineCtx {
                event_loop,
                clock: &mut state.clock,
            };
            state.view.ui(&mut state.sim, &mut ctx, egui_ctx);
        },
    );
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
