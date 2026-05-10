//! Event-loop integration: `run` and `run_with_clock` bind a [`View`] to a
//! `winit` window and a [`SimClock`].

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::sim::{SimClock, Simulation};

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
}

impl<V: View> ApplicationHandler for Handler<V> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title(V::title());
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );
        let renderer = pollster::block_on(Renderer::new(window, V::depth_format()));
        let view = V::init(&renderer);
        let sim = self.sim.take().expect("simulation already taken");
        let clock = self.clock.take().expect("clock already taken");
        self.state = Some(RunState {
            renderer,
            view,
            sim,
            clock,
            last_redraw: Instant::now(),
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
        {
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
                render_frame::<V>(state, alpha);
                state.renderer.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn render_frame<V: View>(state: &mut RunState<V>, alpha: f32) {
    let frame = match state.renderer.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
            let size = state.renderer.window.inner_size();
            state.renderer.resize(size.width, size.height);
            return;
        }
        wgpu::CurrentSurfaceTexture::Timeout
        | wgpu::CurrentSurfaceTexture::Occluded
        | wgpu::CurrentSurfaceTexture::Validation => return,
    };

    let view_tex = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder =
        state
            .renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("currawong frame encoder"),
            });

    {
        let depth_attachment =
            state
                .renderer
                .depth_view()
                .map(|view| wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("currawong frame pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view_tex,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(V::clear_colour()),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth_attachment,
            ..Default::default()
        });
        state
            .view
            .render(&state.sim, alpha, &state.renderer, &mut pass);
    }

    state
        .renderer
        .queue
        .submit(std::iter::once(encoder.finish()));
    frame.present();
}
