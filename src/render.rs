//! View side of the engine.
//!
//! Reads a [`Simulation`] each frame and renders it through wgpu/winit.

use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::sim::{SimClock, Simulation};

/// Perspective camera helper for views that render in 3D world space.
///
/// Held by the user's [`View`] (a View doesn't have to use one — UI/2D
/// views can render directly in clip space). The user updates `position`,
/// `target`, and `aspect` over time; pipelines upload [`Camera::view_proj`]
/// to a uniform buffer each frame.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_radians: 60_f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera {
    /// Right-handed view matrix. Uses `target` as the look-at point.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// Right-handed perspective matrix with depth in `[0, 1]` (wgpu/Vulkan
    /// convention, not OpenGL's `[-1, 1]`).
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y_radians, self.aspect, self.near, self.far)
    }

    pub fn view_proj(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }
}

/// Owns the window and the wgpu device/queue/surface.
pub struct Renderer {
    pub window: Arc<Window>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth: Option<DepthAttachment>,
}

struct DepthAttachment {
    format: wgpu::TextureFormat,
    view: wgpu::TextureView,
}

impl Renderer {
    /// Format pipelines targeting the swapchain should declare.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Depth format the engine has allocated for this view, if any. Pipelines
    /// using depth must declare this format in their `DepthStencilState`.
    pub fn depth_format(&self) -> Option<wgpu::TextureFormat> {
        self.depth.as_ref().map(|d| d.format)
    }

    async fn new(window: Arc<Window>, depth_format: Option<wgpu::TextureFormat>) -> Self {
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

        let depth = depth_format.map(|format| DepthAttachment {
            format,
            view: create_depth_view(&device, config.width, config.height, format),
        });

        Self {
            window,
            device,
            queue,
            surface,
            config,
            depth,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        if let Some(depth) = self.depth.as_mut() {
            depth.view = create_depth_view(
                &self.device,
                self.config.width,
                self.config.height,
                depth.format,
            );
        }
    }
}

fn create_depth_view(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("currawong depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Mutable engine state passed to view callbacks. Use `event_loop` to
/// request exit and `clock` to read or change sim speed and tick rate.
pub struct EngineCtx<'a> {
    pub event_loop: &'a ActiveEventLoop,
    pub clock: &'a mut SimClock,
}

/// A view onto a [`Simulation`].
///
/// `render` receives `&Sim` (read-only), so the rendering path is structurally
/// prevented from mutating the simulation. `input` receives `&mut Sim` so
/// user-driven actions (clicks, key presses) can drive sim changes.
///
/// `init` runs once after the GPU is ready (build pipelines, load assets).
pub trait View: 'static {
    /// The kind of simulation this view reads from.
    type Sim: Simulation;

    fn init(renderer: &Renderer) -> Self;

    /// Render a frame.
    ///
    /// `alpha` is the interpolation factor in `[0, 1]` between the most recent
    /// completed tick and the next pending tick — useful for smooth animation
    /// when tick rate is below refresh rate. Views that don't interpolate can
    /// ignore it.
    fn render(
        &mut self,
        sim: &Self::Sim,
        alpha: f32,
        renderer: &Renderer,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        let _ = (sim, alpha, renderer, pass);
    }

    fn input(&mut self, sim: &mut Self::Sim, ctx: &mut EngineCtx, event: &WindowEvent) {
        let _ = (sim, ctx, event);
    }

    fn title() -> &'static str {
        "currawong"
    }

    fn clear_colour() -> wgpu::Color {
        wgpu::Color {
            r: 0.05,
            g: 0.07,
            b: 0.10,
            a: 1.0,
        }
    }

    /// Return `Some(format)` to have the engine allocate a depth texture and
    /// pre-attach it to the frame's render pass. Pipelines must declare the
    /// same format in their `DepthStencilState`. Default `None` is right for
    /// 2D / UI views that draw in clip space.
    fn depth_format() -> Option<wgpu::TextureFormat> {
        None
    }
}

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
                .depth
                .as_ref()
                .map(|d| wgpu::RenderPassDepthStencilAttachment {
                    view: &d.view,
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
