//! Currawong — a game engine in Rust.
//!
//! Implement [`App`] and call [`run`] to open a window with a configured wgpu
//! surface and a render-pass-per-frame loop.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

pub use wgpu;
pub use winit;

/// Owns the window and the wgpu device/queue/surface for an application.
pub struct Renderer {
    pub window: Arc<Window>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

impl Renderer {
    /// Format pipelines targeting the swapchain should declare.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    async fn new(window: Arc<Window>) -> Self {
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

        Self { window, device, queue, surface, config }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
    }
}

/// User application driven by the engine's event loop.
///
/// `init` runs once after the GPU is ready (build pipelines, load assets, etc.).
/// `render` runs each frame with a render pass that has already been started
/// with the configured clear colour — record draw calls into it.
pub trait App: Sized + 'static {
    fn init(renderer: &Renderer) -> Self;

    fn render(&mut self, renderer: &Renderer, pass: &mut wgpu::RenderPass<'_>) {
        let _ = (renderer, pass);
    }

    fn title() -> &'static str {
        "currawong"
    }

    fn clear_colour() -> wgpu::Color {
        wgpu::Color { r: 0.05, g: 0.07, b: 0.10, a: 1.0 }
    }
}

/// Run an application until its window is closed.
pub fn run<A: App>() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut handler = Handler::<A> { state: None };
    event_loop
        .run_app(&mut handler)
        .expect("event loop terminated unexpectedly");
}

struct Handler<A: App> {
    state: Option<(Renderer, A)>,
}

impl<A: App> ApplicationHandler for Handler<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title(A::title());
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );
        let renderer = pollster::block_on(Renderer::new(window));
        let app = A::init(&renderer);
        self.state = Some((renderer, app));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some((renderer, app)) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                renderer.resize(size.width, size.height);
                renderer.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                render_frame::<A>(renderer, app);
                renderer.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn render_frame<A: App>(renderer: &mut Renderer, app: &mut A) {
    let frame = match renderer.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
            let size = renderer.window.inner_size();
            renderer.resize(size.width, size.height);
            return;
        }
        wgpu::CurrentSurfaceTexture::Timeout
        | wgpu::CurrentSurfaceTexture::Occluded
        | wgpu::CurrentSurfaceTexture::Validation => return,
    };

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("currawong frame encoder"),
        });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("currawong frame pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(A::clear_colour()),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        app.render(renderer, &mut pass);
    }

    renderer.queue.submit(std::iter::once(encoder.finish()));
    frame.present();
}
