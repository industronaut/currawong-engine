//! Window + GPU surface, device, queue, and (optional) depth attachment.

use std::sync::Arc;

use winit::window::Window;

use super::environment::SceneEnvironmentBinding;

/// Owns the window and the wgpu device/queue/surface.
pub struct Renderer {
    pub window: Arc<Window>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub(super) surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth: Option<DepthAttachment>,
    scene: SceneEnvironmentBinding,
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

    /// Current swapchain size in pixels. Useful for screen-space rendering
    /// (e.g. egui's `ScreenDescriptor`).
    pub fn surface_size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Depth format the engine has allocated for this view, if any. Pipelines
    /// using depth must declare this format in their `DepthStencilState`.
    pub fn depth_format(&self) -> Option<wgpu::TextureFormat> {
        self.depth.as_ref().map(|d| d.format)
    }

    /// Depth texture view, if a depth attachment was requested. Used by the
    /// engine's frame loop to attach to the render pass — not normally
    /// needed by views.
    pub(super) fn depth_view(&self) -> Option<&wgpu::TextureView> {
        self.depth.as_ref().map(|d| &d.view)
    }

    /// Bind-group layout for the engine-managed scene environment uniform.
    /// Pass to [`PipelineLayoutDescriptor`](wgpu::PipelineLayoutDescriptor)
    /// when building any pipeline that reads scene lighting (sun direction,
    /// ambient, etc.). The corresponding bind group is bound by the engine
    /// before [`View::render`](crate::View::render) runs — your shader just
    /// needs to declare a `@group(N) @binding(0)` for it.
    pub fn scene_layout(&self) -> &wgpu::BindGroupLayout {
        self.scene.layout()
    }

    /// Bind group for the engine-managed scene environment uniform. Bind at
    /// the slot your pipeline reserved for [`scene_layout`](Self::scene_layout).
    /// The engine writes fresh values into it each frame from
    /// [`View::extract_environment`](crate::View::extract_environment).
    pub fn scene_bind_group(&self) -> &wgpu::BindGroup {
        self.scene.bind_group()
    }

    pub(super) fn write_scene(&self, env: &super::environment::ViewEnvironment) {
        self.scene.write(&self.queue, env);
    }

    /// Allocate (or release) the engine-managed depth attachment. Called by
    /// the runner once after [`View::init`](crate::View::init) returns the
    /// view config. Passing `None` releases any existing depth texture;
    /// passing `Some(format)` creates a depth texture sized to the current
    /// swapchain. The format then participates in [`resize`](Self::resize).
    pub(super) fn configure_depth(&mut self, depth_format: Option<wgpu::TextureFormat>) {
        self.depth = depth_format.map(|format| DepthAttachment {
            format,
            view: create_depth_view(&self.device, self.config.width, self.config.height, format),
        });
    }

    pub(super) async fn new(window: Arc<Window>) -> Self {
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

        let scene = SceneEnvironmentBinding::new(&device);

        Self {
            window,
            device,
            queue,
            surface,
            config,
            depth: None,
            scene,
        }
    }

    pub(super) fn resize(&mut self, width: u32, height: u32) {
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
