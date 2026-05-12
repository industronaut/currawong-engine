//! Window + GPU surface, device, queue, and engine-managed scene resources.

use std::sync::Arc;

use winit::window::Window;

use super::environment::ViewEnvironment;
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
        self.scene.depth_format()
    }

    /// Depth texture view, if a depth attachment was requested. Used by the
    /// engine's frame loop to attach to the render pass — not normally
    /// needed by views.
    pub(super) fn depth_view(&self) -> Option<&wgpu::TextureView> {
        self.scene.depth_view()
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

        Self {
            window,
            device,
            queue,
            surface,
            config,
            scene,
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
