//! Engine-managed per-scene GPU resources: depth attachment + scene
//! environment binding (and, in the future, shadow maps, IBL probes, MSAA
//! resolve targets, post-FX intermediates, …).
//!
//! Split out from [`Renderer`](super::Renderer) so that adding the next
//! engine-managed resource only touches one struct's `new` / `resize` /
//! field-visibility decisions, instead of growing the renderer indefinitely.

use super::environment::{SceneEnvironmentBinding, ViewEnvironment};

/// Engine-owned per-scene resources. Lives on [`Renderer`](super::Renderer);
/// pipelines reach in through the renderer's public accessors
/// ([`scene_layout`](super::Renderer::scene_layout),
/// [`scene_bind_group`](super::Renderer::scene_bind_group),
/// [`depth_format`](super::Renderer::depth_format)).
pub(super) struct SceneResources {
    depth: Option<DepthAttachment>,
    environment: SceneEnvironmentBinding,
}

struct DepthAttachment {
    format: wgpu::TextureFormat,
    view: wgpu::TextureView,
}

impl SceneResources {
    pub(super) fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let depth = depth_format.map(|format| DepthAttachment {
            format,
            view: create_depth_view(device, width, height, format),
        });
        let environment = SceneEnvironmentBinding::new(device);
        Self { depth, environment }
    }

    pub(super) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if let Some(depth) = self.depth.as_mut() {
            depth.view = create_depth_view(device, width, height, depth.format);
        }
    }

    pub(super) fn depth_format(&self) -> Option<wgpu::TextureFormat> {
        self.depth.as_ref().map(|d| d.format)
    }

    pub(super) fn depth_view(&self) -> Option<&wgpu::TextureView> {
        self.depth.as_ref().map(|d| &d.view)
    }

    pub(super) fn scene_layout(&self) -> &wgpu::BindGroupLayout {
        self.environment.layout()
    }

    pub(super) fn scene_bind_group(&self) -> &wgpu::BindGroup {
        self.environment.bind_group()
    }

    pub(super) fn write_scene(&self, queue: &wgpu::Queue, env: &ViewEnvironment) {
        self.environment.write(queue, env);
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
