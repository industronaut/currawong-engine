//! Engine-managed per-scene GPU resources: depth attachment, hit-ID
//! attachment, scene environment binding (and, in the future, shadow maps,
//! IBL probes, MSAA resolve targets, post-FX intermediates, …).
//!
//! Split out from [`Renderer`](super::Renderer) so that adding the next
//! engine-managed resource only touches one struct's `new` / `resize` /
//! field-visibility decisions, instead of growing the renderer indefinitely.

use super::environment::{SceneEnvironmentBinding, ViewEnvironment};

/// Format of the per-pixel hit-ID attachment. 32-bit unsigned integer; `0` is
/// reserved as the no-hit sentinel by the clear value in the opaque pass.
/// See issue #56 for the full HitProxy design.
pub(super) const ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// Engine-owned per-scene resources. Lives on [`Renderer`](super::Renderer);
/// pipelines reach in through the renderer's public accessors
/// ([`scene_layout`](super::Renderer::scene_layout),
/// [`scene_bind_group`](super::Renderer::scene_bind_group),
/// [`depth_format`](super::Renderer::depth_format),
/// [`id_format`](super::Renderer::id_format)).
pub(super) struct SceneResources {
    depth: Option<DepthAttachment>,
    id: IdAttachment,
    environment: SceneEnvironmentBinding,
}

struct DepthAttachment {
    format: wgpu::TextureFormat,
    view: wgpu::TextureView,
}

/// Always-allocated hit-ID attachment for the opaque pass. PR 1 of #56 wires
/// the attachment + clear; later PRs add readback and per-pipeline opt-in.
struct IdAttachment {
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
        let id = IdAttachment {
            view: create_id_view(device, width, height),
        };
        let environment = SceneEnvironmentBinding::new(device);
        Self {
            depth,
            id,
            environment,
        }
    }

    pub(super) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if let Some(depth) = self.depth.as_mut() {
            depth.view = create_depth_view(device, width, height, depth.format);
        }
        self.id.view = create_id_view(device, width, height);
    }

    pub(super) fn depth_format(&self) -> Option<wgpu::TextureFormat> {
        self.depth.as_ref().map(|d| d.format)
    }

    pub(super) fn depth_view(&self) -> Option<&wgpu::TextureView> {
        self.depth.as_ref().map(|d| &d.view)
    }

    pub(super) fn id_format(&self) -> wgpu::TextureFormat {
        ID_FORMAT
    }

    pub(super) fn id_view(&self) -> &wgpu::TextureView {
        &self.id.view
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

fn create_id_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("currawong hit-id"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ID_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
