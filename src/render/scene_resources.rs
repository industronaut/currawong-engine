//! Engine-managed per-scene GPU resources: depth attachment, hit-ID
//! attachment + readback machinery, scene environment binding (and, in the
//! future, shadow maps, IBL probes, MSAA resolve targets, post-FX
//! intermediates, …).
//!
//! Split out from [`Renderer`](super::Renderer) so that adding the next
//! engine-managed resource only touches one struct's `new` / `resize` /
//! field-visibility decisions, instead of growing the renderer indefinitely.

use std::sync::Mutex;

use crate::sim::{ChunkCoord, WorldObjectId, ZoneId};

use super::environment::{SceneEnvironmentBinding, ViewEnvironment};
use super::picking_buffer::{FrameIdTable, HitTarget, IdReadback};

/// Format of the per-pixel hit-ID attachment. 32-bit unsigned integer; `0`
/// is reserved as the no-hit sentinel by the clear value in the opaque pass.
/// See issue #56 for the full HitProxy design.
pub(super) const ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// Format of the directional-light shadow map.
pub(super) const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Fixed cascaded-shadow-map layer count. Matches the camera-side fit (see
/// [`Camera::fit_shadow_cascades`](super::Camera::fit_shadow_cascades)).
pub(super) const SHADOW_CASCADE_COUNT: u32 = 4;

/// Engine-owned per-scene resources. Lives on [`Renderer`](super::Renderer);
/// pipelines reach in through the renderer's public accessors
/// ([`scene_layout`](super::Renderer::scene_layout),
/// [`scene_bind_group`](super::Renderer::scene_bind_group),
/// [`depth_format`](super::Renderer::depth_format),
/// [`id_format`](super::Renderer::id_format)).
pub(super) struct SceneResources {
    depth: Option<DepthAttachment>,
    id: IdAttachment,
    /// Per-frame indirection from rendered hit IDs back to what they stand
    /// for. The runner clears this at the start of every frame; engine
    /// renderers (terrain, future mesh-object pass) call
    /// [`SceneResources::reserve_terrain_chunk`] etc. to register IDs as
    /// they record draws; a snapshot is captured into the readback slot at
    /// submission time so the eventual map-async callback resolves the
    /// sampled u32 through the *right frame's* table.
    frame_id_table: Mutex<FrameIdTable>,
    id_readback: IdReadback,
    /// Bind-group layout for the per-draw "ID base" uniform that
    /// participants in picking declare alongside their normal bind groups.
    /// Holds a single `u32` (in a 16-byte uniform buffer for std140
    /// alignment). Terrain consumes it per chunk in PR 2; future
    /// per-chunk or per-batch writers can reuse the same layout.
    id_base_layout: wgpu::BindGroupLayout,
    environment: SceneEnvironmentBinding,
    shadow: ShadowResources,
}

/// Directional-light shadow map: a 4-layer `Depth32Float` array. When the
/// View opts in via [`ViewConfig::shadow_map_resolution`](super::ViewConfig)
/// the texture is allocated at the requested resolution and each layer is
/// rendered to per frame; otherwise the engine still allocates a 1×1×4
/// placeholder so the scene bind group is always complete and material
/// pipelines don't have to branch on shadows-on/off. Sampling is gated
/// shader-side by the [`SunCascades::disabled`](super::environment::SunCascades::disabled)
/// sentinel (`splits[0] == 0.0`).
pub(super) struct ShadowResources {
    /// `Some` when the View opted in; `None` is the placeholder path.
    resolution: Option<u32>,
    array_view: wgpu::TextureView,
    /// Per-layer views, one per cascade, used as the depth attachment when
    /// recording each cascade's shadow pass.
    layer_views: [wgpu::TextureView; SHADOW_CASCADE_COUNT as usize],
    sampler: wgpu::Sampler,
    /// One uniform buffer per cascade, each holding a single `mat4x4<f32>`
    /// (the cascade's light view-projection). The runner writes these from
    /// the per-frame [`ViewEnvironment::sun_cascades`]; depth-only pipelines
    /// in the shadow pass bind the matching cascade's bind group at group 0.
    cascade_uniforms: [wgpu::Buffer; SHADOW_CASCADE_COUNT as usize],
    cascade_bgl: wgpu::BindGroupLayout,
    cascade_bind_groups: [wgpu::BindGroup; SHADOW_CASCADE_COUNT as usize],
}

struct DepthAttachment {
    format: wgpu::TextureFormat,
    view: wgpu::TextureView,
}

/// Always-allocated hit-ID attachment for the opaque pass. PR 1 wired the
/// clear; PR 2 wires the readback + first writer.
struct IdAttachment {
    /// Texture handle retained so `copy_texture_to_buffer` can target it
    /// each frame for cursor-pixel readback.
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl SceneResources {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        depth_format: Option<wgpu::TextureFormat>,
        shadow_map_resolution: Option<u32>,
    ) -> Self {
        let depth = depth_format.map(|format| DepthAttachment {
            format,
            view: create_depth_view(device, width, height, format),
        });
        let id = create_id_attachment(device, width, height);
        let shadow = ShadowResources::new(device, queue, shadow_map_resolution);
        let environment = SceneEnvironmentBinding::new(device, &shadow.array_view, &shadow.sampler);
        let id_readback = IdReadback::new(device);
        let id_base_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("currawong id-base layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        Self {
            depth,
            id,
            frame_id_table: Mutex::new(FrameIdTable::new()),
            id_readback,
            id_base_layout,
            environment,
            shadow,
        }
    }

    pub(super) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if let Some(depth) = self.depth.as_mut() {
            depth.view = create_depth_view(device, width, height, depth.format);
        }
        self.id = create_id_attachment(device, width, height);
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

    pub(super) fn id_texture(&self) -> &wgpu::Texture {
        &self.id.texture
    }

    pub(super) fn scene_layout(&self) -> &wgpu::BindGroupLayout {
        self.environment.layout()
    }

    pub(super) fn scene_bind_group(&self) -> &wgpu::BindGroup {
        self.environment.bind_group()
    }

    pub(super) fn write_scene(&self, queue: &wgpu::Queue, env: &ViewEnvironment) {
        self.environment.write(queue, env);
        self.shadow
            .write_cascade_matrices(queue, &env.sun_cascades.matrices);
    }

    /// Reserve a chunk's worth of hit IDs in the current frame's table.
    /// Called by `TerrainRenderer::draw_solid` once per drawn chunk; the
    /// returned `base_id` is written into the chunk's per-frame uniform and
    /// added to the per-vertex chunk-local cell index in the shader.
    pub(super) fn reserve_terrain_chunk(&self, zone: ZoneId, chunk: ChunkCoord) -> u32 {
        self.frame_id_table
            .lock()
            .expect("frame_id_table poisoned")
            .reserve_terrain_chunk(zone, chunk)
    }

    /// Reserve a single hit ID for a mesh object in the current frame's
    /// table. Called by user code (or
    /// [`RenderObjectTraversal::for_each_alive_with_hit_id`](super::RenderObjectTraversal::for_each_alive_with_hit_id))
    /// once per drawn object; the returned ID is written into the object's
    /// per-instance attributes so every pixel it covers in the opaque pass
    /// carries that ID in the `R32Uint` attachment.
    pub(super) fn reserve_object(&self, zone: ZoneId, id: WorldObjectId) -> u32 {
        self.frame_id_table
            .lock()
            .expect("frame_id_table poisoned")
            .reserve_object(zone, id)
    }

    /// Clear the frame ID table — called once at the start of each frame.
    pub(super) fn reset_frame_id_table(&self) {
        self.frame_id_table
            .lock()
            .expect("frame_id_table poisoned")
            .clear();
    }

    /// Snapshot the current frame's ID table for handoff to a readback slot.
    pub(super) fn snapshot_frame_id_table(&self) -> FrameIdTable {
        self.frame_id_table
            .lock()
            .expect("frame_id_table poisoned")
            .clone()
    }

    pub(super) fn id_readback(&self) -> &IdReadback {
        &self.id_readback
    }

    pub(super) fn id_base_layout(&self) -> &wgpu::BindGroupLayout {
        &self.id_base_layout
    }

    /// Latest hit target delivered by the readback ring, if any.
    pub(super) fn hit_id_hover(&self) -> Option<HitTarget> {
        self.id_readback.latest()
    }

    pub(super) fn shadow_map_resolution(&self) -> Option<u32> {
        self.shadow.resolution
    }

    pub(super) fn shadow_layer_view(&self, cascade: u32) -> &wgpu::TextureView {
        &self.shadow.layer_views[cascade as usize]
    }

    pub(super) fn shadow_cascade_layout(&self) -> &wgpu::BindGroupLayout {
        &self.shadow.cascade_bgl
    }

    pub(super) fn shadow_cascade_bind_group(&self, cascade: u32) -> &wgpu::BindGroup {
        &self.shadow.cascade_bind_groups[cascade as usize]
    }
}

impl ShadowResources {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, resolution: Option<u32>) -> Self {
        let size = resolution.unwrap_or(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("currawong shadow map"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: SHADOW_CASCADE_COUNT,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("currawong shadow array view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layer_views = std::array::from_fn(|i| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("currawong shadow layer view"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("currawong shadow comparison sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let cascade_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("currawong shadow cascade bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let cascade_uniforms: [wgpu::Buffer; SHADOW_CASCADE_COUNT as usize] =
            std::array::from_fn(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("currawong shadow cascade uniform"),
                    size: 64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
        let cascade_bind_groups: [wgpu::BindGroup; SHADOW_CASCADE_COUNT as usize] =
            std::array::from_fn(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("currawong shadow cascade bind group"),
                    layout: &cascade_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: cascade_uniforms[i].as_entire_binding(),
                    }],
                })
            });
        // Initialise every layer to "max depth" so the texture is in a
        // known sample-ready state from the moment the scene bind group is
        // first used. Without this, wgpu defers an internal "first-use
        // clear" until the first draw that binds the texture for
        // sampling — on some backends (notably Metal) that lazy clear
        // ends up scheduled at a moment that interacts badly with a
        // simultaneous swapchain resize, producing a frame-loop hang.
        // The cost is one zero-load clear pass per cascade layer, once.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("currawong shadow init"),
        });
        for layer in &layer_views {
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("currawong shadow init clear"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        }
        queue.submit(std::iter::once(encoder.finish()));

        Self {
            resolution,
            array_view,
            layer_views,
            sampler,
            cascade_uniforms,
            cascade_bgl,
            cascade_bind_groups,
        }
    }

    fn write_cascade_matrices(&self, queue: &wgpu::Queue, matrices: &[glam::Mat4; 4]) {
        for (i, mat) in matrices.iter().enumerate() {
            let cols = mat.to_cols_array();
            queue.write_buffer(&self.cascade_uniforms[i], 0, bytemuck::cast_slice(&cols));
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

fn create_id_attachment(device: &wgpu::Device, width: u32, height: u32) -> IdAttachment {
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
        // COPY_SRC so the readback ring can copy a 1×1 region under the
        // cursor into a MAP_READ staging buffer each frame.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    IdAttachment { texture, view }
}
