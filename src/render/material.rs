//! Render materials — three-tier (template / instance / per-instance attribs).
//!
//! ## Tiers
//!
//! - **Material template** — a concrete Rust struct per material kind, owning
//!   the compiled pipeline + bind-group layout for the per-material bind
//!   group. Today: [`UnlitColoredMaterial`] (this file) and
//!   [`PbrMaterial`](super::PbrMaterial). Both implement the [`MeshMaterial`]
//!   trait — the shared structural shape.
//! - **Material instance** — a bind group + uniform buffer bound to a
//!   template, holding concrete uniform values. Many sim objects share one
//!   instance ("red metal," "gold trim"). Stored in a
//!   [`MaterialInstanceRegistry`].
//! - **Per-instance attribs** — `repr(C)` `Pod` struct packed into a vertex
//!   buffer with `step_mode = Instance`; varies per drawn copy. Both mesh
//!   materials read the same [`MeshInstanceAttribs`] layout (84 B: model
//!   matrix + tint + GPU hit ID).
//!
//! ## Wiring `UnlitColoredMaterial`
//!
//! ```text
//! init:   let camera = CameraBinding::new(&renderer.device);
//!         let material = UnlitColoredMaterial::new(&renderer, camera.layout());
//!         let red = material.create_instance(&renderer, Vec4::new(0.8, 0.2, 0.2, 1.0));
//! draw:   pass.set_pipeline(material.pipeline());
//!         pass.set_bind_group(0, camera.bind_group(), &[]);
//!         pass.set_bind_group(1, red.bind_group(), &[]);
//!         pass.set_vertex_buffer(0, mesh.vertices.slice(..));  // pos: vec3
//!         pass.set_vertex_buffer(1, instance_buf.slice(..));   // MeshInstanceAttribs
//!         pass.draw_indexed(..., 0..count);
//! ```

use std::collections::HashMap;
use std::hash::Hash;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};

use super::renderer::Renderer;
use super::vertex::PosNormalUv;

// --- Generic instance registry -------------------------------------------

/// Registry of material instances keyed by a user-chosen id type. Parallel
/// in shape to [`RenderRegistry`](super::RenderRegistry), but holding live
/// material instances rather than render-object templates.
///
/// Generic over `I` (the concrete instance type, e.g. [`UnlitColoredInstance`])
/// and `K` (the id type, `Copy + Eq + Hash`). One registry per material kind
/// — materials don't share an interface yet, so they don't share storage.
///
/// Re-registering an id silently replaces the existing instance; the old
/// instance's GPU resources are dropped.
pub struct MaterialInstanceRegistry<I, K>
where
    K: Copy + Eq + Hash,
{
    instances: HashMap<K, I>,
}

impl<I, K> MaterialInstanceRegistry<I, K>
where
    K: Copy + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            instances: HashMap::new(),
        }
    }

    pub fn register(&mut self, id: K, instance: I) {
        self.instances.insert(id, instance);
    }

    pub fn get(&self, id: K) -> Option<&I> {
        self.instances.get(&id)
    }

    /// Mutable variant of [`get`](Self::get). Needed for instances that
    /// expose a per-frame `refresh` method (e.g. `PbrMaterialInstance` —
    /// the bind group has to be rebuilt when its handle transitions).
    pub fn get_mut(&mut self, id: K) -> Option<&mut I> {
        self.instances.get_mut(&id)
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

impl<I, K> Default for MaterialInstanceRegistry<I, K>
where
    K: Copy + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

// --- MeshMaterial: shared trait + per-instance attribs --------------------

/// Per-instance attributes shared by every mesh material (today
/// [`UnlitColoredMaterial`] and [`PbrMaterial`](super::PbrMaterial)). Pack
/// instances of this struct into a `wgpu::Buffer` with `VERTEX` usage, bind
/// it as vertex buffer slot 1, and the material's pipeline reads it directly.
///
/// Layout (84 B total, alignment 4 — no tail padding):
/// - bytes 0..64 — model matrix as four column-major `vec4`s
/// - bytes 64..80 — per-instance tint (linear RGBA, multiplied with material
///   colour in the shader)
/// - bytes 80..84 — GPU hit ID (`0` = no-hit sentinel, matching the engine's
///   `R32Uint` attachment clear value)
///
/// New mesh materials should consume this layout — declaring their own would
/// fragment the per-instance pipeline plumbing without buying anything.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct MeshInstanceAttribs {
    /// `Mat4` as four column-major `vec4`s; layout matches
    /// [`mat4_instance_attributes`](super::mat4_instance_attributes).
    pub model: [[f32; 4]; 4],
    /// Linear RGBA in `[0, 1]`; multiplied with the material's base / albedo
    /// colour. Use `[1.0; 4]` for "no per-instance tint."
    pub tint: [f32; 4],
    /// GPU hit ID stamped into the engine's `R32Uint` attachment for every
    /// pixel this instance covers. `0` = no-hit sentinel. Opt in to picking
    /// via [`Self::with_hit_id`], passing an ID returned by
    /// [`Renderer::reserve_object`](super::Renderer::reserve_object).
    pub hit_id: u32,
}

impl MeshInstanceAttribs {
    pub fn new(model: Mat4, tint: Vec4) -> Self {
        Self {
            model: model.to_cols_array_2d(),
            tint: tint.to_array(),
            hit_id: 0,
        }
    }

    /// Builder: stamp this instance with a GPU hit ID returned by
    /// [`Renderer::reserve_object`](super::Renderer::reserve_object). Every
    /// pixel the instance covers in the opaque pass carries this ID in the
    /// hit-ID attachment, so a cursor over any of them resolves back to the
    /// originating `WorldObjectId`.
    pub fn with_hit_id(mut self, hit_id: u32) -> Self {
        self.hit_id = hit_id;
        self
    }

    /// Vertex attributes for a buffer of `MeshInstanceAttribs` consumed
    /// per-instance. `start_location` is the first `@location(N)` the vertex
    /// shader reserves; six attributes are claimed (four mat4 columns, tint,
    /// hit_id) at locations `start_location..start_location + 6`.
    ///
    /// Pair with a `wgpu::VertexBufferLayout` of stride
    /// `size_of::<MeshInstanceAttribs>()` and `step_mode = Instance`.
    pub const fn vertex_attributes(start_location: u32) -> [wgpu::VertexAttribute; 6] {
        [
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: start_location,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: start_location + 1,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 32,
                shader_location: start_location + 2,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 48,
                shader_location: start_location + 3,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 64,
                shader_location: start_location + 4,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 80,
                shader_location: start_location + 5,
                format: wgpu::VertexFormat::Uint32,
            },
        ]
    }
}

/// Build the render pipeline shared by every "PBR-shaped" mesh material —
/// today [`PbrMaterial`](super::PbrMaterial) and
/// [`PbrAtlasMaterial`](super::PbrAtlasMaterial). The two differ only in
/// their shader source and their per-instance bind-group layout; everything
/// else (pipeline layout, vertex buffers, fragment targets, depth-stencil)
/// is identical.
///
/// The pipeline this builds:
/// - Bind groups: `[camera, scene, instance]` — material instance lives at
///   `@group(2)`.
/// - Vertex buffer slot 0: [`PosNormalUv`] per vertex.
/// - Vertex buffer slot 1: [`MeshInstanceAttribs`] per instance, with
///   attributes at `@location(3..9)` — PosNormalUv consumes 0..3.
/// - Fragment targets: the surface (REPLACE blend) plus the engine's
///   `R32Uint` hit-ID attachment via [`Renderer::id_target_writer`].
/// - Depth-stencil: standard `Less` test with depth write enabled, gated
///   on [`Renderer::depth_format`] so the pipeline auto-omits depth state
///   when the view didn't allocate a depth attachment.
///
/// `label` is the human-readable base used for the pipeline-layout and
/// pipeline labels (`"<label> pipeline layout"` and `"<label> pipeline"`).
/// The caller still owns the shader module and the instance BGL because
/// both are material-specific — the helper just stitches them together.
pub(super) fn build_pbr_style_pipeline(
    renderer: &Renderer,
    label: &str,
    shader: &wgpu::ShaderModule,
    camera_layout: &wgpu::BindGroupLayout,
    instance_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let device = &renderer.device;

    let layout_label = format!("{label} pipeline layout");
    let pipeline_label = format!("{label} pipeline");

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&layout_label),
        bind_group_layouts: &[
            Some(camera_layout),
            Some(renderer.scene_layout()),
            Some(instance_bgl),
        ],
        ..Default::default()
    });

    let pos_normal_uv_attrs = PosNormalUv::attributes(0);
    let instance_attrs = MeshInstanceAttribs::vertex_attributes(3);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&pipeline_label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: PosNormalUv::STRIDE,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &pos_normal_uv_attrs,
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshInstanceAttribs>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &instance_attrs,
                },
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: renderer.surface_format(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                // Per-instance hit IDs to the engine's R32Uint attachment.
                // Instances that don't care leave `hit_id` at 0, matching
                // the attachment's clear value.
                renderer.id_target_writer(),
            ],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: renderer
            .depth_format()
            .map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Common shape of every mesh-material template — the structural pattern
/// behind [`UnlitColoredMaterial`] and [`PbrMaterial`](super::PbrMaterial).
///
/// The contract:
/// - The pipeline reads camera at `@group(0)` and the material-instance bind
///   group at the material's chosen index (per-kind; PBR adds the scene
///   environment in between).
/// - Vertex buffer slot 1 carries [`MeshInstanceAttribs`].
/// - The fragment shader writes per-instance `hit_id` to the engine's
///   `R32Uint` hit-ID attachment.
///
/// The trait is deliberately thin today (one accessor + the
/// [`Instance`](Self::Instance) associated type) — generic draw helpers will
/// land here when a call site actually needs them.
pub trait MeshMaterial {
    /// Concrete material-instance type produced by this template. Each
    /// material kind owns its own instance shape (uniform layout, sampler /
    /// texture bindings) — the trait just names it.
    type Instance;

    /// The compiled render pipeline. Bind via `pass.set_pipeline`.
    fn pipeline(&self) -> &wgpu::RenderPipeline;
}

// --- UnlitColored: the first concrete material ----------------------------

const UNLIT_COLORED_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct Material {
    base_color: vec4<f32>,
};
@group(1) @binding(0) var<uniform> material: Material;

struct VsIn {
    @location(0) pos: vec3<f32>,
    // Per-instance model matrix as four vec4 columns.
    @location(1) m0: vec4<f32>,
    @location(2) m1: vec4<f32>,
    @location(3) m2: vec4<f32>,
    @location(4) m3: vec4<f32>,
    // Per-instance tint, multiplied with the material's base_color.
    @location(5) tint: vec4<f32>,
    // Per-instance hit ID, written to the engine's R32Uint attachment by
    // the fragment shader. 0 = no-hit (matches the attachment's clear
    // value); non-zero comes from Renderer::reserve_object (#56 PR 3).
    @location(6) hit_id: u32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) tint: vec4<f32>,
    // Integer attributes can't be linearly interpolated; flat ships one
    // value per primitive.
    @location(1) @interpolate(flat) hit_id: u32,
};

struct FsOut {
    @location(0) color:  vec4<f32>,
    @location(1) hit_id: u32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let world = model * vec4<f32>(in.pos, 1.0);
    var out: VsOut;
    out.clip   = camera.view_proj * world;
    out.tint   = in.tint;
    out.hit_id = in.hit_id;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    var out: FsOut;
    out.color  = material.base_color * in.tint;
    out.hit_id = in.hit_id;
    return out;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UnlitColoredUniform {
    base_color: [f32; 4],
}

/// Template for the unlit-colored material: owns the compiled pipeline and
/// the bind-group layout for per-material-instance uniforms.
///
/// Build once at `View::init`; create per-colour instances via
/// [`Self::create_instance`]. Pipeline expects:
/// - `@group(0)` — camera uniform (use a
///   [`CameraBinding`](super::CameraBinding))
/// - `@group(1)` — material uniform (`base_color: vec4<f32>`)
/// - vertex buffer slot 0 — `position: vec3<f32>` per vertex
/// - vertex buffer slot 1 — [`MeshInstanceAttribs`] per instance
///
/// Auto-adapts to the View's depth choice via
/// [`Renderer::depth_format`]: includes depth-test state when the renderer
/// has depth, omits it otherwise (right for 2D / UI views).
pub struct UnlitColoredMaterial {
    pipeline: wgpu::RenderPipeline,
    instance_bgl: wgpu::BindGroupLayout,
}

impl UnlitColoredMaterial {
    pub fn new(renderer: &Renderer, camera_layout: &wgpu::BindGroupLayout) -> Self {
        let device = &renderer.device;

        let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("UnlitColored instance bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UnlitColored shader"),
            source: wgpu::ShaderSource::Wgsl(UNLIT_COLORED_SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UnlitColored pipeline layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&instance_bgl)],
            ..Default::default()
        });

        let instance_attrs = MeshInstanceAttribs::vertex_attributes(1);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UnlitColored pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    // Slot 0: per-vertex position.
                    wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        }],
                    },
                    // Slot 1: per-instance Mat4 + tint + hit_id.
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<MeshInstanceAttribs>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance_attrs,
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: renderer.surface_format(),
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Write per-instance hit IDs to the engine's R32Uint
                    // attachment (#56 PR 3). Instances that don't care
                    // about picking leave their `hit_id` at the 0 default,
                    // which matches the attachment's clear value —
                    // semantically identical to PR 1's opt-out shape.
                    renderer.id_target_writer(),
                ],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: renderer
                .depth_format()
                .map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            instance_bgl,
        }
    }

    /// The compiled render pipeline. Bind via `pass.set_pipeline` each draw.
    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    /// Create a material instance with the given `base_color` (linear RGBA
    /// in `[0, 1]`). The instance owns its own uniform buffer and bind group.
    pub fn create_instance(&self, renderer: &Renderer, base_color: Vec4) -> UnlitColoredInstance {
        let data = UnlitColoredUniform {
            base_color: base_color.to_array(),
        };
        let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UnlitColored instance uniform"),
            size: std::mem::size_of::<UnlitColoredUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        renderer
            .queue
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&data));
        let bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("UnlitColored instance bind group"),
                layout: &self.instance_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
        UnlitColoredInstance { buffer, bind_group }
    }
}

impl MeshMaterial for UnlitColoredMaterial {
    type Instance = UnlitColoredInstance;

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        self.pipeline()
    }
}

/// A live material instance for [`UnlitColoredMaterial`] — uniform buffer +
/// bind group. Bind as `@group(1)` when drawing through the material's
/// pipeline.
pub struct UnlitColoredInstance {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl UnlitColoredInstance {
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Re-upload the base colour without rebuilding the bind group. Useful
    /// later when slot values drive per-instance state changes.
    pub fn write_base_color(&self, queue: &wgpu::Queue, base_color: Vec4) {
        let data = UnlitColoredUniform {
            base_color: base_color.to_array(),
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&data));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GPU-creation paths (pipeline / bind groups) need a real wgpu device, so
    // those are exercised by the `materials` example. These tests cover what
    // can be checked without a device: registry semantics and the
    // attribute-struct layout the pipeline depends on.

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum MatId {
        Red,
        Gold,
    }

    struct FakeInstance(u32);

    #[test]
    fn empty_registry_has_no_instances() {
        let reg: MaterialInstanceRegistry<FakeInstance, MatId> = MaterialInstanceRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.get(MatId::Red).is_none());
    }

    #[test]
    fn register_then_lookup() {
        let mut reg = MaterialInstanceRegistry::new();
        reg.register(MatId::Red, FakeInstance(1));
        reg.register(MatId::Gold, FakeInstance(2));

        assert_eq!(reg.len(), 2);
        assert_eq!(reg.get(MatId::Red).map(|i| i.0), Some(1));
        assert_eq!(reg.get(MatId::Gold).map(|i| i.0), Some(2));
    }

    #[test]
    fn re_register_replaces() {
        let mut reg = MaterialInstanceRegistry::new();
        reg.register(MatId::Red, FakeInstance(1));
        reg.register(MatId::Red, FakeInstance(99));
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get(MatId::Red).map(|i| i.0), Some(99));
    }

    #[test]
    fn attribs_size_matches_layout() {
        // Pipeline's instance VertexBufferLayout uses offsets 0/16/32/48
        // for the Mat4 columns, 64 for tint, and 80 for hit_id, total 84.
        // Struct alignment is 4 (largest field alignment among Mat4 (4),
        // Vec4 (4), u32 (4)), so no tail padding. If this assertion fails
        // every mesh-material pipeline that consumes MeshInstanceAttribs is
        // out of sync with the Pod.
        assert_eq!(std::mem::size_of::<MeshInstanceAttribs>(), 84);
    }

    #[test]
    fn attribs_new_round_trips() {
        let m = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let attribs = MeshInstanceAttribs::new(m, Vec4::new(0.5, 0.6, 0.7, 1.0));
        // Mat4 is column-major; column 3 holds the translation.
        assert_eq!(attribs.model[3], [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(attribs.tint, [0.5, 0.6, 0.7, 1.0]);
        // Default hit_id is the no-hit sentinel.
        assert_eq!(attribs.hit_id, 0);
    }

    #[test]
    fn with_hit_id_round_trips() {
        let attribs = MeshInstanceAttribs::new(Mat4::IDENTITY, Vec4::ONE).with_hit_id(42);
        assert_eq!(attribs.hit_id, 42);
    }

    #[test]
    fn vertex_attributes_match_pod_layout() {
        // Pin offsets/locations the pipeline depends on. If MeshInstanceAttribs
        // grows a field, this must change in lockstep.
        let attrs = MeshInstanceAttribs::vertex_attributes(1);
        let offsets: Vec<u64> = attrs.iter().map(|a| a.offset).collect();
        assert_eq!(offsets, vec![0, 16, 32, 48, 64, 80]);
        let locations: Vec<u32> = attrs.iter().map(|a| a.shader_location).collect();
        assert_eq!(locations, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(attrs[5].format, wgpu::VertexFormat::Uint32);
    }
}
