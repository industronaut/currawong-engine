//! Render materials — three-tier (template / instance / per-instance attribs).
//!
//! ## Tiers
//!
//! - **Material template** — a concrete Rust struct per material kind, owning
//!   the compiled pipeline + bind-group layout for the per-material bind
//!   group. Currently the only template is [`UnlitColoredMaterial`].
//! - **Material instance** — a bind group + uniform buffer bound to a
//!   template, holding concrete uniform values. Many sim objects share one
//!   instance ("red metal," "gold trim"). Stored in a
//!   [`MaterialInstanceRegistry`].
//! - **Per-instance attribs** — `repr(C)` `Pod` struct packed into a vertex
//!   buffer with `step_mode = Instance`; varies per drawn copy. For
//!   `UnlitColored` this is [`UnlitColoredAttribs`] (model matrix + tint).
//!
//! The base abstraction is deliberately thin — no `Material` trait yet.
//! Materials share patterns (template + instance + attribs) but not an
//! interface, until we have a second material that justifies one.
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
//!         pass.set_vertex_buffer(1, instance_buf.slice(..));   // UnlitColoredAttribs
//!         pass.draw_indexed(..., 0..count);
//! ```

use std::collections::HashMap;
use std::hash::Hash;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};

use super::renderer::Renderer;

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
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) tint: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let world = model * vec4<f32>(in.pos, 1.0);
    var out: VsOut;
    out.clip = camera.view_proj * world;
    out.tint = in.tint;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return material.base_color * in.tint;
}
"#;

/// Per-instance attributes for [`UnlitColoredMaterial`]: model matrix plus a
/// per-instance tint multiplier (multiplied with the material instance's
/// `base_color` in the shader).
///
/// Pack instances of this struct into a `wgpu::Buffer` with `VERTEX` usage,
/// bind it as vertex buffer slot 1, and the pipeline's baked-in instance
/// layout will read it directly. Stride is `size_of::<UnlitColoredAttribs>()`
/// = 80 bytes (Mat4 columns 0/16/32/48, tint 64).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct UnlitColoredAttribs {
    /// `Mat4` as four column-major vec4s; layout matches
    /// [`mat4_instance_attributes`](super::mat4_instance_attributes).
    pub model: [[f32; 4]; 4],
    /// Linear RGBA in `[0, 1]`; multiplied with the material's base colour.
    /// Use `[1.0; 4]` for "no per-instance tint."
    pub tint: [f32; 4],
}

impl UnlitColoredAttribs {
    pub fn new(model: Mat4, tint: Vec4) -> Self {
        Self {
            model: model.to_cols_array_2d(),
            tint: tint.to_array(),
        }
    }
}

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
/// - vertex buffer slot 1 — [`UnlitColoredAttribs`] per instance
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
                    // Slot 1: per-instance Mat4 + tint.
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<UnlitColoredAttribs>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                            wgpu::VertexAttribute {
                                offset: 16,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                            wgpu::VertexAttribute {
                                offset: 32,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                            wgpu::VertexAttribute {
                                offset: 48,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                            wgpu::VertexAttribute {
                                offset: 64,
                                shader_location: 5,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                        ],
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
                    // Opt out of the hit-ID attachment (#56 PR 1): unlit
                    // meshes become writers in PR 3.
                    renderer.id_target_opt_out(),
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
        // Pipeline's instance VertexBufferLayout uses offsets 0/16/32/48 for
        // the Mat4 columns and 64 for the tint, total 80. If this assertion
        // fails the pipeline's attribute layout is out of sync with the Pod.
        assert_eq!(std::mem::size_of::<UnlitColoredAttribs>(), 80);
    }

    #[test]
    fn attribs_new_round_trips() {
        let m = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let attribs = UnlitColoredAttribs::new(m, Vec4::new(0.5, 0.6, 0.7, 1.0));
        // Mat4 is column-major; column 3 holds the translation.
        assert_eq!(attribs.model[3], [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(attribs.tint, [0.5, 0.6, 0.7, 1.0]);
    }
}
