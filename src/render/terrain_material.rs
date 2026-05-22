//! Material for terrain meshes.
//!
//! Two pipelines compiled from one shader and one vertex format
//! ([`TerrainVertex`](super::TerrainVertex), `pos` + `normal` + per-vertex
//! `color`):
//!
//! - [`Self::opaque_pipeline`] — opaque solid terrain. Depth write on,
//!   `BlendState::REPLACE`.
//! - [`Self::transparent_pipeline`] — alpha-blended liquids. Depth **test**
//!   on (so solid terrain occludes liquid behind it) but depth **write**
//!   off (so liquid doesn't occlude subsequent liquid fragments).
//!
//! Instances ([`TerrainMaterialInstance`]) hold a tint uniform multiplied
//! with the per-vertex colour in the shader. Solid terrain uses a white
//! tint (so per-vertex top/wall shading shows through); each liquid kind
//! gets its own tinted instance (water blue, lava orange, …).
//!
//! ## Bind groups
//!
//! | group | contents                                                       |
//! |-------|----------------------------------------------------------------|
//! | 0     | camera ([`CameraBinding`](super::CameraBinding))               |
//! | 1     | scene env ([`Renderer::scene_layout`](super::Renderer))        |
//! | 2     | material instance — tint uniform                               |
//! | 3     | hit-ID base ([`Renderer::id_base_layout`](super::Renderer)) — per-chunk u32 written by `TerrainRenderer::draw_solid` each frame |
//!
//! ## Wiring
//!
//! ```text
//! init:    let material = TerrainMaterial::new(&renderer, camera.layout());
//!          let solid = material.create_instance(&renderer, Vec4::splat(1.0));
//!          let water = material.create_instance(&renderer,
//!                          Vec4::new(0.2, 0.45, 0.85, 0.55));
//! draw:    pass.set_pipeline(material.opaque_pipeline());
//!          pass.set_bind_group(0, camera.bind_group(), &[]);
//!          pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
//!          // draw_solid binds groups 2 (material instance) and 3 (per-chunk
//!          // ID base) per draw call.
//!          terrain_renderer.draw_solid(pass, renderer, zone, &solid);
//!          pass.set_pipeline(material.transparent_pipeline());
//!          terrain_renderer.draw_liquids(pass, &liquid_instances);
//! ```

use bytemuck::{Pod, Zeroable};
use glam::Vec4;

use super::renderer::Renderer;
use super::terrain::TerrainVertex;

const TERRAIN_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    right:     vec4<f32>,
    up:        vec4<f32>,
    position:  vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct Scene {
    sun_direction:         vec4<f32>,
    sun_color:             vec4<f32>,
    ambient:               vec4<f32>,
    sky_color:             vec4<f32>,
    sun_cascade_view_proj: array<mat4x4<f32>, 4>,
    cascade_far_distances: vec4<f32>,
};
@group(1) @binding(0) var<uniform> scene:          Scene;
@group(1) @binding(1) var          shadow_map:     texture_depth_2d_array;
@group(1) @binding(2) var          shadow_sampler: sampler_comparison;

struct Material {
    tint: vec4<f32>,
};
@group(2) @binding(0) var<uniform> material: Material;

struct IdBase {
    // First u32 is the per-frame chunk base; the three trailing u32s are
    // padding out to 16 bytes (wgpu requires uniform buffers to meet
    // their struct's WGSL size; `vec3<u32>` would force 16-byte alignment
    // and a 32-byte struct, hence individual u32s instead).
    base_id: u32,
    _pad0:   u32,
    _pad1:   u32,
    _pad2:   u32,
};
@group(3) @binding(0) var<uniform> id_base: IdBase;

struct VsIn {
    @location(0) pos:               vec3<f32>,
    @location(1) normal:            vec3<f32>,
    @location(2) color:             vec4<f32>,
    @location(3) cell_id_in_chunk:  u32,
};

struct VsOut {
    @builtin(position)             clip:      vec4<f32>,
    @location(0)                   n:         vec3<f32>,
    @location(1)                   color:     vec4<f32>,
    // Integer values must be flat-interpolated through vertex→fragment.
    @location(2) @interpolate(flat) cell_id:  u32,
    // World position passed through so the fragment shader can sample
    // cascaded shadow maps. Terrain has no model matrix — `in.pos` is
    // already in world space.
    @location(3)                   world_pos: vec3<f32>,
};

struct FsOut {
    @location(0) color:   vec4<f32>,
    // Per-pixel hit ID. The opaque pipeline writes this slot (R32Uint);
    // the transparent pipeline declares the slot but its write mask is
    // empty, so the value is discarded — terrain is the only PR-2 writer.
    @location(1) hit_id:  u32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip      = camera.view_proj * vec4<f32>(in.pos, 1.0);
    // Terrain has no model matrix — vertex positions and normals are already
    // in world space, so the normal passes through untouched.
    out.n         = in.normal;
    out.color     = in.color * material.tint;
    out.cell_id   = in.cell_id_in_chunk + id_base.base_id;
    out.world_pos = in.pos;
    return out;
}

// Cascaded-shadow-map sampling — same shape as `pbr.rs` / `pbr_atlas.rs`.
// Terrain is a receiver only in v1: hills don't cast shadows because
// terrain meshes aren't recorded in `View::shadow_pass`. Hills + cliffs
// casting onto lower ground would need a depth-only `TerrainVertex`
// pipeline, a follow-up.
fn shadow_visibility(world_pos: vec3<f32>, view_z: f32) -> f32 {
    if (scene.cascade_far_distances.x <= 0.0) {
        return 1.0;
    }
    var cascade: i32 = -1;
    for (var i: i32 = 0; i < 4; i = i + 1) {
        if (view_z < scene.cascade_far_distances[i]) {
            cascade = i;
            break;
        }
    }
    if (cascade < 0) {
        return 1.0;
    }
    let light_clip = scene.sun_cascade_view_proj[cascade] * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 ||
        shadow_uv.y < 0.0 || shadow_uv.y > 1.0 ||
        ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    return textureSampleCompareLevel(shadow_map, shadow_sampler, shadow_uv, cascade, ndc.z);
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let n          = normalize(in.n);
    let l          = normalize(scene.sun_direction.xyz);
    let n_dot_l    = max(dot(n, l), 0.0);
    let view_z     = length(in.world_pos - camera.position.xyz);
    let visibility = shadow_visibility(in.world_pos, view_z);
    let direct     = scene.sun_color.rgb * n_dot_l * visibility;
    let lit        = direct + scene.ambient.rgb;
    var out: FsOut;
    out.color  = vec4<f32>(in.color.rgb * lit, in.color.a);
    out.hit_id = in.cell_id;
    return out;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainUniform {
    tint: [f32; 4],
}

/// Terrain material template: opaque + transparent pipelines sharing one
/// vertex format and bind-group layout. Build once at `View::init`; create
/// per-purpose instances ([`TerrainMaterialInstance`]) via
/// [`Self::create_instance`].
pub struct TerrainMaterial {
    opaque_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    instance_bgl: wgpu::BindGroupLayout,
}

impl TerrainMaterial {
    pub fn new(renderer: &Renderer, camera_layout: &wgpu::BindGroupLayout) -> Self {
        let device = &renderer.device;

        let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain instance bgl"),
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain shader"),
            source: wgpu::ShaderSource::Wgsl(TERRAIN_SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain pipeline layout"),
            bind_group_layouts: &[
                Some(camera_layout),
                Some(renderer.scene_layout()),
                Some(&instance_bgl),
                Some(renderer.id_base_layout()),
            ],
            ..Default::default()
        });

        let attributes = [
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x4,
            },
            // cell_id_in_chunk: u32 — feeds the shader's per-vertex hit-ID
            // local index. The shader adds the chunk's frame-scoped
            // `base_id` uniform (group 3) to produce the unique hit ID
            // written to the R32Uint attachment.
            wgpu::VertexAttribute {
                offset: 40,
                shader_location: 3,
                format: wgpu::VertexFormat::Uint32,
            },
        ];
        let buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TerrainVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &attributes,
        }];

        // Walls face four directions; mixing top/wall geometry in one mesh
        // with no fixed winding contract means back-face culling would punch
        // visible holes. Skip culling — slight overdraw, but correctness.
        let primitive = wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        };

        let opaque_blend = wgpu::BlendState::REPLACE;
        let alpha_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };

        let depth_opaque = renderer
            .depth_format()
            .map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });
        let depth_transparent = renderer
            .depth_format()
            .map(|format| wgpu::DepthStencilState {
                format,
                // Test against solid terrain depth so liquid behind a cliff
                // is occluded, but don't write — otherwise the first liquid
                // fragment would occlude later transparents at the same
                // depth.
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });

        let make_pipeline =
            |label: &'static str,
             blend: wgpu::BlendState,
             depth_stencil: Option<wgpu::DepthStencilState>,
             id_target: Option<wgpu::ColorTargetState>| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &buffers,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: renderer.surface_format(),
                                blend: Some(blend),
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            id_target,
                        ],
                    }),
                    primitive,
                    depth_stencil,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };

        // Opaque terrain *writes* hit IDs (the first writer of the #56 ID
        // attachment); transparent liquids share the shader but opt out —
        // alpha-blended IDs make no semantic sense (one pixel = one ID).
        let opaque_pipeline = make_pipeline(
            "Terrain opaque pipeline",
            opaque_blend,
            depth_opaque,
            renderer.id_target_writer(),
        );
        let transparent_pipeline = make_pipeline(
            "Terrain transparent pipeline",
            alpha_blend,
            depth_transparent,
            renderer.id_target_opt_out(),
        );

        Self {
            opaque_pipeline,
            transparent_pipeline,
            instance_bgl,
        }
    }

    pub fn opaque_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.opaque_pipeline
    }

    pub fn transparent_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.transparent_pipeline
    }

    /// Create a material instance with a per-instance tint (linear RGBA in
    /// `[0, 1]`; alpha is meaningful only on the transparent pipeline).
    pub fn create_instance(&self, renderer: &Renderer, tint: Vec4) -> TerrainMaterialInstance {
        let data = TerrainUniform {
            tint: tint.to_array(),
        };
        let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain material instance uniform"),
            size: std::mem::size_of::<TerrainUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        renderer
            .queue
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&data));
        let bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Terrain material instance bg"),
                layout: &self.instance_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
        TerrainMaterialInstance { buffer, bind_group }
    }
}

/// A live instance of [`TerrainMaterial`] — a uniform buffer + bind group
/// holding one tint. Bind as `@group(1)` when drawing through either of the
/// material's pipelines.
pub struct TerrainMaterialInstance {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl TerrainMaterialInstance {
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Re-upload the tint without rebuilding the bind group.
    pub fn write_tint(&self, queue: &wgpu::Queue, tint: Vec4) {
        let data = TerrainUniform {
            tint: tint.to_array(),
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&data));
    }
}
