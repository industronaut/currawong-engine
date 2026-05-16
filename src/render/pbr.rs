//! Physically-based render material — metallic-roughness, one directional
//! light, no IBL, no shadows. Reads the engine-managed camera and scene
//! environment bind groups; takes a per-material albedo texture plus scalar
//! metallic/roughness factors.
//!
//! ## Bind groups
//!
//! | group | contents                                                |
//! |-------|---------------------------------------------------------|
//! | 0     | camera ([`CameraBinding`](super::CameraBinding))        |
//! | 1     | scene env ([`Renderer::scene_layout`](super::Renderer)) |
//! | 2     | material instance — albedo texture + sampler + uniforms |
//!
//! ## Vertex buffers
//!
//! | slot | contents                                                            |
//! |------|---------------------------------------------------------------------|
//! | 0    | [`PosNormalUv`] per vertex (32 B)                                   |
//! | 1    | [`PbrInstanceAttribs`] per instance — model matrix + tint + hit ID (84) |
//!
//! ## Picking (#56 PR 3)
//!
//! Every draw writes the per-instance [`PbrInstanceAttribs::hit_id`] to the
//! engine's `R32Uint` hit-ID attachment, flat-interpolated through
//! `@location(1)`. `hit_id == 0` is the no-hit sentinel and matches the
//! attachment's clear value, so instances that don't care about picking
//! can leave it at the `0` default from [`PbrInstanceAttribs::new`].
//! Callers that do want a clickable mesh call
//! [`Renderer::reserve_object`](super::Renderer::reserve_object) once per
//! parent and feed the result through
//! [`PbrInstanceAttribs::with_hit_id`].
//!
//! ## What's not in v1
//!
//! - **Normal mapping** — needs tangents in the vertex layout. Add a
//!   `PosNormalUvTangent` layout and a `PbrTangentMaterial` (or extend this
//!   one behind a feature flag) when it's wanted.
//! - **MR map** — metallic/roughness are scalar uniforms today. Adding a
//!   texture is a bind-group entry plus a sample in the shader.
//! - **IBL / image-based lighting** — ambient is a flat term from the scene
//!   environment. Reflection / sky probes land alongside a sky dome.
//! - **Shadow maps / multiple lights** — single directional light, no
//!   occlusion. Multi-light support means a different scene-uniform shape.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};

use super::renderer::Renderer;
use super::texture::{SamplerKind, SamplerRegistry, Texture};
use super::vertex::PosNormalUv;

const PBR_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    right:     vec4<f32>,
    up:        vec4<f32>,
    position:  vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct Scene {
    sun_direction: vec4<f32>,
    sun_color:     vec4<f32>,
    ambient:       vec4<f32>,
    sky_color:     vec4<f32>,
};
@group(1) @binding(0) var<uniform> scene: Scene;

struct MaterialFactors {
    albedo:    vec4<f32>,
    metallic:  f32,
    roughness: f32,
    _pad0:     f32,
    _pad1:     f32,
};
@group(2) @binding(0) var<uniform> material: MaterialFactors;
@group(2) @binding(1) var albedo_tex:     texture_2d<f32>;
@group(2) @binding(2) var albedo_sampler: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
    // Per-instance model matrix as four vec4 columns.
    @location(3) m0:     vec4<f32>,
    @location(4) m1:     vec4<f32>,
    @location(5) m2:     vec4<f32>,
    @location(6) m3:     vec4<f32>,
    @location(7) tint:   vec4<f32>,
    // Per-instance hit ID, written into the engine's R32Uint attachment by
    // the fragment shader. 0 = no-hit sentinel (matches the attachment's
    // clear value); non-zero comes from Renderer::reserve_object.
    @location(8) hit_id: u32,
};

struct VsOut {
    @builtin(position) clip:      vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       world_n:   vec3<f32>,
    @location(2)       uv:        vec2<f32>,
    @location(3)       tint:      vec4<f32>,
    // Integer attributes can't be linearly interpolated; flat ships one
    // value per primitive.
    @location(4) @interpolate(flat) hit_id: u32,
};

struct FsOut {
    @location(0) color:  vec4<f32>,
    @location(1) hit_id: u32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let world = model * vec4<f32>(in.position, 1.0);
    // Assumes orthogonal (no non-uniform scale) model matrix — a rotation
    // of the vertex normal by the upper-left 3x3 is the correct transform.
    // If non-uniform scale shows up, switch to transpose(inverse(model_3x3)).
    let n_world = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    var out: VsOut;
    out.clip      = camera.view_proj * world;
    out.world_pos = world.xyz;
    out.world_n   = n_world;
    out.uv        = in.uv;
    out.tint      = in.tint;
    out.hit_id    = in.hit_id;
    return out;
}

// --- BRDF helpers (metallic-roughness, Cook-Torrance specular) -----------

const PI: f32 = 3.14159265358979;

fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn g_smith_ggx(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    // Smith with Schlick-GGX, remapped for direct lighting.
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn f_schlick(v_dot_h: f32, f0: vec3<f32>) -> vec3<f32> {
    let one_minus = clamp(1.0 - v_dot_h, 0.0, 1.0);
    return f0 + (vec3<f32>(1.0) - f0) * pow(one_minus, 5.0);
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let albedo = textureSample(albedo_tex, albedo_sampler, in.uv).rgb
                 * material.albedo.rgb
                 * in.tint.rgb;

    let n = normalize(in.world_n);
    let l = normalize(scene.sun_direction.xyz);
    let v = normalize(camera.position.xyz - in.world_pos);
    let h = normalize(l + v);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    let metallic = clamp(material.metallic, 0.0, 1.0);
    let roughness = clamp(material.roughness, 0.04, 1.0);

    // Non-metals reflect ~4% at normal incidence; metals tint the spec by
    // their albedo. Standard metallic-roughness mix.
    let f0_dielectric = vec3<f32>(0.04);
    let f0 = mix(f0_dielectric, albedo, metallic);

    let d = d_ggx(n_dot_h, roughness);
    let g = g_smith_ggx(n_dot_v, n_dot_l, roughness);
    let f = f_schlick(v_dot_h, f0);

    let specular = (d * g) * f / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;

    let direct = (diffuse + specular) * scene.sun_color.rgb * n_dot_l;
    let ambient = scene.ambient.rgb * albedo;

    var out: FsOut;
    out.color  = vec4<f32>(direct + ambient, material.albedo.a * in.tint.a);
    out.hit_id = in.hit_id;
    return out;
}
"#;

/// Per-instance attribs for [`PbrMaterial`]: model matrix, tint multiplier,
/// and GPU hit ID (#56 PR 3). Identical shape to
/// [`UnlitColoredAttribs`](super::UnlitColoredAttribs), kept as a separate
/// type so each material owns its own layout contract.
///
/// `hit_id` is written to the engine's `R32Uint` hit-ID attachment for
/// every pixel the instance covers. `0` is the no-hit sentinel and matches
/// the attachment's clear value, so the default constructor leaves it at
/// `0` (instance is drawn but contributes nothing to picking). Opt in via
/// [`Self::with_hit_id`], passing an ID returned by
/// [`Renderer::reserve_object`](super::Renderer::reserve_object).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct PbrInstanceAttribs {
    pub model: [[f32; 4]; 4],
    pub tint: [f32; 4],
    pub hit_id: u32,
}

impl PbrInstanceAttribs {
    pub fn new(model: Mat4, tint: Vec4) -> Self {
        Self {
            model: model.to_cols_array_2d(),
            tint: tint.to_array(),
            hit_id: 0,
        }
    }

    /// Builder: stamp this instance with the GPU hit ID returned by
    /// [`Renderer::reserve_object`](super::Renderer::reserve_object). All
    /// pixels the instance covers in the opaque pass will carry this ID in
    /// the hit-ID attachment, so a cursor over any of them resolves back
    /// to the originating `WorldObjectId`.
    pub fn with_hit_id(mut self, hit_id: u32) -> Self {
        self.hit_id = hit_id;
        self
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PbrMaterialUniform {
    albedo: [f32; 4],
    metallic: f32,
    roughness: f32,
    _pad0: f32,
    _pad1: f32,
}

/// Template for the PBR material: compiled pipeline + bind-group layouts.
/// Build once at `View::init`. Create instances with
/// [`Self::create_instance`].
pub struct PbrMaterial {
    pipeline: wgpu::RenderPipeline,
    instance_bgl: wgpu::BindGroupLayout,
}

/// Parameters for [`PbrMaterial::create_instance`].
pub struct PbrMaterialParams<'a> {
    pub albedo: &'a Texture,
    pub sampler: SamplerKind,
    /// Multiplied with `albedo` texture sample. Use `Vec4::ONE` to pass the
    /// texture through unchanged. `albedo.a` doubles as opacity.
    pub albedo_factor: Vec4,
    /// `0.0` = dielectric (plastic, wood), `1.0` = metal.
    pub metallic: f32,
    /// `0.04..1.0` (clamped in shader). Lower = sharper highlights.
    pub roughness: f32,
}

impl PbrMaterial {
    pub fn new(renderer: &Renderer, camera_layout: &wgpu::BindGroupLayout) -> Self {
        let device = &renderer.device;

        let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Pbr instance bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pbr shader"),
            source: wgpu::ShaderSource::Wgsl(PBR_SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pbr pipeline layout"),
            bind_group_layouts: &[
                Some(camera_layout),
                Some(renderer.scene_layout()),
                Some(&instance_bgl),
            ],
            ..Default::default()
        });

        let pos_normal_uv_attrs = PosNormalUv::attributes(0);
        let mat4_attrs = super::mat4_instance_attributes(3);
        let instance_attrs = [
            mat4_attrs[0],
            mat4_attrs[1],
            mat4_attrs[2],
            mat4_attrs[3],
            wgpu::VertexAttribute {
                offset: 64,
                shader_location: 7,
                format: wgpu::VertexFormat::Float32x4,
            },
            // hit_id at offset 80, location 8 — the per-instance counterpart
            // to the per-vertex cell_id terrain already writes (#56 PR 3).
            super::u32_id_instance_attribute(80, 8),
        ];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pbr pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: PosNormalUv::STRIDE,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &pos_normal_uv_attrs,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<PbrInstanceAttribs>() as u64,
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
                    // attachment (#56 PR 3). Instances that don't care about
                    // picking leave their `hit_id` at the 0 default, which
                    // matches the attachment's clear value — semantically
                    // identical to PR 1's opt-out shape.
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

    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    pub fn create_instance(
        &self,
        renderer: &Renderer,
        samplers: &SamplerRegistry,
        params: PbrMaterialParams<'_>,
    ) -> PbrMaterialInstance {
        let data = PbrMaterialUniform {
            albedo: params.albedo_factor.to_array(),
            metallic: params.metallic,
            roughness: params.roughness,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pbr instance uniform"),
            size: std::mem::size_of::<PbrMaterialUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        renderer
            .queue
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&data));
        let bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Pbr instance bind group"),
                layout: &self.instance_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&params.albedo.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(samplers.get(params.sampler)),
                    },
                ],
            });
        PbrMaterialInstance { buffer, bind_group }
    }
}

/// A live PBR material instance — uniform buffer (factors) + bind group
/// holding the albedo texture, sampler, and uniform. Bind as `@group(2)`
/// when drawing through the material's pipeline.
pub struct PbrMaterialInstance {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl PbrMaterialInstance {
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Re-upload the factor uniform without rebuilding the bind group.
    pub fn write_factors(
        &self,
        queue: &wgpu::Queue,
        albedo_factor: Vec4,
        metallic: f32,
        roughness: f32,
    ) {
        let data = PbrMaterialUniform {
            albedo: albedo_factor.to_array(),
            metallic,
            roughness,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&data));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribs_size_matches_layout() {
        // Pipeline reads model at offsets 0/16/32/48, tint at 64, and
        // hit_id at 80; total 84. Struct alignment is 4 (largest field
        // alignment among Mat4 (4), Vec4 (4), u32 (4)), so no tail padding.
        // If this fails the pipeline's instance layout is out of sync with
        // the Pod.
        assert_eq!(std::mem::size_of::<PbrInstanceAttribs>(), 84);
    }

    #[test]
    fn with_hit_id_round_trips() {
        let attribs = PbrInstanceAttribs::new(Mat4::IDENTITY, Vec4::ONE).with_hit_id(7);
        assert_eq!(attribs.hit_id, 7);
        // Default is 0 (no-hit sentinel).
        assert_eq!(PbrInstanceAttribs::new(Mat4::IDENTITY, Vec4::ONE).hit_id, 0);
    }

    #[test]
    fn material_uniform_is_std140_safe() {
        // 16 (albedo) + 4 (metallic) + 4 (roughness) + 8 (pad) = 32 bytes,
        // a multiple of 16. WGSL uniform blocks require the trailing
        // alignment to be a multiple of the struct's largest member (16
        // for vec4); padding ensures it.
        assert_eq!(std::mem::size_of::<PbrMaterialUniform>(), 32);
    }
}
