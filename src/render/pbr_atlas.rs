//! Stylized PBR material that reads its surface parameters from two
//! atlases rather than scalar uniforms. Used by glb-authored assets whose
//! material slot resolves through the [`MaterialRegistry`](super::MaterialRegistry)
//! — the typical entry point is a Blender material named `Lumber` landing
//! at id `gltf:lumber`.
//!
//! ## Channel packing — MRE (not ORM)
//!
//! The "ORM" texture name is the industry default; this material uses a
//! related but distinct convention:
//!
//! | tex             | channel | meaning                                  |
//! |-----------------|---------|------------------------------------------|
//! | `gradient_atlas`| RGB     | albedo (sRGB)                            |
//! | `mre_atlas`     | R       | metallic (linear, `0..1`)                |
//! | `mre_atlas`     | G       | roughness (linear, `0..1`)               |
//! | `mre_atlas`     | B       | emission mask (linear, `0..1`)           |
//!
//! Final emission is `albedo * mre.b` — the gradient atlas doubles as the
//! emission colour, the MRE blue channel is just the per-texel mask.
//! There is no occlusion channel; future stylized variants that need AO
//! pack a different atlas and live as a sibling material type.
//!
//! ## Bind groups
//!
//! | group | contents                                                |
//! |-------|---------------------------------------------------------|
//! | 0     | camera ([`CameraBinding`](super::CameraBinding))        |
//! | 1     | scene env ([`Renderer::scene_layout`](super::Renderer)) |
//! | 2     | material instance — gradient + mre textures + sampler   |
//!
//! ## Vertex buffers
//!
//! Same as [`PbrMaterial`](super::PbrMaterial): slot 0 is
//! [`PosNormalUv`] per vertex, slot 1 is [`MeshInstanceAttribs`] per
//! instance. Hit-ID picking flows through identically — see that file's
//! docs.

use super::material::{MeshInstanceAttribs, MeshMaterial};
use super::renderer::Renderer;
use super::texture::{SamplerKind, SamplerRegistry, Texture};
use super::vertex::PosNormalUv;

const PBR_ATLAS_SHADER: &str = r#"
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

@group(2) @binding(0) var gradient_tex: texture_2d<f32>;
@group(2) @binding(1) var mre_tex:      texture_2d<f32>;
@group(2) @binding(2) var atlas_sampler: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
    @location(3) m0:     vec4<f32>,
    @location(4) m1:     vec4<f32>,
    @location(5) m2:     vec4<f32>,
    @location(6) m3:     vec4<f32>,
    @location(7) tint:   vec4<f32>,
    @location(8) hit_id: u32,
};

struct VsOut {
    @builtin(position) clip:      vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       world_n:   vec3<f32>,
    @location(2)       uv:        vec2<f32>,
    @location(3)       tint:      vec4<f32>,
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

// --- BRDF helpers — identical to pbr.rs, repeated here so the WGSL is
// --- self-contained. If a third PBR variant lands, lift these into a
// --- shared include string.

const PI: f32 = 3.14159265358979;

fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn g_smith_ggx(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
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
    let albedo_sample = textureSample(gradient_tex, atlas_sampler, in.uv);
    let mre           = textureSample(mre_tex,      atlas_sampler, in.uv).rgb;

    let albedo    = albedo_sample.rgb * in.tint.rgb;
    let metallic  = clamp(mre.r, 0.0, 1.0);
    let roughness = clamp(mre.g, 0.04, 1.0);
    // Smooth mask: emission scales linearly with mre.b, so an authored
    // 1.0 lights up fully, 0.0 is unlit, and intermediate values fade.
    let emission  = albedo * mre.b;

    let n = normalize(in.world_n);
    let l = normalize(scene.sun_direction.xyz);
    let v = normalize(camera.position.xyz - in.world_pos);
    let h = normalize(l + v);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

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
    out.color  = vec4<f32>(direct + ambient + emission, in.tint.a);
    out.hit_id = in.hit_id;
    return out;
}
"#;

/// Template for the atlas-PBR material: compiled pipeline + bind-group
/// layouts. Build once at `View::init`; create per-asset instances with
/// [`Self::create_instance`].
pub struct PbrAtlasMaterial {
    pipeline: wgpu::RenderPipeline,
    instance_bgl: wgpu::BindGroupLayout,
}

/// Parameters for [`PbrAtlasMaterial::create_instance`]. Takes owned
/// [`Texture`]s — the instance keeps them alive for the lifetime of its
/// bind group. There's no streaming wiring (no [`Handle`](super::Handle)
/// indirection): if a future caller needs the streaming path, mirror
/// [`PbrMaterialParams`](super::PbrMaterialParams)' handle shape then.
pub struct PbrAtlasMaterialParams {
    /// Albedo / colour atlas, sampled at the mesh's UVs to drive both the
    /// base colour and (via the MRE blue channel) emission. Must be sRGB.
    pub gradient: Texture,
    /// MRE atlas: R=metallic, G=roughness, B=emission mask. Must be
    /// linear (load with `TextureColorSpace::Linear` /
    /// `Texture::from_rgba8(.., srgb=false)`).
    pub mre: Texture,
    /// Must be a clamp-mode sampler so atlas cells don't bleed across
    /// edges. `NearestClamp` for low-poly / stylized assets that read the
    /// atlas as discrete colour bands; `LinearClamp` when smooth blending
    /// across atlas cells is wanted.
    pub sampler: SamplerKind,
}

impl PbrAtlasMaterial {
    pub fn new(renderer: &Renderer, camera_layout: &wgpu::BindGroupLayout) -> Self {
        let device = &renderer.device;

        let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PbrAtlas instance bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
            label: Some("PbrAtlas shader"),
            source: wgpu::ShaderSource::Wgsl(PBR_ATLAS_SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PbrAtlas pipeline layout"),
            bind_group_layouts: &[
                Some(camera_layout),
                Some(renderer.scene_layout()),
                Some(&instance_bgl),
            ],
            ..Default::default()
        });

        let pos_normal_uv_attrs = PosNormalUv::attributes(0);
        let instance_attrs = MeshInstanceAttribs::vertex_attributes(3);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PbrAtlas pipeline"),
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

    /// Build a material instance bound to a fully-loaded pair of atlases.
    /// The instance owns the textures — drop the instance to free them.
    pub fn create_instance(
        &self,
        renderer: &Renderer,
        samplers: &SamplerRegistry,
        params: PbrAtlasMaterialParams,
    ) -> PbrAtlasMaterialInstance {
        let PbrAtlasMaterialParams {
            gradient,
            mre,
            sampler,
        } = params;
        let bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PbrAtlas instance bind group"),
                layout: &self.instance_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gradient.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&mre.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(samplers.get(sampler)),
                    },
                ],
            });
        PbrAtlasMaterialInstance {
            _gradient: gradient,
            _mre: mre,
            bind_group,
        }
    }
}

impl MeshMaterial for PbrAtlasMaterial {
    type Instance = PbrAtlasMaterialInstance;

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        self.pipeline()
    }
}

/// A live atlas-PBR material instance — owns the two atlas textures and
/// the bind group that references them. Bind as `@group(2)` when drawing
/// through [`PbrAtlasMaterial::pipeline`].
///
/// No per-frame refresh: textures are eager-loaded (no
/// [`Handle`](super::Handle) indirection) so the bind group is stable for
/// the instance's lifetime.
pub struct PbrAtlasMaterialInstance {
    // Kept alive so the texture views inside `bind_group` remain valid.
    // wgpu's validator catches use-after-free, but the borrow checker
    // can't see through the bind group, so we hold the Textures here.
    _gradient: Texture,
    _mre: Texture,
    bind_group: wgpu::BindGroup,
}

impl PbrAtlasMaterialInstance {
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}
