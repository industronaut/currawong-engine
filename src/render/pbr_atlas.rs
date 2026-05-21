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
//! | `albedo_atlas`| RGB     | albedo (sRGB)                            |
//! | `mre_atlas`     | R       | metallic (linear, `0..1`)                |
//! | `mre_atlas`     | G       | roughness (linear, `0..1`)               |
//! | `mre_atlas`     | B       | emission mask (linear, `0..1`)           |
//!
//! Final emission is `albedo * mre.b` — the albedo atlas doubles as the
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
//! | 2     | material instance — albedo + mre textures + sampler   |
//!
//! ## Vertex buffers
//!
//! Same as [`PbrMaterial`](super::PbrMaterial): slot 0 is
//! [`PosNormalUv`] per vertex, slot 1 is [`MeshInstanceAttribs`] per
//! instance. Hit-ID picking flows through identically — see that file's
//! docs.

use std::sync::Mutex;

use super::asset_server::{AssetServer, TextureSource};
use super::handle::Handle;
use super::material::{MeshMaterial, build_pbr_style_pipeline};
use super::renderer::Renderer;
use super::texture::{SamplerKind, SamplerRegistry, Texture};

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

@group(2) @binding(0) var albedo_tex: texture_2d<f32>;
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
    let albedo_sample = textureSample(albedo_tex, atlas_sampler, in.uv);
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

/// Parameters for [`PbrAtlasMaterial::create_instance`].
///
/// Both atlases come in as [`Handle<Texture>`]s rather than borrows so
/// the instance can survive its handle's load lifecycle — the bind group
/// is initially built against whatever the [`AssetServer`] currently
/// resolves the handle to (the magenta fallback while it's `Loading`,
/// the real texture if `Ready`), and rebuilt on the frame either handle
/// transitions via [`PbrAtlasMaterialInstance::refresh`]. Pass
/// [`Handle::ready`] if you already have the texture in hand and want a
/// non-streaming wiring.
pub struct PbrAtlasMaterialParams {
    /// Albedo / colour atlas, sampled at the mesh's UVs to drive both the
    /// base colour and (via the MRE blue channel) emission. Must be sRGB.
    pub albedo: Handle<Texture>,
    /// MRE atlas: R=metallic, G=roughness, B=emission mask. Must be
    /// linear (load with `TextureColorSpace::Linear`).
    pub mre: Handle<Texture>,
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

        let pipeline =
            build_pbr_style_pipeline(renderer, "PbrAtlas", &shader, camera_layout, &instance_bgl);

        Self {
            pipeline,
            instance_bgl,
        }
    }

    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    /// Build a material instance bound to a pair of atlas
    /// [`Handle<Texture>`]s.
    ///
    /// The initial bind group is built against whatever the asset server
    /// currently resolves each handle to — the magenta fallback if a
    /// handle is still `Loading`, the real texture if it's already
    /// `Ready` (e.g. when the caller used [`Handle::ready`]). Subsequent
    /// [`PbrAtlasMaterialInstance::refresh`] calls swap the bind group
    /// over as either handle's state changes.
    pub fn create_instance(
        &self,
        renderer: &Renderer,
        samplers: &SamplerRegistry,
        asset_server: &AssetServer,
        params: PbrAtlasMaterialParams,
    ) -> PbrAtlasMaterialInstance {
        let PbrAtlasMaterialParams {
            albedo,
            mre,
            sampler,
        } = params;
        let resolved_albedo = asset_server.resolve_texture(&albedo);
        let resolved_mre = asset_server.resolve_texture(&mre);
        let bind_group = build_instance_bind_group(
            &renderer.device,
            &self.instance_bgl,
            resolved_albedo.view,
            resolved_mre.view,
            samplers.get(sampler),
        );
        // Lift the sources out before moving the handles into the struct
        // — `resolved_*` borrow `albedo` / `mre`, so the borrow has to
        // end before the struct literal moves them.
        let last_albedo_source = resolved_albedo.source;
        let last_mre_source = resolved_mre.source;
        PbrAtlasMaterialInstance {
            pipeline: self.pipeline.clone(),
            instance_bgl: self.instance_bgl.clone(),
            albedo,
            mre,
            sampler_kind: sampler,
            state: Mutex::new(AtlasInstanceState {
                bind_group,
                last_albedo_source,
                last_mre_source,
            }),
        }
    }
}

/// Build the per-instance bind group. Shared between
/// [`PbrAtlasMaterial::create_instance`] and the refresh path so the
/// layout + entry order live in one place.
fn build_instance_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    albedo_view: &wgpu::TextureView,
    mre_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PbrAtlas instance bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(albedo_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(mre_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// A live atlas-PBR material instance — holds streaming handles to the
/// two atlas textures + the cached bind group built against their
/// currently-resolved views. Implements [`MeshMaterial`], so an
/// `Arc<PbrAtlasMaterialInstance>` unsizes into `Arc<dyn MeshMaterial>`
/// and slots into [`MeshTemplate::materials`](super::MeshTemplate) or
/// [`MaterialRegistry`](super::MaterialRegistry).
///
/// The bound textures flex with the underlying [`Handle<Texture>`]s:
/// [`refresh`](MeshMaterial::refresh) checks both handles each frame and
/// rebuilds iff either resolved [`TextureSource`] changed (real ↔
/// fallback ↔ forced-fallback). The mutable bits live behind a `Mutex`
/// so trait `&self` methods compose with `Arc<dyn MeshMaterial>` shared
/// ownership.
pub struct PbrAtlasMaterialInstance {
    pipeline: wgpu::RenderPipeline,
    instance_bgl: wgpu::BindGroupLayout,
    albedo: Handle<Texture>,
    mre: Handle<Texture>,
    sampler_kind: SamplerKind,
    state: Mutex<AtlasInstanceState>,
}

struct AtlasInstanceState {
    bind_group: wgpu::BindGroup,
    /// Which view each side of the cached `bind_group` is currently
    /// built against — used by `refresh` to decide whether a rebuild is
    /// needed.
    last_albedo_source: TextureSource,
    last_mre_source: TextureSource,
}

impl PbrAtlasMaterialInstance {
    /// The albedo atlas handle. Cheap to clone — share across instances
    /// that read the same atlas.
    pub fn albedo_handle(&self) -> &Handle<Texture> {
        &self.albedo
    }

    /// The MRE atlas handle. Cheap to clone.
    pub fn mre_handle(&self) -> &Handle<Texture> {
        &self.mre
    }
}

impl MeshMaterial for PbrAtlasMaterialInstance {
    fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    fn bind(&self, pass: &mut wgpu::RenderPass<'_>, group: u32) {
        let bind_group = self
            .state
            .lock()
            .expect("PbrAtlasMaterial state lock")
            .bind_group
            .clone();
        pass.set_bind_group(group, Some(&bind_group), &[]);
    }

    fn refresh(&self, renderer: &Renderer, samplers: &SamplerRegistry, assets: &AssetServer) {
        let resolved_albedo = assets.resolve_texture(&self.albedo);
        let resolved_mre = assets.resolve_texture(&self.mre);
        let mut state = self.state.lock().expect("PbrAtlasMaterial state lock");
        if resolved_albedo.source == state.last_albedo_source
            && resolved_mre.source == state.last_mre_source
        {
            return;
        }
        state.bind_group = build_instance_bind_group(
            &renderer.device,
            &self.instance_bgl,
            resolved_albedo.view,
            resolved_mre.view,
            samplers.get(self.sampler_kind),
        );
        state.last_albedo_source = resolved_albedo.source;
        state.last_mre_source = resolved_mre.source;
    }
}
