//! View-side environment — sun direction, sky/ambient colour, etc. — and the
//! GPU binding pipelines read it through.
//!
//! ## Roles
//!
//! - [`ViewEnvironment`] is a plain Rust struct describing the per-frame
//!   appearance of the world: sun direction in world space, sun colour,
//!   ambient term, sky/clear colour. Produced by
//!   [`View::extract_environment`](crate::View::extract_environment) from
//!   the sim's [`SimEnvironment`](crate::SimEnvironment) (or anything else
//!   the user wants — fog, weather, etc., when those exist).
//! - [`SceneEnvironmentBinding`] is the engine-owned GPU binding it gets
//!   packed into. The [`Renderer`] holds one and writes to it each frame
//!   before the view's `render` runs. Pipelines that want scene lighting
//!   declare it at a fixed bind-group slot using
//!   [`Renderer::scene_layout`](super::Renderer::scene_layout) and bind it
//!   with [`Renderer::scene_bind_group`](super::Renderer::scene_bind_group).
//!
//! ## Why engine-driven (vs. user-driven like `CameraBinding`)
//!
//! `CameraBinding` is user-held because camera state is user-authored each
//! frame (position, target). The scene environment is *derived* from the sim
//! through a trait method, so the engine has all the inputs it needs and can
//! drive the upload itself. That keeps the View's `render` body focused on
//! drawing.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Per-frame view-side environment. Produced by
/// [`View::extract_environment`](crate::View::extract_environment); consumed
/// by pipelines that read scene lighting.
///
/// Linear-RGB colour space throughout. `sun_color` carries intensity in the
/// magnitude (so `Vec3::ZERO` = night, `Vec3::splat(3.0)` = bright noon).
#[derive(Clone, Copy, Debug)]
pub struct ViewEnvironment {
    /// Unit vector pointing *from the world toward the sun*, world-space.
    /// Below-horizon directions are fine — shaders clamp `dot(n, sun)`.
    pub sun_direction: Vec3,
    /// Linear RGB intensity. `Vec3::ZERO` disables direct sunlight.
    pub sun_color: Vec3,
    /// Linear RGB ambient. Floor for surfaces not directly lit.
    pub ambient: Vec3,
    /// Linear RGB used for the surface clear and (later) the sky dome.
    pub sky_color: Vec3,
    /// Cascaded shadow-map view-projection matrices + split distances. Leave
    /// at [`SunCascades::disabled`] when the View doesn't compute shadows;
    /// the lit shaders treat that as "fully lit" via a sentinel check on the
    /// first split.
    pub sun_cascades: SunCascades,
}

/// CSM transforms + view-space split distances for the four directional-light
/// cascades. Produced by
/// [`Camera::fit_shadow_cascades`](super::Camera::fit_shadow_cascades) plus
/// [`Camera::cascade_split_distances`](super::Camera::cascade_split_distances).
///
/// `matrices[i]` maps world-space positions into the i-th cascade's
/// shadow-map clip space (wgpu `[0, 1]` z, `[-1, 1]` xy). `splits[i]` is the
/// view-space depth where cascade `i` ends; cascade `i` covers the slice
/// from `splits[i-1]` (or the camera near) to `splits[i]`.
///
/// [`Self::disabled`] is the "no shadows" sentinel — `splits[0] == 0.0`,
/// which the lit shaders branch on to skip shadow sampling entirely.
#[derive(Clone, Copy, Debug)]
pub struct SunCascades {
    pub matrices: [Mat4; 4],
    pub splits: [f32; 4],
}

impl SunCascades {
    /// Sentinel value meaning "no shadow data". Lit shaders read
    /// `splits[0] == 0.0` as "skip shadow sampling, fully lit". Stored in
    /// [`ViewEnvironment::neutral`] and on Views that haven't computed
    /// cascades for the current frame.
    pub fn disabled() -> Self {
        Self {
            matrices: [Mat4::IDENTITY; 4],
            splits: [0.0; 4],
        }
    }
}

impl Default for SunCascades {
    fn default() -> Self {
        Self::disabled()
    }
}

impl ViewEnvironment {
    /// A neutral environment: full white ambient, no directional sun, mid-grey
    /// sky. Used when the View doesn't override
    /// [`extract_environment`](crate::View::extract_environment) or has no
    /// active zone.
    pub fn neutral() -> Self {
        Self {
            sun_direction: Vec3::Z,
            sun_color: Vec3::ZERO,
            ambient: Vec3::ONE,
            sky_color: Vec3::new(0.05, 0.07, 0.10),
            sun_cascades: SunCascades::disabled(),
        }
    }
}

impl Default for ViewEnvironment {
    fn default() -> Self {
        Self::neutral()
    }
}

/// GPU layout of [`ViewEnvironment`]. All scalar fields padded to `vec4` for
/// std140-compatible alignment so the WGSL struct can use `vec3<f32>` (which
/// occupies 16 bytes in a uniform block).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniformData {
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    ambient: [f32; 4],
    sky_color: [f32; 4],
    /// World → cascade-i shadow-map clip space, one mat4 per cascade.
    sun_cascade_view_proj: [[[f32; 4]; 4]; 4],
    /// View-space far depth of each cascade (positive distance along the
    /// camera's forward axis). `splits[0] == 0.0` is the disabled sentinel
    /// — shaders branch on it and skip shadow sampling.
    cascade_far_distances: [f32; 4],
}

impl SceneUniformData {
    fn from_env(env: &ViewEnvironment) -> Self {
        let mut sun_cascade_view_proj = [[[0.0; 4]; 4]; 4];
        for (i, m) in env.sun_cascades.matrices.iter().enumerate() {
            sun_cascade_view_proj[i] = m.to_cols_array_2d();
        }
        Self {
            sun_direction: env.sun_direction.extend(0.0).to_array(),
            sun_color: env.sun_color.extend(0.0).to_array(),
            ambient: env.ambient.extend(0.0).to_array(),
            sky_color: env.sky_color.extend(0.0).to_array(),
            sun_cascade_view_proj,
            cascade_far_distances: env.sun_cascades.splits,
        }
    }
}

/// GPU binding behind the scene environment uniform — buffer + bind-group
/// layout + bind group, plus the shadow-map texture array and its comparison
/// sampler. Vertex-and-fragment visibility for the uniform so a single
/// binding serves both shader stages; the shadow texture + sampler are
/// fragment-only.
///
/// Owned by the [`Renderer`](super::Renderer); pipelines pick up
/// [`layout`](Self::layout) at init and the engine binds
/// [`bind_group`](Self::bind_group) automatically each frame.
///
/// The shadow texture is always present in the bind group: when the View
/// hasn't opted into shadows ([`ViewConfig::shadow_map_resolution`](super::ViewConfig) is `None`)
/// the engine allocates a 1×1×4 placeholder so material pipelines don't have
/// to branch on shadows-on/off. Shaders skip sampling via the
/// `splits[0] == 0.0` sentinel from [`SunCascades::disabled`].
pub struct SceneEnvironmentBinding {
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl SceneEnvironmentBinding {
    pub(super) fn new(
        device: &wgpu::Device,
        shadow_array_view: &wgpu::TextureView,
        shadow_sampler: &wgpu::Sampler,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("currawong scene environment uniform"),
            size: std::mem::size_of::<SceneUniformData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("currawong scene environment bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let bind_group =
            build_bind_group(device, &layout, &buffer, shadow_array_view, shadow_sampler);
        Self {
            buffer,
            layout,
            bind_group,
        }
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Upload a new [`ViewEnvironment`] to the GPU. Engine-internal — the
    /// runner calls this each frame from the result of
    /// [`View::extract_environment`](crate::View::extract_environment).
    pub(super) fn write(&self, queue: &wgpu::Queue, env: &ViewEnvironment) {
        let data = SceneUniformData::from_env(env);
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&data));
    }
}

fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
    shadow_array_view: &wgpu::TextureView,
    shadow_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("currawong scene environment bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(shadow_array_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(shadow_sampler),
            },
        ],
    })
}
