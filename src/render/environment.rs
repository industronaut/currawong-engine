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
use glam::Vec3;

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
        }
    }
}

impl Default for ViewEnvironment {
    fn default() -> Self {
        Self::neutral()
    }
}

/// GPU layout of [`ViewEnvironment`]. All fields padded to `vec4` for
/// std140-compatible alignment so the WGSL struct can use `vec3<f32>` (which
/// occupies 16 bytes in a uniform block).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniformData {
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    ambient: [f32; 4],
    sky_color: [f32; 4],
}

impl SceneUniformData {
    fn from_env(env: &ViewEnvironment) -> Self {
        Self {
            sun_direction: env.sun_direction.extend(0.0).to_array(),
            sun_color: env.sun_color.extend(0.0).to_array(),
            ambient: env.ambient.extend(0.0).to_array(),
            sky_color: env.sky_color.extend(0.0).to_array(),
        }
    }
}

/// GPU binding behind the scene environment uniform — buffer + bind-group
/// layout + bind group, with a vertex-and-fragment visibility so a single
/// binding serves both shader stages.
///
/// Owned by the [`Renderer`](super::Renderer); pipelines pick up
/// [`layout`](Self::layout) at init and the engine binds
/// [`bind_group`](Self::bind_group) automatically each frame.
pub struct SceneEnvironmentBinding {
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl SceneEnvironmentBinding {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("currawong scene environment uniform"),
            size: std::mem::size_of::<SceneUniformData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("currawong scene environment bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("currawong scene environment bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
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
