//! Depth-only pipelines for the directional-light shadow pass.
//!
//! Views that opt into shadows (via
//! [`ViewConfig::shadow_map_resolution`](super::ViewConfig)) build a
//! [`ShadowMeshPipeline`] in `init` and record draws into each cascade's
//! depth attachment from [`View::shadow_pass`](super::View::shadow_pass).
//!
//! The pipeline reads the per-cascade light view-projection at
//! `@group(0) @binding(0)` — the engine binds the right cascade's bind
//! group ([`Renderer::shadow_cascade_bind_group`](super::Renderer::shadow_cascade_bind_group))
//! before invoking the View callback. Vertex layout matches the standard
//! mesh layout ([`PosNormalUv`] + [`MeshInstanceAttribs`]) so views can reuse
//! the same vertex + instance buffers they already drive their lit pipelines
//! with — only normal/uv/tint/hit_id are ignored.
//!
//! A small constant depth bias is baked into the pipeline state to mitigate
//! self-shadowing acne; receivers compensate with a per-fragment slope-scale
//! bias when sampling.

use super::material::MeshInstanceAttribs;
use super::renderer::Renderer;
use super::vertex::PosNormalUv;

const SHADOW_SHADER: &str = r#"
struct CascadeUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> cascade: CascadeUniform;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) _normal:  vec3<f32>,
    @location(2) _uv:      vec2<f32>,
    // Per-instance model matrix as four vec4 columns.
    @location(3) m0:       vec4<f32>,
    @location(4) m1:       vec4<f32>,
    @location(5) m2:       vec4<f32>,
    @location(6) m3:       vec4<f32>,
    @location(7) _tint:    vec4<f32>,
    @location(8) _hit_id:  u32,
};

@vertex
fn vs_main(in: VsIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let world = model * vec4<f32>(in.position, 1.0);
    return cascade.view_proj * world;
}
"#;

/// Depth-only pipeline keyed on the canonical [`PosNormalUv`] +
/// [`MeshInstanceAttribs`] vertex layout. Build once in `View::init`; bind
/// in [`View::shadow_pass`](super::View::shadow_pass) along with the engine's
/// cascade bind group at `@group(0)`.
///
/// Future direction: parallel pipelines for other canonical vertex layouts
/// (e.g. terrain's `TerrainVertex`) would live alongside this one in this
/// module.
pub struct ShadowMeshPipeline {
    pipeline: wgpu::RenderPipeline,
}

impl ShadowMeshPipeline {
    /// Build the depth-only mesh pipeline. Targets the engine's shadow-map
    /// format ([`Renderer::shadow_map_format`]) with depth writes enabled
    /// and a small constant + slope-scale bias to counter self-shadowing
    /// acne.
    pub fn new(renderer: &Renderer) -> Self {
        let device = &renderer.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("currawong shadow mesh shader"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("currawong shadow mesh pipeline layout"),
            bind_group_layouts: &[Some(renderer.shadow_cascade_layout())],
            ..Default::default()
        });

        let pos_normal_uv_attrs = PosNormalUv::attributes(0);
        let instance_attrs = MeshInstanceAttribs::vertex_attributes(3);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("currawong shadow mesh pipeline"),
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
            fragment: None,
            primitive: wgpu::PrimitiveState {
                // Front-face culling lifts the depth-test surface to the
                // back of each occluder, which together with the constant
                // bias below kills most self-shadowing acne on convex
                // meshes. Concave geometry needs additional per-receiver
                // bias; PBR shader does slope-scale on the receiver side.
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: renderer.shadow_map_format(),
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self { pipeline }
    }

    /// Underlying [`wgpu::RenderPipeline`]. Bind in
    /// [`View::shadow_pass`](super::View::shadow_pass) before recording
    /// draws.
    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }
}
