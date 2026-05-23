//! Unlit line-list material — the debug-gizmo counterpart to
//! [`UnlitColoredMaterial`](super::UnlitColoredMaterial).
//!
//! Same template / instance / per-instance-attrib shape as the unlit
//! triangle material; the only structural difference is the pipeline's
//! [`PrimitiveTopology::LineList`](wgpu::PrimitiveTopology::LineList). Useful
//! for bounding-box overlays, axis gizmos, and other editor / debug
//! visualisations.
//!
//! Lines are not pickable, so the fragment shader writes the no-hit sentinel
//! (`0`) to the engine's R32Uint hit-ID attachment — matching the attachment's
//! clear value, so the line draws don't disturb mesh picking.
//!
//! ## Wiring
//!
//! ```text
//! init:   let camera   = CameraBinding::new(&renderer.device);
//!         let material = LineMaterial::new(&renderer, camera.layout());
//!         let yellow   = material.create_instance(&renderer, Vec4::new(1.0, 1.0, 0.0, 1.0));
//! draw:   pass.set_pipeline(material.pipeline());
//!         pass.set_bind_group(0, camera.bind_group(), &[]);
//!         pass.set_bind_group(1, yellow.bind_group(), &[]);
//!         pass.set_vertex_buffer(0, line_vertices.slice(..));  // pos: vec3
//!         pass.set_vertex_buffer(1, instance_buf.slice(..));   // MeshInstanceAttribs
//!         pass.set_index_buffer(line_indices.slice(..), wgpu::IndexFormat::Uint16);
//!         pass.draw_indexed(0..index_count, 0, 0..instance_count);
//! ```

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};

use super::material::{MeshInstanceAttribs, MeshMaterial};
use super::renderer::Renderer;

const LINE_SHADER: &str = r#"
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
    @location(1) m0: vec4<f32>,
    @location(2) m1: vec4<f32>,
    @location(3) m2: vec4<f32>,
    @location(4) m3: vec4<f32>,
    @location(5) tint: vec4<f32>,
    // hit_id is present in MeshInstanceAttribs but unused — lines aren't
    // pickable. Declared so the vertex layout still matches the Pod, even
    // though the fragment shader writes the no-hit sentinel unconditionally.
    @location(6) hit_id: u32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) tint: vec4<f32>,
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
    out.clip = camera.view_proj * world;
    out.tint = in.tint;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    var out: FsOut;
    out.color  = material.base_color * in.tint;
    out.hit_id = 0u;
    return out;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LineUniform {
    base_color: [f32; 4],
}

/// Template for the line material: pipeline + per-instance bind-group layout.
///
/// Pipeline expects:
/// - `@group(0)` — camera uniform (use a
///   [`CameraBinding`](super::CameraBinding))
/// - `@group(1)` — material uniform (`base_color: vec4<f32>`)
/// - vertex buffer slot 0 — `position: vec3<f32>` per vertex
/// - vertex buffer slot 1 — [`MeshInstanceAttribs`] per instance
/// - index buffer — `Uint16` (see [`unit_cube_line_geometry`])
pub struct LineMaterial {
    pipeline: wgpu::RenderPipeline,
    instance_bgl: wgpu::BindGroupLayout,
}

impl LineMaterial {
    pub fn new(renderer: &Renderer, camera_layout: &wgpu::BindGroupLayout) -> Self {
        let device = &renderer.device;

        let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Line instance bgl"),
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
            label: Some("Line shader"),
            source: wgpu::ShaderSource::Wgsl(LINE_SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Line pipeline layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&instance_bgl)],
            ..Default::default()
        });

        let instance_attrs = MeshInstanceAttribs::vertex_attributes(1);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        }],
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
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
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

    /// Create a material instance with the given `base_color` (linear RGBA
    /// in `[0, 1]`). The instance owns its own uniform buffer and bind group.
    pub fn create_instance(&self, renderer: &Renderer, base_color: Vec4) -> LineMaterialInstance {
        let data = LineUniform {
            base_color: base_color.to_array(),
        };
        let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Line instance uniform"),
            size: std::mem::size_of::<LineUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        renderer
            .queue
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&data));
        let bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Line instance bind group"),
                layout: &self.instance_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
        LineMaterialInstance { buffer, bind_group }
    }
}

impl MeshMaterial for LineMaterial {
    type Instance = LineMaterialInstance;

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        self.pipeline()
    }
}

/// A live material instance for [`LineMaterial`]. Bind as `@group(1)` when
/// drawing through the material's pipeline.
pub struct LineMaterialInstance {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl LineMaterialInstance {
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Re-upload the base colour without rebuilding the bind group.
    pub fn write_base_color(&self, queue: &wgpu::Queue, base_color: Vec4) {
        let data = LineUniform {
            base_color: base_color.to_array(),
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&data));
    }
}

/// Position-only vertices and `LineList` indices for a unit cube spanning
/// `[-0.5, 0.5]³` (12 edges → 24 indices). Pair with a [`LineMaterial`] and
/// a model matrix that scales + translates the unit cube to world space:
///
/// ```text
/// let scale = aabb.max - aabb.min;
/// let model = Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, aabb.center());
/// ```
pub fn unit_cube_line_geometry() -> ([Vec3; 8], [u16; 24]) {
    let h = 0.5;
    let vertices = [
        Vec3::new(-h, -h, -h), // 0: ---
        Vec3::new(h, -h, -h),  // 1: +--
        Vec3::new(h, h, -h),   // 2: ++-
        Vec3::new(-h, h, -h),  // 3: -+-
        Vec3::new(-h, -h, h),  // 4: --+
        Vec3::new(h, -h, h),   // 5: +-+
        Vec3::new(h, h, h),    // 6: +++
        Vec3::new(-h, h, h),   // 7: -++
    ];
    // 4 bottom edges, 4 top edges, 4 vertical edges.
    let indices = [
        0, 1, 1, 2, 2, 3, 3, 0, // bottom (z = -h)
        4, 5, 5, 6, 6, 7, 7, 4, // top    (z = +h)
        0, 4, 1, 5, 2, 6, 3, 7, // verticals
    ];
    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_cube_geometry_has_12_edges() {
        let (vertices, indices) = unit_cube_line_geometry();
        assert_eq!(vertices.len(), 8);
        // 12 edges × 2 indices = 24.
        assert_eq!(indices.len(), 24);
        // Every index must reference a valid vertex.
        for i in indices {
            assert!((i as usize) < vertices.len());
        }
    }

    #[test]
    fn unit_cube_corners_span_pm_half() {
        let (vertices, _) = unit_cube_line_geometry();
        let mut min = vertices[0];
        let mut max = vertices[0];
        for v in &vertices[1..] {
            min = min.min(*v);
            max = max.max(*v);
        }
        assert_eq!(min, Vec3::splat(-0.5));
        assert_eq!(max, Vec3::splat(0.5));
    }

    #[test]
    fn unit_cube_edges_have_unit_length() {
        // Every edge of a [-0.5, 0.5]³ cube has length exactly 1.0, which
        // catches accidental diagonal indices (a common copy-paste error).
        let (vertices, indices) = unit_cube_line_geometry();
        for pair in indices.chunks_exact(2) {
            let a = vertices[pair[0] as usize];
            let b = vertices[pair[1] as usize];
            let len = (a - b).length();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "edge {pair:?} has length {len}, expected 1.0",
            );
        }
    }
}
