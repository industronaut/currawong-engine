//! Quad-expanded "fat lines" — screen-space-thick line segments rendered
//! as triangles, since WebGPU / wgpu have no `lineWidth > 1.0` knob.
//!
//! Each input line segment is expanded in the vertex shader into a quad
//! aligned with the segment's screen-space direction and `width_px` pixels
//! tall. Every vertex carries both segment endpoints (`pos_a`, `pos_b`) plus
//! a `side` flag (±1, which side of the segment to offset to) and an
//! `endpoint` flag (0 or 1, which endpoint this vertex sits at). The
//! material uniform supplies `width_px` and the current viewport size in
//! pixels.
//!
//! ## Limitations (deliberate, fixable when needed)
//!
//! - **Butt caps, no miter join.** Adjacent segments meet at perpendicular
//!   stubs with a small overlap diamond. Invisible at 1–4 px widths,
//!   noticeable past ~8 px. Miter would require each vertex to know its
//!   neighbour segments (a longer vertex layout) and a length-cap on the
//!   miter to handle acute angles — not worth it until a consumer needs it.
//! - **Width is per material instance, not per vertex.** "Variable width
//!   along a polyline" is one more `f32` per vertex; the shader plumbing is
//!   ready for it (it already does the screen-space math per vertex).
//! - **No dashed / dotted styles.** Add `length_along_segment` per vertex
//!   and a `dash_period_px` uniform when needed.
//!
//! For 1-pixel debug gizmos the simpler [`LineMaterial`](super::LineMaterial)
//! is cheaper — no per-segment quad expansion — and remains the right tool
//! for dense overlays like terrain grids.
//!
//! ## Wiring
//!
//! ```text
//! init:   let camera = CameraBinding::new(&renderer.device);
//!         let material = FatLineMaterial::new(&renderer, camera.layout());
//!         let yellow = material.create_instance(&renderer, FatLineMaterialParams {
//!             base_color: Vec4::new(1.0, 1.0, 0.0, 1.0),
//!             width_px: 2.0,
//!         });
//! frame:  yellow.write_viewport(&renderer.queue, viewport_size);   // each frame
//!         pass.set_pipeline(material.pipeline());
//!         pass.set_bind_group(0, camera.bind_group(), &[]);
//!         pass.set_bind_group(1, yellow.bind_group(), &[]);
//!         pass.set_vertex_buffer(0, fat_line_vertices.slice(..));
//!         pass.set_vertex_buffer(1, instance_buf.slice(..));
//!         pass.set_index_buffer(fat_line_indices.slice(..), wgpu::IndexFormat::Uint16);
//!         pass.draw_indexed(0..index_count, 0, 0..1);
//! ```

use bytemuck::{Pod, Zeroable};
use glam::{UVec2, Vec4};

use super::material::{MeshInstanceAttribs, MeshMaterial};
use super::renderer::Renderer;

const FAT_LINE_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct Material {
    base_color: vec4<f32>,
    viewport_size: vec2<f32>,
    width_px: f32,
    _pad: f32,
};
@group(1) @binding(0) var<uniform> material: Material;

struct VsIn {
    @location(0) pos_a: vec3<f32>,
    @location(1) pos_b: vec3<f32>,
    // -1 or +1: which side of the segment this vertex offsets to.
    @location(2) side: f32,
    // 0 or 1: which endpoint of the segment this vertex sits at.
    @location(3) endpoint: f32,
    // Per-instance model matrix as four vec4 columns.
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) tint: vec4<f32>,
    // Per-instance hit_id is declared so the MeshInstanceAttribs Pod layout
    // matches, but lines aren't pickable — the fragment shader writes the
    // no-hit sentinel unconditionally.
    @location(9) hit_id: u32,
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
    let clip_a = camera.view_proj * model * vec4<f32>(in.pos_a, 1.0);
    let clip_b = camera.view_proj * model * vec4<f32>(in.pos_b, 1.0);

    // Endpoint this vertex represents. Mix in clip space then undo the
    // perspective divide locally — applying the offset after the divide
    // would warp the segment near the near plane.
    let clip = mix(clip_a, clip_b, in.endpoint);

    // NDC.xy for both endpoints; convert to pixel-space direction so the
    // perpendicular has true screen orientation independent of aspect.
    let ndc_a = clip_a.xy / clip_a.w;
    let ndc_b = clip_b.xy / clip_b.w;
    let half_viewport = material.viewport_size * 0.5;
    let dir_px = (ndc_b - ndc_a) * half_viewport;

    // Degenerate (zero-length or view-aligned) segments collapse to a line
    // of zero thickness — the offset goes to zero and the quad pinches
    // shut. Acceptable: prevents NaN from `normalize(vec2(0.0))`.
    var perp_px = vec2<f32>(0.0, 0.0);
    let len = length(dir_px);
    if (len > 1e-5) {
        let dir = dir_px / len;
        perp_px = vec2<f32>(-dir.y, dir.x);
    }

    // Offset in pixels, converted back to NDC and pre-multiplied by clip.w
    // so the post-divide displacement lands at the requested pixel count
    // regardless of depth.
    let offset_px = perp_px * in.side * material.width_px * 0.5;
    let offset_ndc = offset_px / half_viewport;
    var out_clip = clip;
    out_clip.x += offset_ndc.x * clip.w;
    out_clip.y += offset_ndc.y * clip.w;

    var out: VsOut;
    out.clip = out_clip;
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

/// Per-vertex layout for fat lines. Each segment uses 4 vertices (two
/// endpoints × two sides) and 6 indices (one quad).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct FatLineVertex {
    pub pos_a: [f32; 3],
    pub pos_b: [f32; 3],
    /// `-1.0` or `+1.0` — which side of the segment this vertex offsets to.
    pub side: f32,
    /// `0.0` or `1.0` — which endpoint of the segment this vertex sits at.
    pub endpoint: f32,
}

impl FatLineVertex {
    pub const STRIDE: u64 = std::mem::size_of::<Self>() as u64;

    pub const fn attributes(start_location: u32) -> [wgpu::VertexAttribute; 4] {
        [
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: start_location,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: start_location + 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: start_location + 2,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: 28,
                shader_location: start_location + 3,
                format: wgpu::VertexFormat::Float32,
            },
        ]
    }
}

/// Construction parameters for a [`FatLineMaterialInstance`].
#[derive(Clone, Copy, Debug)]
pub struct FatLineMaterialParams {
    /// Linear RGBA in `[0, 1]`; multiplied with the per-instance tint.
    pub base_color: Vec4,
    /// Screen-space line width in pixels.
    pub width_px: f32,
}

/// Layout matches the WGSL `Material` struct (28 B + 4 B padding → 32 B,
/// rounded to the struct's `vec4` alignment). `viewport_size` is rewritten
/// per frame; `base_color` and `width_px` are stable per instance.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FatLineUniform {
    base_color: [f32; 4],
    viewport_size: [f32; 2],
    width_px: f32,
    _pad: f32,
}

/// Template for the fat-line material: pipeline + per-instance bind-group
/// layout. Pipeline expects:
///
/// - `@group(0)` — camera uniform
/// - `@group(1)` — material uniform (base color, viewport size, width)
/// - vertex buffer slot 0 — [`FatLineVertex`] per vertex
/// - vertex buffer slot 1 — [`MeshInstanceAttribs`] per instance
/// - index buffer — `Uint16` (see [`unit_cube_fat_line_geometry`])
pub struct FatLineMaterial {
    pipeline: wgpu::RenderPipeline,
    instance_bgl: wgpu::BindGroupLayout,
}

impl FatLineMaterial {
    pub fn new(renderer: &Renderer, camera_layout: &wgpu::BindGroupLayout) -> Self {
        let device = &renderer.device;

        let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FatLine instance bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                // Vertex stage reads viewport_size + width_px; fragment reads
                // base_color. Both need the binding.
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FatLine shader"),
            source: wgpu::ShaderSource::Wgsl(FAT_LINE_SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FatLine pipeline layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&instance_bgl)],
            ..Default::default()
        });

        let vertex_attrs = FatLineVertex::attributes(0);
        let instance_attrs = MeshInstanceAttribs::vertex_attributes(4);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FatLine pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: FatLineVertex::STRIDE,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
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
            // Quads — back-face culling off because the perpendicular offset
            // can flip winding when a segment turns away from the camera.
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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

    pub fn create_instance(
        &self,
        renderer: &Renderer,
        params: FatLineMaterialParams,
    ) -> FatLineMaterialInstance {
        let data = FatLineUniform {
            base_color: params.base_color.to_array(),
            // Placeholder; the View calls `write_viewport` every frame.
            viewport_size: [1.0, 1.0],
            width_px: params.width_px,
            _pad: 0.0,
        };
        let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FatLine instance uniform"),
            size: std::mem::size_of::<FatLineUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        renderer
            .queue
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&data));
        let bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("FatLine instance bind group"),
                layout: &self.instance_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
        FatLineMaterialInstance { buffer, bind_group }
    }
}

impl MeshMaterial for FatLineMaterial {
    type Instance = FatLineMaterialInstance;

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        self.pipeline()
    }
}

/// Live instance for [`FatLineMaterial`]. The viewport size must be
/// rewritten via [`Self::write_viewport`] each frame (or whenever the
/// surface resizes) so the screen-space perpendicular lands at the right
/// pixel count.
pub struct FatLineMaterialInstance {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl FatLineMaterialInstance {
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Re-upload the viewport size (in pixels) without rebuilding the bind
    /// group. Touches the two `f32`s at offset 16 — the GPU sees the change
    /// on the next draw.
    pub fn write_viewport(&self, queue: &wgpu::Queue, size: UVec2) {
        let viewport = [size.x as f32, size.y as f32];
        queue.write_buffer(&self.buffer, 16, bytemuck::bytes_of(&viewport));
    }

    pub fn write_base_color(&self, queue: &wgpu::Queue, base_color: Vec4) {
        let data = base_color.to_array();
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&data));
    }

    pub fn write_width_px(&self, queue: &wgpu::Queue, width_px: f32) {
        queue.write_buffer(&self.buffer, 24, bytemuck::bytes_of(&width_px));
    }
}

/// Fat-line vertex + index data for a unit cube spanning `[-0.5, 0.5]³`
/// (12 edges → 48 vertices, 72 triangle-list indices). Pair with a
/// [`FatLineMaterial`] instance and a model matrix that scales /
/// translates the unit cube onto any AABB — same convention as
/// [`unit_cube_line_geometry`](super::unit_cube_line_geometry).
pub fn unit_cube_fat_line_geometry() -> (Vec<FatLineVertex>, Vec<u16>) {
    let h = 0.5;
    let corners: [[f32; 3]; 8] = [
        [-h, -h, -h], // 0
        [h, -h, -h],  // 1
        [h, h, -h],   // 2
        [-h, h, -h],  // 3
        [-h, -h, h],  // 4
        [h, -h, h],   // 5
        [h, h, h],    // 6
        [-h, h, h],   // 7
    ];
    // 4 bottom edges, 4 top edges, 4 vertical edges.
    let edges: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    let mut vertices = Vec::with_capacity(48);
    let mut indices = Vec::with_capacity(72);
    for (a, b) in edges {
        let pos_a = corners[a];
        let pos_b = corners[b];
        let base = vertices.len() as u16;
        // 4 verts per segment: (endpoint, side) in [(0,-1), (0,+1), (1,-1), (1,+1)].
        for &(endpoint, side) in &[(0.0_f32, -1.0_f32), (0.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            vertices.push(FatLineVertex {
                pos_a,
                pos_b,
                side,
                endpoint,
            });
        }
        // Two CCW triangles when looking down the segment from a→b: the
        // shader pushes side=-1 to the left and side=+1 to the right of
        // the screen-space direction.
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }
    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_stride_is_32() {
        assert_eq!(std::mem::size_of::<FatLineVertex>(), 32);
        assert_eq!(FatLineVertex::STRIDE, 32);
    }

    #[test]
    fn uniform_size_matches_wgsl_layout() {
        // WGSL `struct Material { vec4 base_color; vec2 viewport_size; f32
        // width_px; }` rounds up to the struct's vec4 alignment → 32 bytes.
        // If this assertion fails, `write_viewport` and `write_width_px`'s
        // hard-coded offsets are out of sync.
        assert_eq!(std::mem::size_of::<FatLineUniform>(), 32);
        // Offsets the per-field write methods assume.
        let u = FatLineUniform {
            base_color: [0.0; 4],
            viewport_size: [0.0; 2],
            width_px: 0.0,
            _pad: 0.0,
        };
        let base = &u as *const _ as usize;
        let viewport_offset = &u.viewport_size as *const _ as usize - base;
        let width_offset = &u.width_px as *const _ as usize - base;
        assert_eq!(viewport_offset, 16);
        assert_eq!(width_offset, 24);
    }

    #[test]
    fn unit_cube_has_12_segments() {
        let (vertices, indices) = unit_cube_fat_line_geometry();
        // 12 segments × 4 vertices each.
        assert_eq!(vertices.len(), 48);
        // 12 segments × 6 indices (one quad) each.
        assert_eq!(indices.len(), 72);
        for i in &indices {
            assert!((*i as usize) < vertices.len());
        }
    }

    #[test]
    fn unit_cube_segments_have_unit_length() {
        let (vertices, _) = unit_cube_fat_line_geometry();
        for chunk in vertices.chunks_exact(4) {
            // All 4 verts in a segment share the same endpoints.
            for v in chunk {
                assert_eq!(v.pos_a, chunk[0].pos_a);
                assert_eq!(v.pos_b, chunk[0].pos_b);
            }
            let dx = chunk[0].pos_b[0] - chunk[0].pos_a[0];
            let dy = chunk[0].pos_b[1] - chunk[0].pos_a[1];
            let dz = chunk[0].pos_b[2] - chunk[0].pos_a[2];
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "segment length {len} != 1.0 (a={:?}, b={:?})",
                chunk[0].pos_a,
                chunk[0].pos_b,
            );
        }
    }

    #[test]
    fn unit_cube_segment_sides_and_endpoints_cover_quad() {
        let (vertices, _) = unit_cube_fat_line_geometry();
        for chunk in vertices.chunks_exact(4) {
            // Expect the canonical (endpoint, side) sequence used by the
            // index layout. If this rotates, the triangle indices need to
            // rotate with it.
            let actual: Vec<(f32, f32)> = chunk.iter().map(|v| (v.endpoint, v.side)).collect();
            assert_eq!(
                actual,
                vec![(0.0, -1.0), (0.0, 1.0), (1.0, -1.0), (1.0, 1.0)],
            );
        }
    }
}
