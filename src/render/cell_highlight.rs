//! Single-cell highlight overlay — a translucent coloured fan painted over
//! one tile in world space.
//!
//! The canonical "you are hovering over this cell" cursor in tile-grid
//! games (Civ, Age of Wonders, RimWorld). [`TerrainPicker`](super::TerrainPicker)
//! says *which* cell; [`CellHighlight`] paints it.
//!
//! Generic over [`Grid`] at the call site — `set_cell::<SquareGrid>` and
//! `set_cell::<HexGrid>` both work without the highlight knowing or caring
//! about topology; it just asks the grid for the cell centre and corners
//! and triangulates a fan around the centre.
//!
//! A filled fan, not a line outline: wgpu / WebGPU enforces a fixed line
//! width of 1 px, which reads as a faint pencil scratch under any
//! meaningful zoom. A semi-transparent fill is the visual the strategy
//! games above all use, and it stays legible at any camera distance.
//!
//! ## Pipeline
//!
//! - `@group(0)` — camera uniform (same layout as
//!   [`CameraBinding`](super::CameraBinding))
//! - vertex slot 0 — `[Vec3 pos, Vec4 color]` per vertex, `TriangleList`
//!   topology. Each cell is emitted as N triangles fanned around the
//!   centre, one per edge.
//! - Auto-adapts to the View's depth attachment via
//!   [`Renderer::depth_format`](super::Renderer::depth_format): depth-tests
//!   when the renderer has depth (so the overlay hides behind walls),
//!   skips the depth state entirely for 2D / UI views.
//!
//! The fill is *depth-tested* (so it hides behind walls in front of it)
//! but does not *write* depth — it doesn't shadow later transparent
//! passes. The caller controls Z-fight avoidance by passing a `z` slightly
//! above the tile top in [`set_cell`](CellHighlight::set_cell): typically
//! `floor_height * height_unit + small_lift`.

use bytemuck::{Pod, Zeroable};
use glam::{IVec2, Vec4};

use crate::grid::Grid;

use super::renderer::Renderer;

/// Per-vertex layout for the outline mesh.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OutlineVertex {
    pos: [f32; 3],
    color: [f32; 4],
}

const VERTEX_SIZE: u64 = std::mem::size_of::<OutlineVertex>() as u64;
/// Upper bound on overlay vertices. Three per edge (centre + two
/// consecutive corners); hexes have 6 edges → 18 verts. Sized at 32 to
/// keep the buffer write a single short transfer with headroom.
const MAX_VERTICES: u64 = 32;

const SHADER: &str = r#"
struct Camera { view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> camera: Camera;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// Draws a single-cell outline. Hold one alongside your camera + picker,
/// call [`set_cell`](Self::set_cell) whenever the highlighted cell changes,
/// and [`draw`](Self::draw) after your terrain pass.
///
/// State is kept tiny: a 32-vertex GPU buffer and a `vertex_count` cursor.
/// When `vertex_count == 0` the [`draw`](Self::draw) call is a no-op, so
/// clearing the highlight is just `clear()`.
pub struct CellHighlight {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    color: Vec4,
}

impl CellHighlight {
    /// Build the highlight pipeline. `camera_layout` is the camera bind-group
    /// layout the highlight will read from (group 0); `color` is the outline
    /// colour as linear RGBA.
    pub fn new(renderer: &Renderer, camera_layout: &wgpu::BindGroupLayout, color: Vec4) -> Self {
        let device = &renderer.device;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CellHighlight vertices"),
            size: MAX_VERTICES * VERTEX_SIZE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CellHighlight shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CellHighlight pipeline layout"),
            bind_group_layouts: &[Some(camera_layout)],
            ..Default::default()
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CellHighlight pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: VERTEX_SIZE,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: renderer.surface_format(),
                        // Alpha blend so semi-transparent outlines layer
                        // cleanly over the terrain — most callers will want
                        // full alpha, but the door is open for 50% selection
                        // ghosting.
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Opt out of the hit-ID attachment (#56 PR 1): overlays
                    // are decorative and shouldn't pollute the ID buffer
                    // with their (alpha-blended, semantically meaningless)
                    // IDs over the terrain underneath.
                    renderer.id_target_opt_out(),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // No backface culling — for a flat overlay both windings
                // are valid depending on camera elevation, and the fans we
                // emit don't have a consistent winding for the hex case
                // versus square. Cheap to draw both faces.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: renderer
                .depth_format()
                .map(|format| wgpu::DepthStencilState {
                    format,
                    // Depth-test on so terrain in front of the highlight
                    // occludes it correctly. Depth-write off because the
                    // outline is an overlay — later transparent passes
                    // shouldn't see a depth value at the outline's Z.
                    depth_write_enabled: Some(false),
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
            vertex_buffer,
            vertex_count: 0,
            color,
        }
    }

    /// Re-upload the outline to trace `cell` on `grid`. The outline sits in
    /// the horizontal plane `z = z` and scales by `tile_size` (matching the
    /// mesher's tile scaling). Replaces whatever cell was last set.
    ///
    /// `tile_size` is the same number the terrain mesher uses to scale
    /// canonical grid units into world units — pass the mesher's
    /// `tile_size` field verbatim so the outline aligns with the rendered
    /// tile.
    pub fn set_cell<G: Grid>(
        &mut self,
        renderer: &Renderer,
        grid: &G,
        cell: IVec2,
        z: f32,
        tile_size: f32,
    ) {
        // Fan around the cell centre: for each edge i, emit triangle
        // (centre, corner_i, corner_{i+1}). N triangles = 3*N vertices.
        let mut verts: [OutlineVertex; MAX_VERTICES as usize] =
            [OutlineVertex::zeroed(); MAX_VERTICES as usize];
        let color = self.color.to_array();
        let n = G::CORNERS_PER_CELL;
        let centre = grid.cell_center(cell) * tile_size;
        let centre_v = OutlineVertex {
            pos: [centre.x, centre.y, z],
            color,
        };
        let mut count = 0usize;
        for i in 0..n {
            let a = grid.corner_xy(cell, i) * tile_size;
            let b = grid.corner_xy(cell, (i + 1) % n) * tile_size;
            verts[count] = centre_v;
            verts[count + 1] = OutlineVertex {
                pos: [a.x, a.y, z],
                color,
            };
            verts[count + 2] = OutlineVertex {
                pos: [b.x, b.y, z],
                color,
            };
            count += 3;
        }
        let written = &verts[..count];
        renderer
            .queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(written));
        self.vertex_count = count as u32;
    }

    /// Hide the outline. Subsequent [`draw`](Self::draw) calls are no-ops
    /// until [`set_cell`](Self::set_cell) re-populates the buffer.
    pub fn clear(&mut self) {
        self.vertex_count = 0;
    }

    /// Change the outline colour for the next [`set_cell`](Self::set_cell).
    /// Doesn't affect already-uploaded geometry; call `set_cell` again if
    /// the highlight is currently visible.
    pub fn set_color(&mut self, color: Vec4) {
        self.color = color;
    }

    /// Record the draw. Caller must have bound the camera bind group at
    /// group 0 already (the engine's standard
    /// [`CameraBinding`](super::CameraBinding) layout). No-op when nothing
    /// is set.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.vertex_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..self.vertex_count, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use crate::grid::{HexGrid, SquareGrid};

    use super::*;

    /// Bound check: even a hex (the widest grid) writes within the buffer's
    /// vertex cap. Touches no GPU resources — exercises just the local
    /// count math.
    #[test]
    fn fan_vertex_count_fits_buffer() {
        let square_tris = SquareGrid::CORNERS_PER_CELL;
        let hex_tris = HexGrid::CORNERS_PER_CELL;
        assert!(3 * square_tris as u64 <= MAX_VERTICES);
        assert!(3 * hex_tris as u64 <= MAX_VERTICES);
    }

    #[test]
    fn outline_vertex_struct_is_pod_size() {
        // 3 floats position + 4 floats color = 28 bytes. Pin the size so a
        // mis-edit to the struct doesn't silently break the vertex layout.
        assert_eq!(std::mem::size_of::<OutlineVertex>(), 28);
    }
}
