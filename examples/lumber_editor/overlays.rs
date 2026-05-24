//! Data-driven fat-line overlays for kind metadata. All three share the
//! same [`FatLineMaterial`] pipeline (constant screen-space pixel width
//! regardless of camera distance) and the same one-static-geometry,
//! one-per-frame-instance-buffer packaging:
//!
//! - **Bounds** — yellow wireframe of the selected kind's visual AABB.
//!   Single instance, rewritten on selection change.
//! - **Interaction tiles** — green outlined squares on the ground, one
//!   per tile from the kind's `Interaction`.
//! - **Footprint tiles** — orange outlined squares with diagonals
//!   (visually distinct from interaction tiles), one per tile from the
//!   kind's `Footprint`.
//!
//! The single-instance arrow and the floor live in [`crate::scene`]
//! because they're "establishing" geometry; these three are per-kind data
//! visualisations.

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Aabb, FatLineMaterial, FatLineMaterialInstance, FatLineMaterialParams, FatLineVertex,
    MeshInstanceAttribs, Renderer, unit_cube_fat_line_geometry, wgpu,
};

// --- Bounds overlay ----------------------------------------------------

/// Yellow used for the bounding-box wireframe. Saturated primary so the
/// box reads against both the lit kind body and the muted ground checker.
const BOUNDS_COLOR: Vec4 = Vec4::new(1.0, 1.0, 0.0, 1.0);
/// Screen-space line width in pixels. Thick enough to read clearly at the
/// editor's typical orbit distance without dominating the figure.
const BOUNDS_WIDTH_PX: f32 = 2.5;

/// GPU resources for the bounding-box overlay. Uses [`FatLineMaterial`] for
/// a constant pixel width regardless of camera distance — wgpu has no
/// `lineWidth` knob, so each segment is expanded to a screen-space quad in
/// the vertex shader. Unit-cube vertex/index buffers are static; the
/// single-instance buffer holds a model matrix that maps `[-0.5, 0.5]³` to
/// the active kind's visual AABB and is rewritten every frame in `render`.
pub(crate) struct BoundsOverlay {
    pub(crate) material: FatLineMaterial,
    pub(crate) color: FatLineMaterialInstance,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) index_count: u32,
    pub(crate) instance_buffer: wgpu::Buffer,
}

pub(crate) fn build_bounds_overlay(
    renderer: &Renderer,
    camera_layout: &wgpu::BindGroupLayout,
) -> BoundsOverlay {
    use wgpu::util::DeviceExt;

    let material = FatLineMaterial::new(renderer, camera_layout);
    let color = material.create_instance(
        renderer,
        FatLineMaterialParams {
            base_color: BOUNDS_COLOR,
            width_px: BOUNDS_WIDTH_PX,
        },
    );

    let (vertices, indices) = unit_cube_fat_line_geometry();
    let vertex_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor bounds vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let index_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor bounds indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    // Placeholder instance — rewritten every frame in `render` from the
    // active kind's AABB. Pre-sized to MeshInstanceAttribs so the buffer
    // never has to grow.
    let placeholder = MeshInstanceAttribs::new(Mat4::IDENTITY, Vec4::ONE);
    let instance_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor bounds instance"),
            contents: bytemuck::bytes_of(&placeholder),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

    BoundsOverlay {
        material,
        color,
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        instance_buffer,
    }
}

/// Rewrite the bounds-overlay instance buffer so the unit cube maps to
/// `aabb`. Scales `[-0.5, 0.5]³` by `(max - min)` and translates to the
/// AABB centre.
pub(crate) fn write_bounds_instance(queue: &wgpu::Queue, buffer: &wgpu::Buffer, aabb: Aabb) {
    let scale = aabb.max - aabb.min;
    let model = Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, aabb.center());
    let attribs = MeshInstanceAttribs::new(model, Vec4::ONE);
    queue.write_buffer(buffer, 0, bytemuck::bytes_of(&attribs));
}

// --- Interaction-tiles overlay ----------------------------------------

/// Saturated lime green for the interaction-tile outlines. Reads cleanly
/// against the muted ground checker and stays clear of the yellow bounds
/// wireframe so the two overlays can be parsed at a glance.
const INTERACTION_TILE_COLOR: Vec4 = Vec4::new(0.20, 0.95, 0.35, 1.0);
/// Screen-space stroke width for the outline in pixels. Picked to match
/// the user-visible weight requested in the editor — heavy enough to read
/// as a deliberate marker, light enough not to dominate small kinds.
const INTERACTION_TILE_WIDTH_PX: f32 = 8.0;
/// Vertical lift above the ground plane to avoid z-fighting with the
/// checker. Small enough to be visually flush, large enough that depth
/// quantization at the far end of the orbit-rig view never collapses it
/// back to the ground.
const INTERACTION_TILE_Z_EPSILON: f32 = 0.01;
/// Cap on the per-frame instance count. `Surround { radius_tiles: 5 }`
/// yields 120 tiles; 256 leaves comfortable headroom for the editor's
/// single-subject scope without ever growing the buffer mid-frame.
const MAX_INTERACTION_TILES: u32 = 256;

/// GPU resources for the interaction-tiles overlay. Four-edge fat-line
/// square in the XY plane shared across kinds (constant 8 px screen-space
/// stroke), with a per-frame instance buffer pre-sized to
/// [`MAX_INTERACTION_TILES`] and rewritten each frame from the selected
/// kind's [`currawong::Interaction::tiles`]. Draws zero instances when the
/// selection has [`currawong::Interaction::None`].
pub(crate) struct InteractionTilesOverlay {
    pub(crate) material: FatLineMaterial,
    pub(crate) color: FatLineMaterialInstance,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) index_count: u32,
    pub(crate) instance_buffer: wgpu::Buffer,
    /// Scratch buffer for assembling per-tile instance attribs before the
    /// single `write_buffer` upload. Kept on the struct so the allocation
    /// is reused across frames.
    instance_scratch: Vec<MeshInstanceAttribs>,
}

/// Fat-line vertex + index data for a unit square outline spanning
/// `[-0.5, 0.5]² × {0}` in XY. Four edges → 16 verts, 24 triangle-list
/// indices. Sibling to [`unit_cube_fat_line_geometry`] — same packing
/// convention so the same [`FatLineMaterial`] pipeline draws both.
fn unit_square_fat_line_geometry() -> (Vec<FatLineVertex>, Vec<u16>) {
    let h = 0.5;
    let corners: [[f32; 3]; 4] = [
        [-h, -h, 0.0], // 0: -X -Y
        [h, -h, 0.0],  // 1: +X -Y
        [h, h, 0.0],   // 2: +X +Y
        [-h, h, 0.0],  // 3: -X +Y
    ];
    // CCW when viewed from +Z (the orbit-rig camera looks down at the
    // ground), matching the pipeline's `FrontFace::Ccw`.
    let edges: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

    let mut vertices = Vec::with_capacity(16);
    let mut indices = Vec::with_capacity(24);
    for (a, b) in edges {
        let pos_a = corners[a];
        let pos_b = corners[b];
        let base = vertices.len() as u16;
        for &(endpoint, side) in &[(0.0_f32, -1.0_f32), (0.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            vertices.push(FatLineVertex {
                pos_a,
                pos_b,
                side,
                endpoint,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }
    (vertices, indices)
}

pub(crate) fn build_interaction_overlay(
    renderer: &Renderer,
    camera_layout: &wgpu::BindGroupLayout,
) -> InteractionTilesOverlay {
    use wgpu::util::DeviceExt;

    let material = FatLineMaterial::new(renderer, camera_layout);
    let color = material.create_instance(
        renderer,
        FatLineMaterialParams {
            base_color: INTERACTION_TILE_COLOR,
            width_px: INTERACTION_TILE_WIDTH_PX,
        },
    );

    let (vertices, indices) = unit_square_fat_line_geometry();
    let vertex_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor interaction-tile vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let index_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor interaction-tile indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    let instance_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lumber-editor interaction-tile instances"),
        size: (MAX_INTERACTION_TILES as u64) * std::mem::size_of::<MeshInstanceAttribs>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    InteractionTilesOverlay {
        material,
        color,
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        instance_buffer,
        instance_scratch: Vec::with_capacity(MAX_INTERACTION_TILES as usize),
    }
}

impl InteractionTilesOverlay {
    /// Rebuild the per-instance buffer from `tiles` (integer world tile
    /// coords from [`currawong::Interaction::tiles`]). Returns the instance
    /// count to draw — clamped to [`MAX_INTERACTION_TILES`] so a future
    /// radius-10 surround can't overflow the buffer (it would just clip
    /// silently; fine for an editor preview, the alternative is a panic
    /// mid-render).
    ///
    /// Each square is centred on the *integer* world coord `(tx, ty)` — not
    /// `(tx+0.5, ty+0.5)`. The kind body is drawn at
    /// `transform.position.to_vec3()` (which is the tile-corner value), so
    /// the editor treats the body as "occupying the cell centred on its
    /// origin", and the surrounding interaction squares ring it cleanly at
    /// `(±1, 0)`, `(0, ±1)`, etc. The `+0.5` tile-corner→tile-centre offset
    /// is therefore the wrong convention for this visualisation.
    pub(crate) fn refresh(&mut self, queue: &wgpu::Queue, tiles: &[(i32, i32, i32)]) -> u32 {
        self.instance_scratch.clear();
        let cap = MAX_INTERACTION_TILES as usize;
        for &(tx, ty, tz) in tiles.iter().take(cap) {
            let translation =
                Vec3::new(tx as f32, ty as f32, tz as f32 + INTERACTION_TILE_Z_EPSILON);
            let model =
                Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, translation);
            self.instance_scratch
                .push(MeshInstanceAttribs::new(model, Vec4::ONE));
        }
        let count = self.instance_scratch.len() as u32;
        if count > 0 {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instance_scratch),
            );
        }
        count
    }
}

// --- Footprint-tiles overlay ------------------------------------------

/// Saturated orange for the placement tile markers. Sits clearly between
/// the green interaction tiles and the yellow bounds wireframe so all
/// three overlays can be enabled simultaneously and still parsed at a
/// glance.
const FOOTPRINT_TILE_COLOR: Vec4 = Vec4::new(1.0, 0.55, 0.10, 1.0);
/// Screen-space stroke width for the outline + diagonals in pixels.
/// Matches the interaction-tile weight so neighbouring squares read as
/// the same class of marker.
const FOOTPRINT_TILE_WIDTH_PX: f32 = 8.0;
/// Vertical lift above the ground plane to avoid z-fighting. Slightly
/// above the interaction-tile epsilon so the two overlays don't fight
/// each other when they happen to land on the same cell.
const FOOTPRINT_TILE_Z_EPSILON: f32 = 0.015;
/// Cap on the per-frame instance count. A footprint of "thousands of
/// tiles" doesn't make sense; 256 is generous for any plausibly-authored
/// kind and matches the interaction overlay's cap.
const MAX_FOOTPRINT_TILES: u32 = 256;

/// GPU resources for the placement-tiles overlay. Identical packaging to
/// [`InteractionTilesOverlay`] — fat-line pipeline, per-frame instance
/// buffer pre-sized to [`MAX_FOOTPRINT_TILES`] — but the static
/// vertex/index data is a unit square with two diagonals (six fat-line
/// segments per tile, drawn in orange).
pub(crate) struct FootprintTilesOverlay {
    pub(crate) material: FatLineMaterial,
    pub(crate) color: FatLineMaterialInstance,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) index_count: u32,
    pub(crate) instance_buffer: wgpu::Buffer,
    instance_scratch: Vec<MeshInstanceAttribs>,
}

/// Fat-line vertex + index data for a unit square *with two diagonals*
/// spanning `[-0.5, 0.5]² × {0}` in XY. Six segments (4 edges + 2
/// diagonals) → 24 verts, 36 triangle-list indices. Sibling to
/// [`unit_square_fat_line_geometry`] above; the diagonals are what
/// visually distinguishes a placement tile from a (plain-outline)
/// interaction tile.
fn unit_square_with_diagonals_fat_line_geometry() -> (Vec<FatLineVertex>, Vec<u16>) {
    let h = 0.5;
    let corners: [[f32; 3]; 4] = [
        [-h, -h, 0.0], // 0: -X -Y
        [h, -h, 0.0],  // 1: +X -Y
        [h, h, 0.0],   // 2: +X +Y
        [-h, h, 0.0],  // 3: -X +Y
    ];
    // 4 perimeter edges + 2 diagonals forming an X across the square.
    // CCW perimeter orientation matches the pipeline's `FrontFace::Ccw`.
    let segments: [(usize, usize); 6] = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (a, b) in segments {
        let pos_a = corners[a];
        let pos_b = corners[b];
        let base = vertices.len() as u16;
        for &(endpoint, side) in &[(0.0_f32, -1.0_f32), (0.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            vertices.push(FatLineVertex {
                pos_a,
                pos_b,
                side,
                endpoint,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }
    (vertices, indices)
}

pub(crate) fn build_footprint_overlay(
    renderer: &Renderer,
    camera_layout: &wgpu::BindGroupLayout,
) -> FootprintTilesOverlay {
    use wgpu::util::DeviceExt;

    let material = FatLineMaterial::new(renderer, camera_layout);
    let color = material.create_instance(
        renderer,
        FatLineMaterialParams {
            base_color: FOOTPRINT_TILE_COLOR,
            width_px: FOOTPRINT_TILE_WIDTH_PX,
        },
    );

    let (vertices, indices) = unit_square_with_diagonals_fat_line_geometry();
    let vertex_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor footprint-tile vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let index_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor footprint-tile indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    let instance_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lumber-editor footprint-tile instances"),
        size: (MAX_FOOTPRINT_TILES as u64) * std::mem::size_of::<MeshInstanceAttribs>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    FootprintTilesOverlay {
        material,
        color,
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        instance_buffer,
        instance_scratch: Vec::with_capacity(MAX_FOOTPRINT_TILES as usize),
    }
}

impl FootprintTilesOverlay {
    /// Rebuild the per-instance buffer from `tiles` (integer world tile
    /// coords from [`currawong::Footprint::tiles`]). Same centring
    /// convention as [`InteractionTilesOverlay::refresh`] — each square is
    /// centred on the integer world coord, matching how the editor draws
    /// the kind body and its interaction tiles. Clamped to
    /// [`MAX_FOOTPRINT_TILES`].
    pub(crate) fn refresh(&mut self, queue: &wgpu::Queue, tiles: &[(i32, i32, i32)]) -> u32 {
        self.instance_scratch.clear();
        let cap = MAX_FOOTPRINT_TILES as usize;
        for &(tx, ty, tz) in tiles.iter().take(cap) {
            let translation = Vec3::new(tx as f32, ty as f32, tz as f32 + FOOTPRINT_TILE_Z_EPSILON);
            let model =
                Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, translation);
            self.instance_scratch
                .push(MeshInstanceAttribs::new(model, Vec4::ONE));
        }
        let count = self.instance_scratch.len() as u32;
        if count > 0 {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instance_scratch),
            );
        }
        count
    }
}
