//! Static scene elements that establish the viewing context: the
//! checkerboard ground plane that catches shadows and provides
//! figure-ground contrast for the displayed kind, and the yellow
//! facing-direction arrow that indicates the subject's orientation.
//!
//! Neither is tied to a sim object — the floor is a single fixed-instance
//! draw with an identity model matrix, the arrow's per-instance matrix is
//! rewritten each frame from the subject's `Facing` and AABB. They share
//! this module because both function as "what am I looking at" rather than
//! the data-driven overlays (visual bounds, interaction tiles, footprint
//! tiles) which describe per-kind metadata.

use currawong::glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use currawong::{
    Aabb, AssetServer, FatLineMaterial, FatLineMaterialInstance, FatLineMaterialParams,
    FatLineVertex, Handle, MeshInstanceAttribs, PbrMaterial, PbrMaterialInstance,
    PbrMaterialParams, PrimitiveMesh, Renderer, SamplerKind, SamplerRegistry, Texture,
    WorldTransform, wgpu,
};

// --- Ground plane ------------------------------------------------------

/// Edge length of the ground plane in metres. Bigger than any kind the
/// editor is likely to show (lumber-camp's biggest is ~6 m) so the floor
/// always reaches past the visible orbit-rig frustum.
const GROUND_SIZE: f32 = 100.0;
/// World-space size of one checkerboard cell. 25 cm reads as a fine-grained
/// scale reference for sub-metre kinds without becoming visual noise on the
/// larger ones.
const GROUND_CELL_SIZE: f32 = 0.25;

/// GPU resources for the editor's static checkerboard floor. One quad, one
/// instance, one PBR material — sized large enough to fill the camera for
/// every kind, so its model matrix is identity and never updates.
pub(crate) struct GroundPlane {
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) index_count: u32,
    pub(crate) instance_buffer: wgpu::Buffer,
    pub(crate) material: PbrMaterialInstance,
}

pub(crate) fn build_ground_plane(
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
) -> GroundPlane {
    use wgpu::util::DeviceExt;

    // A 2×2 checker baked into a 64×64 texture, tiled across the plane
    // with `LinearRepeat`. UV scale is chosen so each repeat covers two
    // cells (one light + one dark) at `GROUND_CELL_SIZE` metres each.
    let texture = make_checker_texture(renderer);
    let albedo = Handle::ready(texture);
    let ground_material = material.create_instance(
        renderer,
        samplers,
        asset_server,
        PbrMaterialParams {
            albedo,
            sampler: SamplerKind::LinearRepeat,
            albedo_factor: Vec4::ONE,
            // Matte dielectric — the surface should look like rough painted
            // concrete, not a polished display table; keeps the kind's
            // specular highlights the obvious figure-ground signal.
            metallic: 0.0,
            roughness: 0.95,
        },
    );

    // One-quad plane on XY at z=0; UV scaled so `LinearRepeat` tiles the
    // 2×2 checker `GROUND_SIZE / (2 * GROUND_CELL_SIZE)` times per axis.
    let mut mesh = PrimitiveMesh::plane(Vec2::splat(GROUND_SIZE), UVec2::ONE);
    let uv_scale = GROUND_SIZE / (2.0 * GROUND_CELL_SIZE);
    for v in &mut mesh.vertices {
        v.uv[0] *= uv_scale;
        v.uv[1] *= uv_scale;
    }
    let vertex_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor ground vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let index_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor ground indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    // Static one-instance buffer — identity model, no tint, no hit ID.
    // Never rewritten, so we don't need a separate scratch + upload path.
    let instance = MeshInstanceAttribs::new(Mat4::IDENTITY, Vec4::ONE);
    let instance_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor ground instance"),
            contents: bytemuck::bytes_of(&instance),
            usage: wgpu::BufferUsages::VERTEX,
        });

    GroundPlane {
        vertex_buffer,
        index_buffer,
        index_count: mesh.index_count(),
        instance_buffer,
        material: ground_material,
    }
}

/// Bake a 64×64 RGBA8 checkerboard intended for `LinearRepeat` tiling. Two
/// soft greys keep the floor reading as a backdrop rather than competing
/// with the displayed kind. Sharp cell edges at 32-px boundaries mean the
/// mip chain handles distant cells cleanly without bleeding the two tones
/// together.
fn make_checker_texture(renderer: &Renderer) -> Texture {
    const SIZE: u32 = 64;
    const CELL_PX: u32 = SIZE / 2;
    const LIGHT: [u8; 4] = [160, 160, 160, 255];
    const DARK: [u8; 4] = [110, 110, 110, 255];
    let mut bytes = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let cx = (x / CELL_PX) & 1;
            let cy = (y / CELL_PX) & 1;
            let c = if cx == cy { LIGHT } else { DARK };
            bytes.extend_from_slice(&c);
        }
    }
    Texture::from_rgba8(renderer, "lumber-editor checker", SIZE, SIZE, &bytes, true)
}

// --- Facing-arrow overlay --------------------------------------------

/// Saturated yellow for the facing-direction arrow. Matches the bounds
/// wireframe — both describe the selected kind's orientation envelope.
const FACING_ARROW_COLOR: Vec4 = Vec4::new(1.0, 1.0, 0.0, 1.0);
/// Screen-space stroke width for the arrow in pixels. Thick enough to
/// read clearly against the checker floor; heavier than the bounds
/// wireframe so the two overlays don't visually conflate where they
/// share the yellow tone.
const FACING_ARROW_WIDTH_PX: f32 = 6.0;
/// Length of the arrow shaft in metres, measured outward from the AABB
/// front face in the facing direction. Per the user-facing spec for the
/// editor.
const FACING_ARROW_LENGTH: f32 = 1.0;
/// Length of each arrowhead segment as a fraction of the shaft length.
/// Picked to read clearly at typical orbit distances without dominating
/// the shaft.
const FACING_ARROW_HEAD_FRAC: f32 = 0.25;
/// Vertical lift above the ground plane to avoid z-fighting with the
/// checker. Slightly above the footprint tiles' epsilon so the arrow
/// reads on top where they overlap.
const FACING_ARROW_Z_EPSILON: f32 = 0.02;

/// GPU resources for the facing-direction arrow overlay. Same fat-line
/// pipeline as the bounds wireframe and tile overlays, but the static
/// vertex/index data is a unit-length shaft along +X plus two
/// arrowhead segments. The per-instance model matrix is rewritten each
/// frame in `render` to translate the shaft origin to the front face
/// of the selected kind's visual AABB and rotate by the subject's
/// `Facing`.
pub(crate) struct FacingArrowOverlay {
    pub(crate) material: FatLineMaterial,
    pub(crate) color: FatLineMaterialInstance,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) index_count: u32,
    pub(crate) instance_buffer: wgpu::Buffer,
}

/// Fat-line vertex + index data for a unit-length facing arrow in local
/// space: shaft from `(0, 0, 0)` to `(FACING_ARROW_LENGTH, 0, 0)`, plus
/// two arrowhead segments forming the tip. Three segments → 12 verts,
/// 18 triangle-list indices. Sibling to [`unit_square_fat_line_geometry`].
fn unit_facing_arrow_fat_line_geometry() -> (Vec<FatLineVertex>, Vec<u16>) {
    let length = FACING_ARROW_LENGTH;
    let head = length * FACING_ARROW_HEAD_FRAC;
    let tip: [f32; 3] = [length, 0.0, 0.0];
    let base_x = length - head;
    // Shaft + two arrowhead lines. All Z = 0; lifted off the ground via
    // the model matrix in `write_facing_arrow_instance`.
    let segments: [([f32; 3], [f32; 3]); 3] = [
        ([0.0, 0.0, 0.0], tip),
        (tip, [base_x, head, 0.0]),
        (tip, [base_x, -head, 0.0]),
    ];

    let mut vertices = Vec::with_capacity(12);
    let mut indices = Vec::with_capacity(18);
    for (pos_a, pos_b) in segments {
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

pub(crate) fn build_facing_arrow_overlay(
    renderer: &Renderer,
    camera_layout: &wgpu::BindGroupLayout,
) -> FacingArrowOverlay {
    use wgpu::util::DeviceExt;

    let material = FatLineMaterial::new(renderer, camera_layout);
    let color = material.create_instance(
        renderer,
        FatLineMaterialParams {
            base_color: FACING_ARROW_COLOR,
            width_px: FACING_ARROW_WIDTH_PX,
        },
    );

    let (vertices, indices) = unit_facing_arrow_fat_line_geometry();
    let vertex_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor facing-arrow vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let index_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor facing-arrow indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    let placeholder = MeshInstanceAttribs::new(Mat4::IDENTITY, Vec4::ONE);
    let instance_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor facing-arrow instance"),
            contents: bytemuck::bytes_of(&placeholder),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

    FacingArrowOverlay {
        material,
        color,
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        instance_buffer,
    }
}

/// Rewrite the facing-arrow instance buffer so the unit-length local-X
/// shaft starts at the AABB's front face (in the subject's facing
/// direction) and points outward along that direction. `Facing` is
/// yaw-only, so the rotation only affects X/Y — the world-space Z stays
/// at [`FACING_ARROW_Z_EPSILON`] regardless.
pub(crate) fn write_facing_arrow_instance(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    transform: WorldTransform,
    aabb: Aabb,
) {
    let position = transform.position.to_vec3();
    let rotation = transform.facing.to_quat();
    let aabb_offset = rotation * Vec3::new(aabb.max.x, 0.0, 0.0);
    let translation = Vec3::new(
        position.x + aabb_offset.x,
        position.y + aabb_offset.y,
        FACING_ARROW_Z_EPSILON,
    );
    let model = Mat4::from_scale_rotation_translation(Vec3::ONE, rotation, translation);
    let attribs = MeshInstanceAttribs::new(model, Vec4::ONE);
    queue.write_buffer(buffer, 0, bytemuck::bytes_of(&attribs));
}
