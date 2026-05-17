//! View-side state and helpers specific to drawing trees.
//!
//! Owns the foliage-green tree body template, the designation-marker
//! template + instance buffer, and the click-to-toggle input handler.
//! Same dispatch shape as [`super::pawn`]: the body template is built
//! here but registered in the central templates map for the bucket draw
//! loop, while marker scratch and draws live alongside the data.
//!
//! Markers ride above any tree with a [`Designated`] component. They don't
//! reserve hit IDs — the tree underneath is the click target, not the
//! marker — so they live entirely outside `InstanceBuckets` and use a
//! dedicated per-frame upload like the carried-log path does.

use std::f32::consts::PI;

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    MeshInstanceAttribs, PbrMaterial, PrimitiveMesh, Renderer, SamplerRegistry, Texture,
    WorldObjectId, WorldObjectRef, WorldTransform, Zone, wgpu,
};

use super::{MeshTemplate, TemplateParams};
use crate::sim::{Designated, Game, RenderId};

/// Upper bound on simultaneously-designated trees rendered in one frame.
/// Way above the playable count; sized for the marker instance buffer only.
const MAX_MARKERS: u32 = 64;
/// Half the tree's vertical extent — used to compute the world-space apex
/// of a tree whose transform position is its centre. Mirrors the cone height
/// used to build the tree mesh.
const TREE_HALF_HEIGHT: f32 = 1.0;
/// Vertical air gap between the tree apex and the marker apex. Keeps the
/// marker from kissing the canopy at this camera distance.
const MARKER_GAP: f32 = 0.12;
/// Half the marker cone's height. Matches the marker mesh built in
/// [`TreeRenderer::new`].
const MARKER_HALF_HEIGHT: f32 = 0.175;

/// Build the tree *body* template (mesh + PBR material instance). Called
/// from [`super::LumberCampView::init`] and registered in the central
/// templates map.
pub fn new_body_template(
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    albedo: &Texture,
) -> MeshTemplate {
    MeshTemplate::new(
        renderer,
        material,
        samplers,
        albedo,
        &PrimitiveMesh::cone(0.60, 2.0, 16, true),
        TemplateParams {
            label: "lumber-camp tree",
            albedo_factor: Vec4::new(0.20, 0.50, 0.18, 1.0), // foliage green
            metallic: 0.0,
            roughness: 0.95,
        },
    )
}

/// Per-frame tree-only state: the designation marker template, its instance
/// buffer, and the per-frame scratch the fused walk fills via
/// [`push_marker_if_designated`](Self::push_marker_if_designated).
pub struct TreeRenderer {
    marker_template: MeshTemplate,
    marker_buffer: wgpu::Buffer,
    marker_scratch: Vec<MeshInstanceAttribs>,
}

impl TreeRenderer {
    pub fn new(
        renderer: &Renderer,
        material: &PbrMaterial,
        samplers: &SamplerRegistry,
        albedo: &Texture,
    ) -> Self {
        // Small cone, oriented apex-down at render time by the per-instance
        // model matrix. Bright red so it reads against the green canopy
        // without needing a per-frame tint.
        let marker_template = MeshTemplate::new(
            renderer,
            material,
            samplers,
            albedo,
            &PrimitiveMesh::cone(0.18, 0.35, 12, true),
            TemplateParams {
                label: "lumber-camp designation marker",
                albedo_factor: Vec4::new(1.0, 0.15, 0.15, 1.0), // bright red
                metallic: 0.0,
                roughness: 0.55,
            },
        );
        let marker_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lumber-camp marker instances"),
            size: u64::from(MAX_MARKERS) * std::mem::size_of::<MeshInstanceAttribs>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            marker_template,
            marker_buffer,
            marker_scratch: Vec::with_capacity(MAX_MARKERS as usize),
        }
    }

    /// Reset per-frame marker scratch. Called once by the top-level view
    /// at the start of each render.
    pub fn begin_frame(&mut self) {
        self.marker_scratch.clear();
    }

    /// If the tree at `id` has a [`Designated`] component, queue a marker
    /// instance floating apex-down above its canopy. Caller is the fused
    /// walk; passing the tree's [`WorldTransform`] avoids a second
    /// `zone.get(id)` for the position.
    pub fn push_marker_if_designated(
        &mut self,
        zone: &Zone,
        id: WorldObjectId,
        transform: &WorldTransform,
    ) {
        if zone.components().get::<Designated>(id).is_none() {
            return;
        }
        if self.marker_scratch.len() >= MAX_MARKERS as usize {
            return;
        }
        let tree_apex_z = transform.position.z + TREE_HALF_HEIGHT;
        let marker_centre = Vec3::new(
            transform.position.x,
            transform.position.y,
            tree_apex_z + MARKER_GAP + MARKER_HALF_HEIGHT,
        );
        let model = Mat4::from_rotation_translation(Quat::from_rotation_x(PI), marker_centre);
        self.marker_scratch
            .push(MeshInstanceAttribs::new(model, Vec4::ONE));
    }

    /// Upload this frame's designation markers. No-op when nothing was
    /// queued.
    pub fn upload_markers(&self, queue: &wgpu::Queue) {
        if self.marker_scratch.is_empty() {
            return;
        }
        queue.write_buffer(
            &self.marker_buffer,
            0,
            bytemuck::cast_slice(&self.marker_scratch),
        );
    }

    /// Draw the queued markers. Caller must have bound the PBR pipeline and
    /// the camera+scene bind groups already; this binds the marker
    /// material/mesh/instance buffers and issues the indexed-instanced draw.
    pub fn draw_markers(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.marker_scratch.is_empty() {
            return;
        }
        pass.set_bind_group(2, self.marker_template.material.bind_group(), &[]);
        pass.set_vertex_buffer(0, self.marker_template.vertices.slice(..));
        pass.set_vertex_buffer(1, self.marker_buffer.slice(..));
        pass.set_index_buffer(
            self.marker_template.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(
            0..self.marker_template.index_count,
            0,
            0..self.marker_scratch.len() as u32,
        );
    }
}

/// Toggle a [`Designated`] component on whichever sim object is currently
/// under the cursor — but only if it's a tree. Pawns and the stockpile
/// click through (left-click on those will mean "select" in a later slice).
/// Lives here rather than in the top-level view because the "is this a
/// tree?" filter and the `Designated` component are both tree-side.
pub fn toggle_designation_under_cursor(sim: &mut Game, hovered: Option<WorldObjectRef>) {
    let Some(WorldObjectRef { zone, id }) = hovered else {
        return;
    };
    let Some(zone) = sim.zones.get_mut(zone) else {
        return;
    };
    // A readback id can be stale across object removal; component lookup on a
    // dead id returns None and we no-op.
    if !matches!(zone.components().get::<RenderId>(id), Some(RenderId::Tree)) {
        return;
    }
    if zone.components().get::<Designated>(id).is_some() {
        zone.components_mut().remove::<Designated>(id);
    } else {
        zone.components_mut().insert(id, Designated);
    }
}
