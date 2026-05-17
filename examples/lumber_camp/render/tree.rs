//! View-side state and helpers specific to drawing trees.
//!
//! The tree body and the designation marker are both
//! [`MeshPart`](currawong::MeshPart)s on the engine
//! [`RenderTemplate`](currawong::RenderTemplate) registered by
//! [`super::LumberCampView::init`]. This module exports each part's
//! factory, the marker's local-frame transform (apex-down above the
//! canopy), the tree's visual AABB, and the click-to-toggle input
//! handler. Marker visibility is a view-side decision set by the
//! per-instance update closure in [`super`]:
//! `instance.mesh_parts[MARKER_PART].visible = components.get::<Designated>(id).is_some()`.

use std::f32::consts::PI;

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Aabb, PbrMaterial, PrimitiveMesh, Renderer, SamplerRegistry, Texture, WorldObjectRef,
};

use super::{MeshTemplate, TemplateParams};
use crate::sim::{Designated, Game, RenderId};

/// Index of the designation-marker mesh part in the tree render template.
pub const MARKER_PART: usize = 1;

/// Half the tree cone's height — used to position the marker above the
/// canopy in the *local* frame.
const TREE_HALF_HEIGHT: f32 = 1.0;
/// Vertical air gap between the tree apex and the marker apex. Keeps the
/// marker from kissing the canopy at this camera distance.
const MARKER_GAP: f32 = 0.12;
/// Half the marker cone's height.
const MARKER_HALF_HEIGHT: f32 = 0.175;

/// Build the tree *body* template — foliage-green cone.
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

/// Build the designation-marker template — a small bright-red cone,
/// oriented apex-down by [`marker_local_transform`].
pub fn new_marker_template(
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
        &PrimitiveMesh::cone(0.18, 0.35, 12, true),
        TemplateParams {
            label: "lumber-camp designation marker",
            albedo_factor: Vec4::new(1.0, 0.15, 0.15, 1.0), // bright red
            metallic: 0.0,
            roughness: 0.55,
        },
    )
}

/// Local-frame transform for the marker: rotate the cone apex-down and
/// sit it just above the tree's canopy. Engine composes
/// `parent.world_xform * marker_local_transform()` per draw.
pub fn marker_local_transform() -> Mat4 {
    let marker_centre_z = TREE_HALF_HEIGHT + MARKER_GAP + MARKER_HALF_HEIGHT;
    Mat4::from_rotation_translation(
        Quat::from_rotation_x(PI),
        Vec3::new(0.0, 0.0, marker_centre_z),
    )
}

/// Visual AABB in the tree's local frame. Encloses the cone (radius 0.60,
/// height 2.0 centred on origin) plus the marker's reach above the
/// canopy. Used by the engine's frustum cull even when the tree isn't
/// designated — cheaper than dynamic bounds, and grazing-edge correct.
pub fn visual_bounds() -> Aabb {
    Aabb::new(Vec3::new(-0.65, -0.65, -1.05), Vec3::new(0.65, 0.65, 1.50))
}

/// Toggle a [`Designated`] component on whichever sim object is currently
/// under the cursor — but only if it's a tree. Pawns and the stockpile
/// click through (left-click on those will mean "select" in a later
/// slice). Lives here rather than in the top-level view because the
/// "is this a tree?" filter and the `Designated` component are both
/// tree-side.
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
