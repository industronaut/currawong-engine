//! View-side helpers for trees.
//!
//! The tree's *body* mesh + texture come from each species' RON def via
//! [`super::build_body_template`] — oak.glb / oak_bark.png for
//! `currawong:oak_tree`, pine.glb / pine_bark.png for
//! `currawong:pine_tree`. The *designation marker* is a single shared
//! procedural red cone built once at init and reused across every species'
//! template via [`super::PartKey::Marker`]. Marker visibility is a
//! view-side decision set by [`update_instance`] from the sim's
//! [`Designated`] component.

use std::f32::consts::PI;

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Aabb, AssetServer, Components, InlineTemplate, MeshTemplate, PbrMaterial, PbrMaterialInstance,
    PrimitiveMesh, RenderProxy, Renderer, SamplerRegistry, WorldObjectRef,
};

use crate::sim::Designated;

/// Index of the designation-marker mesh part in every tree's render
/// template — second part after the body. Stable because
/// [`super::RenderShape::register_template`] adds parts in declaration
/// order and only the tree shape adds the marker.
const MARKER_PART: usize = 1;

/// Vertical air gap between the tree apex and the marker apex. Keeps the
/// marker from kissing the canopy at this camera distance.
const MARKER_GAP: f32 = 0.12;
/// Half the marker cone's height.
const MARKER_HALF_HEIGHT: f32 = 0.175;

/// Build the shared designation-marker template — a small bright-red
/// cone, oriented apex-down by [`marker_local_transform`]. Procedural
/// rather than streamed because the marker is a stylized HUD indicator,
/// not authored content.
pub fn new_marker_template(
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
) -> MeshTemplate<PbrMaterialInstance> {
    material.inline_template(
        renderer,
        samplers,
        asset_server,
        InlineTemplate {
            label: "lumber-camp designation marker",
            mesh: &PrimitiveMesh::cone(0.18, 0.35, 12, true),
            bounds: Aabb::new(
                Vec3::new(-0.18, -0.18, -0.175),
                Vec3::new(0.18, 0.18, 0.175),
            ),
            albedo_factor: Vec4::new(1.0, 0.15, 0.15, 1.0), // bright red
            metallic: 0.0,
            roughness: 0.55,
        },
    )
}

/// Local-frame transform for the marker, given the tree's *body* bounds.
/// Sits the marker just above the canopy with a small gap, apex-down.
/// Per-species because pine canopies sit higher than oak — the marker
/// rides each species' own height.
pub fn marker_local_transform(body_bounds: &Aabb) -> Mat4 {
    let marker_centre_z = body_bounds.max.z + MARKER_GAP + MARKER_HALF_HEIGHT;
    Mat4::from_rotation_translation(
        Quat::from_rotation_x(PI),
        Vec3::new(0.0, 0.0, marker_centre_z),
    )
}

/// Visual AABB for a tree species' [`RenderTemplate`](currawong::RenderTemplate),
/// extending the body's bounds upward to enclose the designation marker so
/// the engine frustum-cull doesn't pop the tree out when only its marker
/// is on-screen.
pub fn extended_bounds(body_bounds: &Aabb) -> Aabb {
    let marker_top = body_bounds.max.z + MARKER_GAP + 2.0 * MARKER_HALF_HEIGHT;
    let xy_extent = body_bounds.max.x.max(body_bounds.max.y).max(0.65);
    Aabb::new(
        Vec3::new(-xy_extent, -xy_extent, body_bounds.min.z),
        Vec3::new(xy_extent, xy_extent, marker_top),
    )
}

/// Per-instance update for a live tree proxy. Called by the dispatcher
/// in [`super`] from inside
/// [`RenderObjectTraversal::update_instances`](currawong::RenderObjectTraversal::update_instances).
///
/// Owns every sim→view decision for trees: gates the designation marker
/// on the [`Designated`] component.
pub fn update_instance(
    parent: WorldObjectRef,
    components: &Components,
    instance: &mut RenderProxy,
) {
    instance.mesh_parts[MARKER_PART].visible = components.get::<Designated>(parent.id).is_some();
}

/// Emit a [`Command::ToggleDesignation`] for whichever sim object is
/// currently under the cursor. The view doesn't check whether the target
/// is a tree — [`Game::apply_command`] re-validates, which is the right
/// place anyway: a future replay or scripted-test command stream gets the
/// same validation without round-tripping through the view.
pub fn emit_toggle_designation(
    cmds: &mut currawong::CommandQueue<crate::sim::Command>,
    hovered: Option<WorldObjectRef>,
) {
    let Some(target) = hovered else {
        return;
    };
    cmds.push_now(crate::sim::Command::ToggleDesignation { target });
}
