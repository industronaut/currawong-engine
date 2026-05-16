//! View-side state and helpers specific to drawing stockpiles.
//!
//! The trivial case in the per-kind pattern: a stockpile carries no
//! per-instance scratch, no input handler, and no ancillary draws — just a
//! body template. So this module is a single factory function and no
//! struct. The empty `RenderId::Stockpile` arm in [`super`]'s fused walk
//! dispatch reflects the same fact.
//!
//! When stockpiles grow visual state (a pile that fills as `WoodStored`
//! climbs, a foreman idle animation, …) they grow it here without
//! touching [`super`]. That's the whole point of the per-kind layout:
//! kinds only carry what they need, and growing a kind is a local edit.

use currawong::glam::{Vec3, Vec4};
use currawong::{PbrMaterial, PrimitiveMesh, Renderer, SamplerRegistry, Texture};

use super::{MeshTemplate, TemplateParams};

/// Build the stockpile body template (mesh + PBR material instance).
/// Called from [`super::LumberCampView::init`] and registered in the
/// central templates map.
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
        &PrimitiveMesh::cube(Vec3::ONE),
        TemplateParams {
            label: "lumber-camp stockpile",
            albedo_factor: Vec4::new(0.55, 0.32, 0.18, 1.0), // wood brown
            metallic: 0.0,
            roughness: 0.85,
        },
    )
}
