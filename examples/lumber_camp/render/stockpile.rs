//! View-side state and helpers specific to drawing stockpiles.
//!
//! The trivial case in the per-kind pattern: a stockpile is a single
//! [`MeshPart`](currawong::MeshPart) on the engine
//! [`RenderTemplate`](currawong::RenderTemplate), with no per-instance
//! update logic and no ancillary scratch. So this module is just two
//! factories — body template + visual AABB.
//!
//! When stockpiles grow visual state (a pile that fills as
//! [`WoodStored`](crate::sim::WoodStored) climbs, a foreman idle
//! animation, …) they grow it here: another `MeshPart` on the template,
//! another arm in the per-instance update closure in [`super`]. The
//! kind's footprint in [`super`] stays one registration line.

use currawong::glam::{Vec3, Vec4};
use currawong::{Aabb, PbrMaterial, PrimitiveMesh, Renderer, SamplerRegistry, Texture};

use super::{MeshTemplate, TemplateParams};

/// Build the stockpile body template (1 m cube, wood-brown).
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

/// Visual AABB in the stockpile's local frame. Encloses the 1 m cube
/// centred on the origin with a touch of slack.
pub fn visual_bounds() -> Aabb {
    Aabb::new(Vec3::splat(-0.55), Vec3::splat(0.55))
}
