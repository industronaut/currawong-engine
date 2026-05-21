//! Render-object templates and their registry.
//!
//! A render object is a view-side template — a hierarchy of meshes, emitters,
//! materials, and other GPU resources — instanced when the camera shows
//! a sim object that names it. Templates are identified by a user-chosen id
//! type `R` (`Copy + Eq + Hash`; conventionally named `RenderId` at the call
//! site) and stored in a [`RenderRegistry`] owned by the `View`.
//!
//! Templates carry two kinds of parts: [`MeshPart<M, MK>`] for static
//! geometry and [`EmitterPart<E, S>`] for particle emitter attachments.
//! Each part has a local transform from the template root. Per-instance
//! state lives on the persistent
//! [`RenderProxy`](super::RenderProxy); the per-instance update
//! closure reads sim [`Components`](crate::sim::Components) and writes it
//! there each frame.
//!
//! `RenderTemplate` and `RenderRegistry` default `M`, `MK`, `E`, `S` to
//! `()` so callers that haven't committed to a given part kind can use the
//! shorter forms — e.g. `RenderTemplate<MyMesh, MyMat>` for mesh-only
//! templates, or just `RenderTemplate` / `RenderRegistry<R>` for empty
//! templates.

use std::collections::HashMap;
use std::hash::Hash;

use glam::Mat4;

use super::visibility::Aabb;

/// One drawable piece of a [`RenderTemplate`]: a mesh handle, the material
/// key it draws with, and a transform relative to the template root.
///
/// `M` is the user's mesh-handle type (commonly a small enum like
/// `enum MeshId { Cube, Tetra }`); `MK` is the user's material-instance
/// key (commonly `enum MaterialKey { Wood, Stone }`). The engine doesn't
/// own meshes or material instances — it stores these handles and the
/// View resolves them against its own tables when rendering.
#[derive(Clone, Debug)]
pub struct MeshPart<M, MK> {
    pub mesh: M,
    pub material: MK,
    /// Transform from the part's local frame to the template's root frame.
    /// World transform of a drawn instance is
    /// `world_xform_of_object * local_transform`.
    pub local_transform: Mat4,
}

impl<M, MK> MeshPart<M, MK> {
    pub fn new(mesh: M, material: MK, local_transform: Mat4) -> Self {
        Self {
            mesh,
            material,
            local_transform,
        }
    }
}

/// An emitter attachment declared by a [`RenderTemplate`]: which emitter
/// template `E` to spawn, which `S` attachment id keys it (so one
/// template can carry several emitters keyed independently — e.g. flame +
/// smoke + sparks), and a transform relative to the template root.
///
/// The View resolves `E` and `S` against an
/// [`EmitterReconciler<E, S>`](super::EmitterReconciler), which owns the
/// emitter lifecycle and particle integration. The render-object system
/// only declares attachments; the reconciler handles state.
#[derive(Clone, Debug)]
pub struct EmitterPart<E, S> {
    pub template: E,
    pub attachment: S,
    /// Transform from the part's local frame to the template's root frame.
    /// World transform of a declared emitter is
    /// `world_xform_of_object * local_transform`.
    pub local_transform: Mat4,
}

impl<E, S> EmitterPart<E, S> {
    pub fn new(template: E, attachment: S, local_transform: Mat4) -> Self {
        Self {
            template,
            attachment,
            local_transform,
        }
    }
}

/// Static template describing a renderable object. Many sim objects may
/// reference one template (every oak tree → `tree_oak`); per-instance
/// variation lives in transforms and in the
/// [`RenderProxy`](super::RenderProxy) per-instance state the
/// update closure writes.
///
/// Templates are built with [`Self::new`] then chained
/// [`Self::with_mesh_part`] / [`Self::with_emitter_part`] calls. `M`, `MK`,
/// `E`, `S` default to `()` for callers that don't need the corresponding
/// part kind yet.
#[derive(Clone, Debug)]
pub struct RenderTemplate<M = (), MK = (), E = (), S = ()> {
    /// Human-readable name. Used in panics, logs, and tracing.
    pub label: String,
    mesh_parts: Vec<MeshPart<M, MK>>,
    emitter_parts: Vec<EmitterPart<E, S>>,
    visual_bounds: Option<Aabb>,
}

impl<M, MK, E, S> RenderTemplate<M, MK, E, S> {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            mesh_parts: Vec::new(),
            emitter_parts: Vec::new(),
            visual_bounds: None,
        }
    }

    /// Add a [`MeshPart`] to the template. Parts are stored in insertion
    /// order; the View walks them per drawn instance, composing each part's
    /// `local_transform` with the sim object's world transform.
    pub fn with_mesh_part(mut self, mesh: M, material: MK, local_transform: Mat4) -> Self {
        self.mesh_parts
            .push(MeshPart::new(mesh, material, local_transform));
        self
    }

    /// Add an [`EmitterPart`] to the template. The View walks emitter parts
    /// during extraction and declares each on an
    /// [`EmitterReconciler<E, S>`](super::EmitterReconciler), composing the
    /// part's local transform with the sim object's world transform.
    pub fn with_emitter_part(mut self, template: E, attachment: S, local_transform: Mat4) -> Self {
        self.emitter_parts
            .push(EmitterPart::new(template, attachment, local_transform));
        self
    }

    /// All mesh parts in declaration order.
    pub fn mesh_parts(&self) -> &[MeshPart<M, MK>] {
        &self.mesh_parts
    }

    /// All emitter parts in declaration order.
    pub fn emitter_parts(&self) -> &[EmitterPart<E, S>] {
        &self.emitter_parts
    }

    /// Set the template's *visual* AABB — the region of space the template
    /// occupies when rendered, including emitter reach and other effects.
    /// Used by [`RenderProxies::cull`](super::RenderProxies::cull); a
    /// template without visual bounds is never culled.
    ///
    /// CLAUDE.md invariant: visual bounds differ from sim bounds. A 0.5 m
    /// campfire with a 6 m smoke column has a visual AABB that includes
    /// the column.
    pub fn with_visual_bounds(mut self, aabb: Aabb) -> Self {
        self.visual_bounds = Some(aabb);
        self
    }

    /// Visual AABB declared by [`Self::with_visual_bounds`], if any.
    pub fn visual_bounds(&self) -> Option<Aabb> {
        self.visual_bounds
    }
}

/// Registry of [`RenderTemplate`]s keyed by a user-chosen id type `R`.
///
/// Generic over `R` so callers pick the id flavour: a small enum for
/// hand-authored content (`enum RenderId { TreeOak, Campfire }`) or a
/// numeric/asset handle later when templates are data-driven. `M`, `MK`,
/// `E`, `S` flow through to [`RenderTemplate`]; they default to `()` for
/// callers that haven't yet committed to the corresponding part kind.
pub struct RenderRegistry<R, M = (), MK = (), E = (), S = ()>
where
    R: Clone + Eq + Hash,
{
    templates: HashMap<R, RenderTemplate<M, MK, E, S>>,
}

impl<R, M, MK, E, S> RenderRegistry<R, M, MK, E, S>
where
    R: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Register `template` under `id`. Replaces any existing template
    /// with the same id; live instances built from the old template
    /// are unaffected until they are rebuilt (a later-step concern).
    pub fn register(&mut self, id: R, template: RenderTemplate<M, MK, E, S>) {
        self.templates.insert(id, template);
    }

    /// Look up the template registered under `id`, if any. Takes `&R` so
    /// non-`Copy` keys (e.g. [`KindId`](crate::data::KindId)) can be looked
    /// up without consuming an owned copy.
    pub fn get(&self, id: &R) -> Option<&RenderTemplate<M, MK, E, S>> {
        self.templates.get(id)
    }

    /// Number of registered templates.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// True when no templates are registered.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

impl<R, M, MK, E, S> Default for RenderRegistry<R, M, MK, E, S>
where
    R: Clone + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum RenderId {
        TreeOak,
        Campfire,
    }

    #[test]
    fn empty_registry_has_no_templates() {
        let reg: RenderRegistry<RenderId> = RenderRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.get(&RenderId::TreeOak).is_none());
    }

    #[test]
    fn register_then_lookup_returns_template() {
        let mut reg: RenderRegistry<RenderId> = RenderRegistry::new();
        reg.register(RenderId::TreeOak, RenderTemplate::new("tree_oak"));

        assert_eq!(reg.len(), 1);
        assert_eq!(
            reg.get(&RenderId::TreeOak).map(|t| t.label.as_str()),
            Some("tree_oak")
        );
        assert!(reg.get(&RenderId::Campfire).is_none());
    }

    #[test]
    fn re_register_replaces_template() {
        let mut reg: RenderRegistry<RenderId> = RenderRegistry::new();
        reg.register(RenderId::TreeOak, RenderTemplate::new("first"));
        reg.register(RenderId::TreeOak, RenderTemplate::new("second"));

        assert_eq!(reg.len(), 1);
        assert_eq!(
            reg.get(&RenderId::TreeOak).map(|t| t.label.as_str()),
            Some("second")
        );
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum TestMesh {
        Cube,
        Plane,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum TestMat {
        Wood,
        Stone,
    }

    #[test]
    fn template_has_no_mesh_parts_by_default() {
        let t: RenderTemplate<TestMesh, TestMat> = RenderTemplate::new("bare");
        assert!(t.mesh_parts().is_empty());
    }

    #[test]
    fn with_mesh_part_preserves_declaration_order() {
        let t: RenderTemplate<TestMesh, TestMat> = RenderTemplate::new("campfire")
            .with_mesh_part(TestMesh::Cube, TestMat::Wood, Mat4::IDENTITY)
            .with_mesh_part(
                TestMesh::Plane,
                TestMat::Stone,
                Mat4::from_translation(Vec3::new(0.0, -0.5, 0.0)),
            );

        let parts = t.mesh_parts();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].mesh, TestMesh::Cube);
        assert_eq!(parts[0].material, TestMat::Wood);
        assert_eq!(parts[1].mesh, TestMesh::Plane);
        assert_eq!(parts[1].material, TestMat::Stone);
        // Column 3 of an Mat4 holds the translation.
        assert_eq!(parts[1].local_transform.col(3).truncate().y, -0.5);
    }

    #[test]
    fn registry_stores_template_with_parts() {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        enum RId {
            Campfire,
        }

        let mut reg: RenderRegistry<RId, TestMesh, TestMat> = RenderRegistry::new();
        reg.register(
            RId::Campfire,
            RenderTemplate::new("campfire")
                .with_mesh_part(TestMesh::Cube, TestMat::Wood, Mat4::IDENTITY)
                .with_mesh_part(
                    TestMesh::Plane,
                    TestMat::Stone,
                    Mat4::from_translation(Vec3::new(0.0, -0.5, 0.0)),
                ),
        );

        let template = reg.get(&RId::Campfire).expect("registered");
        assert_eq!(template.label, "campfire");
        assert_eq!(template.mesh_parts().len(), 2);
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum TestEmitter {
        Flame,
        Smoke,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum TestEmitterSlot {
        Main,
        Secondary,
    }

    #[test]
    fn template_has_no_emitter_parts_by_default() {
        let t: RenderTemplate = RenderTemplate::new("bare");
        assert!(t.emitter_parts().is_empty());
    }

    #[test]
    fn with_emitter_part_preserves_declaration_order() {
        let t: RenderTemplate<(), (), TestEmitter, TestEmitterSlot> =
            RenderTemplate::new("campfire")
                .with_emitter_part(
                    TestEmitter::Flame,
                    TestEmitterSlot::Main,
                    Mat4::from_translation(Vec3::new(0.0, 0.45, 0.0)),
                )
                .with_emitter_part(
                    TestEmitter::Smoke,
                    TestEmitterSlot::Secondary,
                    Mat4::from_translation(Vec3::new(0.0, 0.85, 0.0)),
                );

        let parts = t.emitter_parts();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].template, TestEmitter::Flame);
        assert_eq!(parts[0].attachment, TestEmitterSlot::Main);
        assert_eq!(parts[1].template, TestEmitter::Smoke);
        assert_eq!(parts[1].attachment, TestEmitterSlot::Secondary);
        assert_eq!(parts[1].local_transform.col(3).truncate().y, 0.85);
    }

    #[test]
    fn template_can_carry_all_part_kinds() {
        let t: RenderTemplate<TestMesh, TestMat, TestEmitter, TestEmitterSlot> =
            RenderTemplate::new("rich")
                .with_mesh_part(TestMesh::Cube, TestMat::Wood, Mat4::IDENTITY)
                .with_emitter_part(TestEmitter::Flame, TestEmitterSlot::Main, Mat4::IDENTITY);

        assert_eq!(t.mesh_parts().len(), 1);
        assert_eq!(t.emitter_parts().len(), 1);
    }

    #[test]
    fn visual_bounds_default_to_none() {
        let t: RenderTemplate = RenderTemplate::new("bare");
        assert!(t.visual_bounds().is_none());
    }

    #[test]
    fn with_visual_bounds_records_the_aabb() {
        let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 2.5, 1.0));
        let t: RenderTemplate = RenderTemplate::new("campfire").with_visual_bounds(bounds);
        assert_eq!(t.visual_bounds(), Some(bounds));
    }
}
