//! Render-object templates and their registry.
//!
//! A render object is a view-side template — a hierarchy of meshes, emitters,
//! materials, and other GPU resources — instanced when the camera shows
//! a sim object that names it. Templates are identified by a user-chosen id
//! type `R` (`Copy + Eq + Hash`; conventionally named `RenderId` at the call
//! site) and stored in a [`RenderRegistry`] owned by the `View`.
//!
//! Templates declare typed **slots** — named parameters with a fixed
//! [`SlotKind`] — that instances later bind to concrete [`SlotValue`]s.
//! Templates carry two kinds of parts: [`MeshPart<M, MK>`] for static
//! geometry and [`EmitterPart<E, S>`] for particle emitter attachments.
//! Each part has a local transform from the template root. Slot-driven
//! parameter routing and visual-behaviour bindings land in later steps.
//! Re-registering an id silently replaces the previous template — same
//! idiom as
//! [`EmitterReconciler::register_template`](super::EmitterReconciler::register_template)
//! and [`InstanceBuckets::register`](super::InstanceBuckets::register).
//!
//! `RenderTemplate` and `RenderRegistry` default `M`, `MK`, `E`, `S` to
//! `()` so callers that haven't committed to a given part kind can use the
//! shorter forms — e.g. `RenderTemplate<MyMesh, MyMat>` for mesh-only
//! templates, or just `RenderTemplate` / `RenderRegistry<R>` for
//! slot-schema-only templates.

use std::collections::HashMap;
use std::hash::Hash;

use glam::{Mat4, Vec2, Vec3, Vec4};

/// Type of a parameter slot on a [`RenderTemplate`]. Each variant pairs with
/// a [`SlotValue`] of the same name. The set is deliberately closed: adding
/// a kind requires a code change, not a string tag, which keeps slots strongly
/// typed end-to-end (Godot `@export` / Unreal `UPROPERTY`-style).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SlotKind {
    F32,
    Vec2,
    Vec3,
    Vec4,
    /// Linear RGBA in `[0, 1]`. Packed identically to [`SlotKind::Vec4`];
    /// separated so consumers can route to colour-aware bindings (sRGB
    /// textures, tone mapping) without sniffing slot names.
    Color,
    Bool,
    I32,
    U32,
}

/// Concrete value for a slot. Each variant corresponds to a [`SlotKind`] of
/// the same name; use [`SlotValue::kind`] to recover the kind for validation
/// against a template's schema.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SlotValue {
    F32(f32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
    Color(Vec4),
    Bool(bool),
    I32(i32),
    U32(u32),
}

impl SlotValue {
    /// The [`SlotKind`] this value satisfies.
    pub fn kind(&self) -> SlotKind {
        match self {
            SlotValue::F32(_) => SlotKind::F32,
            SlotValue::Vec2(_) => SlotKind::Vec2,
            SlotValue::Vec3(_) => SlotKind::Vec3,
            SlotValue::Vec4(_) => SlotKind::Vec4,
            SlotValue::Color(_) => SlotKind::Color,
            SlotValue::Bool(_) => SlotKind::Bool,
            SlotValue::I32(_) => SlotKind::I32,
            SlotValue::U32(_) => SlotKind::U32,
        }
    }
}

/// Schema entry for one slot on a [`RenderTemplate`]: a name plus its kind.
/// Slots are stored in declaration order so consumers can derive stable
/// uniform / instance-buffer layouts from them.
#[derive(Clone, Debug)]
pub struct SlotDescriptor {
    pub name: String,
    pub kind: SlotKind,
}

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
/// template `E` to spawn, which `S` slot the attachment fills (so one
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
    pub slot: S,
    /// Transform from the part's local frame to the template's root frame.
    /// World transform of a declared emitter is
    /// `world_xform_of_object * local_transform`.
    pub local_transform: Mat4,
}

impl<E, S> EmitterPart<E, S> {
    pub fn new(template: E, slot: S, local_transform: Mat4) -> Self {
        Self {
            template,
            slot,
            local_transform,
        }
    }
}

/// Static template describing a renderable object. Many sim objects may
/// reference one template (every oak tree → `tree_oak`); per-instance
/// variation lives in transforms and slot values.
///
/// Templates are built with [`Self::new`] then chained
/// [`Self::with_slot`] / [`Self::with_mesh_part`] / [`Self::with_emitter_part`]
/// calls. `M`, `MK`, `E`, `S` default to `()` for callers that don't need
/// the corresponding part kind yet.
#[derive(Clone, Debug)]
pub struct RenderTemplate<M = (), MK = (), E = (), S = ()> {
    /// Human-readable name. Used in panics, logs, and tracing.
    pub label: String,
    slots: Vec<SlotDescriptor>,
    mesh_parts: Vec<MeshPart<M, MK>>,
    emitter_parts: Vec<EmitterPart<E, S>>,
}

impl<M, MK, E, S> RenderTemplate<M, MK, E, S> {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            slots: Vec::new(),
            mesh_parts: Vec::new(),
            emitter_parts: Vec::new(),
        }
    }

    /// Declare a slot. Slot names must be unique within a template; passing
    /// a duplicate panics — programming error, not runtime condition. Returns
    /// `self` for chaining.
    pub fn with_slot(mut self, name: impl Into<String>, kind: SlotKind) -> Self {
        let name = name.into();
        assert!(
            !self.slots.iter().any(|s| s.name == name),
            "RenderTemplate '{}' already has a slot named '{}'",
            self.label,
            name
        );
        self.slots.push(SlotDescriptor { name, kind });
        self
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
    pub fn with_emitter_part(mut self, template: E, slot: S, local_transform: Mat4) -> Self {
        self.emitter_parts
            .push(EmitterPart::new(template, slot, local_transform));
        self
    }

    /// All slots in declaration order.
    pub fn slots(&self) -> &[SlotDescriptor] {
        &self.slots
    }

    /// Look up a slot descriptor by name.
    pub fn slot(&self, name: &str) -> Option<&SlotDescriptor> {
        self.slots.iter().find(|s| s.name == name)
    }

    /// All mesh parts in declaration order.
    pub fn mesh_parts(&self) -> &[MeshPart<M, MK>] {
        &self.mesh_parts
    }

    /// All emitter parts in declaration order.
    pub fn emitter_parts(&self) -> &[EmitterPart<E, S>] {
        &self.emitter_parts
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
    R: Copy + Eq + Hash,
{
    templates: HashMap<R, RenderTemplate<M, MK, E, S>>,
}

impl<R, M, MK, E, S> RenderRegistry<R, M, MK, E, S>
where
    R: Copy + Eq + Hash,
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

    /// Look up the template registered under `id`, if any.
    pub fn get(&self, id: R) -> Option<&RenderTemplate<M, MK, E, S>> {
        self.templates.get(&id)
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
    R: Copy + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(reg.get(RenderId::TreeOak).is_none());
    }

    #[test]
    fn register_then_lookup_returns_template() {
        let mut reg: RenderRegistry<RenderId> = RenderRegistry::new();
        reg.register(RenderId::TreeOak, RenderTemplate::new("tree_oak"));

        assert_eq!(reg.len(), 1);
        assert_eq!(
            reg.get(RenderId::TreeOak).map(|t| t.label.as_str()),
            Some("tree_oak")
        );
        assert!(reg.get(RenderId::Campfire).is_none());
    }

    #[test]
    fn re_register_replaces_template() {
        let mut reg: RenderRegistry<RenderId> = RenderRegistry::new();
        reg.register(RenderId::TreeOak, RenderTemplate::new("first"));
        reg.register(RenderId::TreeOak, RenderTemplate::new("second"));

        assert_eq!(reg.len(), 1);
        assert_eq!(
            reg.get(RenderId::TreeOak).map(|t| t.label.as_str()),
            Some("second")
        );
    }

    #[test]
    fn template_has_no_slots_by_default() {
        let t: RenderTemplate = RenderTemplate::new("bare");
        assert!(t.slots().is_empty());
        assert!(t.slot("anything").is_none());
    }

    #[test]
    fn with_slot_preserves_declaration_order() {
        let t: RenderTemplate = RenderTemplate::new("campfire")
            .with_slot("intensity", SlotKind::F32)
            .with_slot("tint", SlotKind::Color)
            .with_slot("lit", SlotKind::Bool);

        let names: Vec<&str> = t.slots().iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["intensity", "tint", "lit"]);

        let kinds: Vec<SlotKind> = t.slots().iter().map(|s| s.kind).collect();
        assert_eq!(kinds, vec![SlotKind::F32, SlotKind::Color, SlotKind::Bool]);
    }

    #[test]
    fn slot_lookup_by_name() {
        let t: RenderTemplate = RenderTemplate::new("tree")
            .with_slot("trunk_height", SlotKind::F32)
            .with_slot("leaf_tint", SlotKind::Color);

        assert_eq!(t.slot("trunk_height").map(|s| s.kind), Some(SlotKind::F32));
        assert_eq!(t.slot("leaf_tint").map(|s| s.kind), Some(SlotKind::Color));
        assert!(t.slot("missing").is_none());
    }

    #[test]
    #[should_panic(expected = "already has a slot named")]
    fn duplicate_slot_name_panics() {
        let _: RenderTemplate = RenderTemplate::new("bad")
            .with_slot("intensity", SlotKind::F32)
            .with_slot("intensity", SlotKind::Color);
    }

    #[test]
    fn slot_value_kind_matches_variant() {
        assert_eq!(SlotValue::F32(1.0).kind(), SlotKind::F32);
        assert_eq!(SlotValue::Vec2(Vec2::ZERO).kind(), SlotKind::Vec2);
        assert_eq!(SlotValue::Vec3(Vec3::ZERO).kind(), SlotKind::Vec3);
        assert_eq!(SlotValue::Vec4(Vec4::ZERO).kind(), SlotKind::Vec4);
        assert_eq!(SlotValue::Color(Vec4::ONE).kind(), SlotKind::Color);
        assert_eq!(SlotValue::Bool(true).kind(), SlotKind::Bool);
        assert_eq!(SlotValue::I32(-1).kind(), SlotKind::I32);
        assert_eq!(SlotValue::U32(7).kind(), SlotKind::U32);
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
    fn template_carries_slots_and_parts_together() {
        let t: RenderTemplate<TestMesh, TestMat> = RenderTemplate::new("torch")
            .with_slot("intensity", SlotKind::F32)
            .with_mesh_part(TestMesh::Cube, TestMat::Wood, Mat4::IDENTITY);

        assert_eq!(t.slots().len(), 1);
        assert_eq!(t.mesh_parts().len(), 1);
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

        let template = reg.get(RId::Campfire).expect("registered");
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
        assert_eq!(parts[0].slot, TestEmitterSlot::Main);
        assert_eq!(parts[1].template, TestEmitter::Smoke);
        assert_eq!(parts[1].slot, TestEmitterSlot::Secondary);
        assert_eq!(parts[1].local_transform.col(3).truncate().y, 0.85);
    }

    #[test]
    fn template_can_carry_all_part_kinds() {
        let t: RenderTemplate<TestMesh, TestMat, TestEmitter, TestEmitterSlot> =
            RenderTemplate::new("rich")
                .with_slot("intensity", SlotKind::F32)
                .with_mesh_part(TestMesh::Cube, TestMat::Wood, Mat4::IDENTITY)
                .with_emitter_part(TestEmitter::Flame, TestEmitterSlot::Main, Mat4::IDENTITY);

        assert_eq!(t.slots().len(), 1);
        assert_eq!(t.mesh_parts().len(), 1);
        assert_eq!(t.emitter_parts().len(), 1);
    }
}
