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

use super::visibility::Aabb;

/// Reserved slot name for instance-root visibility. Every template
/// implicitly honours this: if a parent object's [`SlotValues`] has
/// `(VISIBLE_SLOT, SlotValue::Bool(false))`, the engine's render-object
/// walk skips the entire instance — no mesh parts, no emitter parts, no
/// hit-ID reservation. Missing slot = visible. See the [`render_object_pass`](super::render_object_pass)
/// helpers for the gating logic.
///
/// Templates declaring a slot with this name and a non-[`SlotKind::Bool`]
/// kind panic at template-build time (see [`RenderTemplate::with_routed_slot`]).
pub const VISIBLE_SLOT: &str = "visible";

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

/// How a slot's value is delivered to a draw call. Declared per slot at
/// template-build time so the engine can pick the right packing strategy
/// without runtime inference.
///
/// `Instance` is the only routing currently implemented. `Uniform` is a
/// doc-only reservation: [`RenderTemplate::with_routed_slot`] panics if
/// asked for it, so the variant cannot reach a draw call. It exists so
/// that adding the packing path later — gather across instances, allocate
/// the uniform array buffer, write the values, bind to the pipeline,
/// index by `instance_index` in shaders — doesn't break this enum's API.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum SlotRouting {
    /// Packed into the per-instance attribute buffer once per draw. Right
    /// for values that vary per instance and change frequently
    /// (transform-adjacent floats, per-instance colours).
    #[default]
    Instance,
    /// **Not yet implemented — reserved for the future.** Will pack values
    /// into a per-instance uniform array indexed by `instance_index`, for
    /// larger payloads that don't fit cleanly into a vertex-attribute slot.
    /// [`RenderTemplate::with_routed_slot`] rejects this variant today.
    Uniform,
}

/// Schema entry for one slot on a [`RenderTemplate`]: a name plus its kind
/// plus its [`SlotRouting`]. Slots are stored in declaration order so
/// consumers can derive stable uniform / instance-buffer layouts from them.
#[derive(Clone, Debug)]
pub struct SlotDescriptor {
    pub name: String,
    pub kind: SlotKind,
    pub routing: SlotRouting,
}

/// Named [`SlotValue`]s for one render instance. The sim provides these
/// per object (typically as a component keyed by `WorldObjectId`); the View
/// reads them at render time to drive per-instance attribs, material
/// instance selection, visual-script state, and so on. Templates declare
/// the *schema* — names and kinds — via [`RenderTemplate::with_slot`];
/// `SlotValues` is the *data* matching that schema.
///
/// Storage is a `HashMap` keyed by `&'static str`. Slot names are
/// template-declared string literals at the call site, so the keys never
/// outlive the program and no per-value `String` allocation is needed.
/// Lookup is O(1). Iteration order is non-deterministic, matching the
/// `HashMap`-backed [`Components`](crate::sim::components::Components)
/// registry.
#[derive(Clone, Debug, Default)]
pub struct SlotValues {
    values: HashMap<&'static str, SlotValue>,
}

impl SlotValues {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Set the value for `name`, replacing any existing entry. Returns
    /// `self` for chained builder use at sim-init time.
    pub fn with(mut self, name: &'static str, value: SlotValue) -> Self {
        self.set(name, value);
        self
    }

    /// In-place set / overwrite.
    pub fn set(&mut self, name: &'static str, value: SlotValue) {
        self.values.insert(name, value);
    }

    /// Look up a value by name. Returns the `SlotValue` by value (it's
    /// `Copy`); the caller `match`es to extract the typed inner.
    pub fn get(&self, name: &str) -> Option<SlotValue> {
        self.values.get(name).copied()
    }

    /// `(name, value)` iterator. Order is unspecified.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, SlotValue)> + '_ {
        self.values.iter().map(|(&n, &v)| (n, v))
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
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
    /// Optional binding to a `SlotKind::Bool` slot on the parent template.
    /// When set, the engine's render-object walk skips this part if the
    /// parent's [`SlotValues`] has `(name, SlotValue::Bool(false))`. Missing
    /// slot = visible (same convention as [`VISIBLE_SLOT`]).
    ///
    /// Construct via [`RenderTemplate::with_mesh_part_gated`] so the
    /// template-build-time check that the named slot exists and is `Bool`
    /// runs; bypassing that builder shifts the failure mode to a
    /// debug-build slot-validation panic when the part is first walked.
    pub visibility_slot: Option<&'static str>,
}

impl<M, MK> MeshPart<M, MK> {
    pub fn new(mesh: M, material: MK, local_transform: Mat4) -> Self {
        Self {
            mesh,
            material,
            local_transform,
            visibility_slot: None,
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
///
/// `attachment` is distinct from a [`SlotKind`] template slot — it's an
/// emitter-keying id chosen by the user, not a typed template parameter.
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
    visual_bounds: Option<Aabb>,
}

impl<M, MK, E, S> RenderTemplate<M, MK, E, S> {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            slots: Vec::new(),
            mesh_parts: Vec::new(),
            emitter_parts: Vec::new(),
            visual_bounds: None,
        }
    }

    /// Declare an instance-routed slot. Slot names must be unique within a
    /// template; passing a duplicate panics — programming error, not
    /// runtime condition. Returns `self` for chaining.
    ///
    /// Equivalent to [`Self::with_routed_slot`] with
    /// [`SlotRouting::Instance`]; covers the common case where a slot
    /// drives per-instance attribs.
    pub fn with_slot(self, name: impl Into<String>, kind: SlotKind) -> Self {
        self.with_routed_slot(name, kind, SlotRouting::Instance)
    }

    /// Declare a slot with an explicit [`SlotRouting`]. Slot names must be
    /// unique within a template; passing a duplicate panics — programming
    /// error, not runtime condition. Returns `self` for chaining.
    ///
    /// [`SlotRouting::Uniform`] is not yet implemented and is rejected here
    /// at template-build time, so the failure surfaces at the API boundary
    /// rather than as a draw-time trap. Use [`SlotRouting::Instance`] until
    /// the uniform packing path lands.
    pub fn with_routed_slot(
        mut self,
        name: impl Into<String>,
        kind: SlotKind,
        routing: SlotRouting,
    ) -> Self {
        let name = name.into();
        assert!(
            routing != SlotRouting::Uniform,
            "RenderTemplate '{}' slot '{}' requested SlotRouting::Uniform, \
             which is not yet implemented — declare it as SlotRouting::Instance \
             until uniform-routed packing lands.",
            self.label,
            name,
        );
        assert!(
            !(name == VISIBLE_SLOT && kind != SlotKind::Bool),
            "RenderTemplate '{}' declared reserved slot '{}' with kind {:?}; \
             VISIBLE_SLOT is reserved for instance-root visibility and must be SlotKind::Bool.",
            self.label,
            name,
            kind,
        );
        assert!(
            !self.slots.iter().any(|s| s.name == name),
            "RenderTemplate '{}' already has a slot named '{}'",
            self.label,
            name
        );
        self.slots.push(SlotDescriptor {
            name,
            kind,
            routing,
        });
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

    /// Add a [`MeshPart`] gated on a `SlotKind::Bool` slot. The engine
    /// skips this part when the parent's [`SlotValues`] has
    /// `(visibility_slot, SlotValue::Bool(false))`. Missing slot = visible.
    ///
    /// The named slot must already be declared on this template with
    /// [`SlotKind::Bool`] — typically via [`Self::with_slot`] earlier in
    /// the builder chain — or this call panics. [`VISIBLE_SLOT`] is
    /// accepted without an explicit declaration since it's reserved
    /// engine-wide, though gating a part on it is redundant with root
    /// visibility (which already short-circuits the entire instance).
    pub fn with_mesh_part_gated(
        mut self,
        mesh: M,
        material: MK,
        local_transform: Mat4,
        visibility_slot: &'static str,
    ) -> Self {
        if visibility_slot != VISIBLE_SLOT {
            let slot = self.slot(visibility_slot).unwrap_or_else(|| {
                panic!(
                    "RenderTemplate '{}' gated mesh part on slot '{}', \
                     but no such slot is declared — call with_slot('{}', SlotKind::Bool) first.",
                    self.label, visibility_slot, visibility_slot,
                )
            });
            assert!(
                slot.kind == SlotKind::Bool,
                "RenderTemplate '{}' gated mesh part on slot '{}' of kind {:?}; \
                 per-part visibility slots must be SlotKind::Bool.",
                self.label,
                visibility_slot,
                slot.kind,
            );
        }
        let mut part = MeshPart::new(mesh, material, local_transform);
        part.visibility_slot = Some(visibility_slot);
        self.mesh_parts.push(part);
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

    /// Set the template's *visual* AABB — the region of space the template
    /// occupies when rendered, including emitter reach and other effects.
    /// Used by [`LiveRenderObjects::cull`](super::LiveRenderObjects::cull); a
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
    fn with_slot_defaults_to_instance_routing() {
        let t: RenderTemplate = RenderTemplate::new("oak")
            .with_slot("height", SlotKind::F32)
            .with_slot("tint", SlotKind::Color);

        let routings: Vec<SlotRouting> = t.slots().iter().map(|s| s.routing).collect();
        assert_eq!(
            routings,
            vec![SlotRouting::Instance, SlotRouting::Instance],
            "with_slot must default routing to Instance",
        );
    }

    #[test]
    fn with_routed_slot_records_routing() {
        let t: RenderTemplate = RenderTemplate::new("torch")
            .with_routed_slot("brightness", SlotKind::F32, SlotRouting::Instance)
            .with_routed_slot("tint", SlotKind::Color, SlotRouting::Instance);

        let routings: Vec<SlotRouting> = t.slots().iter().map(|s| s.routing).collect();
        assert_eq!(routings, vec![SlotRouting::Instance, SlotRouting::Instance]);
    }

    #[test]
    #[should_panic(expected = "SlotRouting::Uniform")]
    fn with_routed_slot_uniform_panics() {
        let _: RenderTemplate = RenderTemplate::new("torch").with_routed_slot(
            "brightness",
            SlotKind::F32,
            SlotRouting::Uniform,
        );
    }

    #[test]
    fn slot_routing_default_is_instance() {
        assert_eq!(SlotRouting::default(), SlotRouting::Instance);
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
        assert_eq!(parts[0].attachment, TestEmitterSlot::Main);
        assert_eq!(parts[1].template, TestEmitter::Smoke);
        assert_eq!(parts[1].attachment, TestEmitterSlot::Secondary);
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

    #[test]
    fn slot_values_empty_by_default() {
        let sv = SlotValues::new();
        assert!(sv.is_empty());
        assert_eq!(sv.len(), 0);
        assert!(sv.get("anything").is_none());
    }

    #[test]
    fn slot_values_with_and_get() {
        let sv = SlotValues::new()
            .with("intensity", SlotValue::F32(0.7))
            .with("tint", SlotValue::Color(Vec4::new(1.0, 0.5, 0.2, 1.0)));

        assert_eq!(sv.len(), 2);
        assert_eq!(sv.get("intensity"), Some(SlotValue::F32(0.7)));
        assert_eq!(
            sv.get("tint"),
            Some(SlotValue::Color(Vec4::new(1.0, 0.5, 0.2, 1.0)))
        );
        assert!(sv.get("missing").is_none());
    }

    #[test]
    fn slot_values_set_overwrites() {
        let mut sv = SlotValues::new();
        sv.set("intensity", SlotValue::F32(0.5));
        sv.set("intensity", SlotValue::F32(0.9));

        assert_eq!(sv.len(), 1, "set must overwrite, not append");
        assert_eq!(sv.get("intensity"), Some(SlotValue::F32(0.9)));
    }

    #[test]
    fn mesh_part_visibility_slot_defaults_to_none() {
        let part: MeshPart<TestMesh, TestMat> =
            MeshPart::new(TestMesh::Cube, TestMat::Wood, Mat4::IDENTITY);
        assert!(part.visibility_slot.is_none());
    }

    #[test]
    fn with_mesh_part_gated_records_visibility_slot() {
        let t: RenderTemplate<TestMesh, TestMat> = RenderTemplate::new("pawn")
            .with_slot("carrying", SlotKind::Bool)
            .with_mesh_part_gated(TestMesh::Cube, TestMat::Wood, Mat4::IDENTITY, "carrying");

        let parts = t.mesh_parts();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].visibility_slot, Some("carrying"));
    }

    #[test]
    fn with_mesh_part_gated_accepts_visible_slot_without_declaration() {
        let t: RenderTemplate<TestMesh, TestMat> = RenderTemplate::new("rune")
            .with_mesh_part_gated(TestMesh::Cube, TestMat::Wood, Mat4::IDENTITY, VISIBLE_SLOT);

        assert_eq!(t.mesh_parts()[0].visibility_slot, Some(VISIBLE_SLOT));
    }

    #[test]
    #[should_panic(expected = "no such slot is declared")]
    fn with_mesh_part_gated_panics_when_slot_missing() {
        let _: RenderTemplate<TestMesh, TestMat> = RenderTemplate::new("pawn")
            .with_mesh_part_gated(TestMesh::Cube, TestMat::Wood, Mat4::IDENTITY, "carrying");
    }

    #[test]
    #[should_panic(expected = "per-part visibility slots must be SlotKind::Bool")]
    fn with_mesh_part_gated_panics_on_non_bool_slot() {
        let _: RenderTemplate<TestMesh, TestMat> = RenderTemplate::new("pawn")
            .with_slot("intensity", SlotKind::F32)
            .with_mesh_part_gated(TestMesh::Cube, TestMat::Wood, Mat4::IDENTITY, "intensity");
    }

    #[test]
    #[should_panic(expected = "VISIBLE_SLOT is reserved")]
    fn declaring_visible_slot_with_non_bool_kind_panics() {
        let _: RenderTemplate = RenderTemplate::new("bad").with_slot(VISIBLE_SLOT, SlotKind::F32);
    }

    #[test]
    fn declaring_visible_slot_with_bool_kind_is_allowed() {
        let t: RenderTemplate = RenderTemplate::new("ok").with_slot(VISIBLE_SLOT, SlotKind::Bool);
        assert_eq!(t.slot(VISIBLE_SLOT).map(|s| s.kind), Some(SlotKind::Bool));
    }

    #[test]
    fn slot_values_iter_visits_every_entry() {
        let sv = SlotValues::new()
            .with("a", SlotValue::I32(1))
            .with("b", SlotValue::I32(2))
            .with("c", SlotValue::I32(3));

        // Iteration order is unspecified — sort by name for a deterministic check.
        let mut entries: Vec<(&'static str, SlotValue)> = sv.iter().collect();
        entries.sort_by_key(|(n, _)| *n);
        assert_eq!(
            entries,
            vec![
                ("a", SlotValue::I32(1)),
                ("b", SlotValue::I32(2)),
                ("c", SlotValue::I32(3)),
            ]
        );
    }
}
