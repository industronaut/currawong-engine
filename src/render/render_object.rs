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
//! Mesh hierarchies, material references, and the live-instance side land
//! in subsequent steps. Re-registering an id silently replaces the previous
//! template — same idiom as
//! [`EmitterReconciler::register_template`](super::EmitterReconciler::register_template)
//! and [`InstanceBuckets::register`](super::InstanceBuckets::register).

use std::collections::HashMap;
use std::hash::Hash;

use glam::{Vec2, Vec3, Vec4};

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

/// Static template describing a renderable object. Many sim objects may
/// reference one template (every oak tree → `tree_oak`); per-instance
/// variation lives in transforms and slot values.
///
/// Templates are built with the [`Self::new`] / [`Self::with_slot`] builder
/// pair. Mesh hierarchy, material references, and visual-behaviour bindings
/// land in subsequent steps as the system grows.
#[derive(Clone, Debug)]
pub struct RenderTemplate {
    /// Human-readable name. Used in panics, logs, and tracing.
    pub label: String,
    slots: Vec<SlotDescriptor>,
}

impl RenderTemplate {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            slots: Vec::new(),
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

    /// All slots in declaration order.
    pub fn slots(&self) -> &[SlotDescriptor] {
        &self.slots
    }

    /// Look up a slot descriptor by name.
    pub fn slot(&self, name: &str) -> Option<&SlotDescriptor> {
        self.slots.iter().find(|s| s.name == name)
    }
}

/// Registry of [`RenderTemplate`]s keyed by a user-chosen id type `R`.
///
/// Generic over `R` so callers pick the id flavour: a small enum for
/// hand-authored content (`enum RenderId { TreeOak, Campfire }`) or a
/// numeric/asset handle later when templates are data-driven. The
/// registry is the same shape either way.
pub struct RenderRegistry<R>
where
    R: Copy + Eq + Hash,
{
    templates: HashMap<R, RenderTemplate>,
}

impl<R> RenderRegistry<R>
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
    pub fn register(&mut self, id: R, template: RenderTemplate) {
        self.templates.insert(id, template);
    }

    /// Look up the template registered under `id`, if any.
    pub fn get(&self, id: R) -> Option<&RenderTemplate> {
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

impl<R> Default for RenderRegistry<R>
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
        let mut reg = RenderRegistry::new();
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
        let mut reg = RenderRegistry::new();
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
        let t = RenderTemplate::new("bare");
        assert!(t.slots().is_empty());
        assert!(t.slot("anything").is_none());
    }

    #[test]
    fn with_slot_preserves_declaration_order() {
        let t = RenderTemplate::new("campfire")
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
        let t = RenderTemplate::new("tree")
            .with_slot("trunk_height", SlotKind::F32)
            .with_slot("leaf_tint", SlotKind::Color);

        assert_eq!(t.slot("trunk_height").map(|s| s.kind), Some(SlotKind::F32));
        assert_eq!(t.slot("leaf_tint").map(|s| s.kind), Some(SlotKind::Color));
        assert!(t.slot("missing").is_none());
    }

    #[test]
    #[should_panic(expected = "already has a slot named")]
    fn duplicate_slot_name_panics() {
        let _ = RenderTemplate::new("bad")
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
}
