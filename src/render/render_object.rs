//! Render-object templates and their registry.
//!
//! A render object is a view-side template — a hierarchy of named nodes
//! that hold meshes, emitters, or empty attachment transforms — instanced
//! when the camera shows a sim object that names it. Templates are
//! identified by a user-chosen id type `R` (`Copy + Eq + Hash`;
//! conventionally named `RenderId` at the call site) and stored in a
//! [`RenderRegistry`] owned by the `View`.
//!
//! Each [`TemplateNode`] carries a stable [`NodeId`] (assigned at editor
//! time and persisted in the template definition), a name, a parent link,
//! a local transform, and a [`NodeKind`] payload — `Empty` for pure
//! attachment / grouping nodes, `Mesh(MeshPart)` for static geometry,
//! `Emitter(EmitterPart)` for particle attachments. World transforms and
//! visibility cascade down the tree once per frame; per-instance state
//! lives on the persistent [`RenderProxy`](super::RenderProxy), indexed by
//! the dense slot the template assigns each node at build time.
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

/// Stable per-template identifier for a [`TemplateNode`]. Authored at
/// editor time, written into the template's definition file, and never
/// reused on delete — so consumer code addressing a node by id never
/// silently rebinds to a different node after a structural edit.
///
/// Per-template id space: a `NodeId(7)` in template A means nothing in
/// template B. Range is `u16` (64k slots per template), which is more than
/// any single template needs in practice. Holes left by deletions are
/// fine; compaction is a separate editor concern.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct NodeId(pub u16);

/// One drawable piece of geometry referenced by a [`TemplateNode`] whose
/// kind is `Mesh`: a mesh handle and the material key it draws with.
///
/// `M` is the user's mesh-handle type (commonly a small enum like
/// `enum MeshId { Cube, Tetra }`); `MK` is the user's material-instance
/// key (commonly `enum MaterialKey { Wood, Stone }`). The engine doesn't
/// own meshes or material instances — it stores these handles and the
/// View resolves them against its own tables when rendering.
///
/// The transform from the part's local frame up to the template root is
/// owned by the [`TemplateNode`], not the part — node-owned transforms
/// make tree traversal cleanly composable.
#[derive(Clone, Debug)]
pub struct MeshPart<M, MK> {
    pub mesh: M,
    pub material: MK,
}

impl<M, MK> MeshPart<M, MK> {
    pub fn new(mesh: M, material: MK) -> Self {
        Self { mesh, material }
    }
}

/// An emitter attachment referenced by a [`TemplateNode`] whose kind is
/// `Emitter`: which emitter template `E` to spawn and which `S` attachment
/// id keys it (so one template can carry several emitters keyed
/// independently — e.g. flame + smoke + sparks).
///
/// The View resolves `E` and `S` against an
/// [`EmitterReconciler<E, S>`](super::EmitterReconciler), which owns the
/// emitter lifecycle and particle integration. The render-object system
/// only declares attachments; the reconciler handles state.
///
/// As with [`MeshPart`], the local transform lives on the parent
/// [`TemplateNode`].
#[derive(Clone, Debug)]
pub struct EmitterPart<E, S> {
    pub template: E,
    pub attachment: S,
}

impl<E, S> EmitterPart<E, S> {
    pub fn new(template: E, attachment: S) -> Self {
        Self {
            template,
            attachment,
        }
    }
}

/// The payload carried by a [`TemplateNode`]: empty (transform-only
/// attachment / grouping), a mesh part, or an emitter part.
///
/// Empty nodes don't draw anything themselves but are still walked by
/// the tree traversal — useful for named attachment points (where the
/// editor parents emitters or grafted child geometry) and for grouping
/// nodes that should hide together via the visibility cascade.
#[derive(Clone, Debug)]
pub enum NodeKind<M = (), MK = (), E = (), S = ()> {
    Empty,
    Mesh(MeshPart<M, MK>),
    Emitter(EmitterPart<E, S>),
}

/// One node in a [`RenderTemplate`]'s hierarchy: a stable id, a display
/// name, a parent link (or `None` for a root), a local transform up to
/// the parent's frame, and a [`NodeKind`] payload.
///
/// Built directly via field syntax or one of the [`TemplateNode::empty`] /
/// [`TemplateNode::mesh`] / [`TemplateNode::emitter`] constructors and fed
/// into [`RenderTemplate::with_node`].
#[derive(Clone, Debug)]
pub struct TemplateNode<M = (), MK = (), E = (), S = ()> {
    pub id: NodeId,
    pub name: String,
    pub parent: Option<NodeId>,
    /// Transform from this node's local frame up to its parent's local
    /// frame (or up to the template root if `parent` is `None`). World
    /// transform of a drawn part is the composed product down the tree
    /// times the proxy's world transform.
    pub local_transform: Mat4,
    pub kind: NodeKind<M, MK, E, S>,
}

impl<M, MK, E, S> TemplateNode<M, MK, E, S> {
    /// Construct an [`NodeKind::Empty`] node — a pure transform /
    /// attachment point with no draw payload.
    pub fn empty(
        id: NodeId,
        name: impl Into<String>,
        parent: Option<NodeId>,
        local_transform: Mat4,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            parent,
            local_transform,
            kind: NodeKind::Empty,
        }
    }

    /// Construct an [`NodeKind::Mesh`] node from a [`MeshPart`].
    pub fn mesh(
        id: NodeId,
        name: impl Into<String>,
        parent: Option<NodeId>,
        local_transform: Mat4,
        part: MeshPart<M, MK>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            parent,
            local_transform,
            kind: NodeKind::Mesh(part),
        }
    }

    /// Construct an [`NodeKind::Emitter`] node from an [`EmitterPart`].
    pub fn emitter(
        id: NodeId,
        name: impl Into<String>,
        parent: Option<NodeId>,
        local_transform: Mat4,
        part: EmitterPart<E, S>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            parent,
            local_transform,
            kind: NodeKind::Emitter(part),
        }
    }
}

/// Static template describing a renderable object. Many sim objects may
/// reference one template (every oak tree → `tree_oak`); per-instance
/// variation lives on the [`RenderProxy`](super::RenderProxy)'s per-node
/// state, written by the update hook each frame.
///
/// Internally the template stores three things derived from the node
/// list: a dense `Vec<TemplateNode>` (the canonical storage),
/// `id_to_slot` (`Vec<Option<u32>>` indexed by raw `NodeId`, for O(1) id →
/// slot lookup), and `children_slots` parallel to the nodes (for tree
/// traversal). All three are kept consistent by [`Self::with_node`].
///
/// `M`, `MK`, `E`, `S` default to `()` for callers that don't need the
/// corresponding part kind yet.
#[derive(Clone, Debug)]
pub struct RenderTemplate<M = (), MK = (), E = (), S = ()> {
    /// Human-readable name. Used in panics, logs, and tracing.
    pub label: String,
    nodes: Vec<TemplateNode<M, MK, E, S>>,
    /// Sparse: `id_to_slot[id as usize] == Some(slot)` for live nodes.
    id_to_slot: Vec<Option<u32>>,
    /// Parallel to `nodes`: `children_slots[slot]` lists the slots of
    /// direct children, in declaration order.
    children_slots: Vec<Vec<u32>>,
    /// Slots whose nodes have `parent: None`. Traversal walks from here.
    root_slots: Vec<u32>,
    visual_bounds: Option<Aabb>,
}

impl<M, MK, E, S> RenderTemplate<M, MK, E, S> {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            nodes: Vec::new(),
            id_to_slot: Vec::new(),
            children_slots: Vec::new(),
            root_slots: Vec::new(),
            visual_bounds: None,
        }
    }

    /// Add `node` to the template. Panics if its [`NodeId`] is already in
    /// use, or if its `parent` references an id this template doesn't
    /// have — nodes must be added parent-first.
    pub fn with_node(mut self, node: TemplateNode<M, MK, E, S>) -> Self {
        let id = node.id;
        let slot = self.nodes.len() as u32;

        let id_idx = id.0 as usize;
        if id_idx >= self.id_to_slot.len() {
            self.id_to_slot.resize(id_idx + 1, None);
        }
        assert!(
            self.id_to_slot[id_idx].is_none(),
            "RenderTemplate `{}`: NodeId({}) is already in use",
            self.label,
            id.0
        );

        match node.parent {
            None => self.root_slots.push(slot),
            Some(parent_id) => {
                let parent_slot = self
                    .slot_of(parent_id)
                    .unwrap_or_else(|| {
                        panic!(
                            "RenderTemplate `{}`: NodeId({}) parents NodeId({}) which hasn't been added yet",
                            self.label, id.0, parent_id.0,
                        )
                    });
                self.children_slots[parent_slot as usize].push(slot);
            }
        }

        self.id_to_slot[id_idx] = Some(slot);
        self.children_slots.push(Vec::new());
        self.nodes.push(node);
        self
    }

    /// Convenience: add a [`MeshPart`] as a root-level [`NodeKind::Mesh`]
    /// node, allocating the next unused [`NodeId`]. Carries the call
    /// sites that don't author hierarchy explicitly.
    pub fn with_mesh_part(self, mesh: M, material: MK, local_transform: Mat4) -> Self {
        let id = self.next_free_node_id();
        let name = format!("mesh_{}", id.0);
        self.with_node(TemplateNode::mesh(
            id,
            name,
            None,
            local_transform,
            MeshPart::new(mesh, material),
        ))
    }

    /// Convenience: add an [`EmitterPart`] as a root-level
    /// [`NodeKind::Emitter`] node, allocating the next unused
    /// [`NodeId`].
    pub fn with_emitter_part(self, template: E, attachment: S, local_transform: Mat4) -> Self {
        let id = self.next_free_node_id();
        let name = format!("emitter_{}", id.0);
        self.with_node(TemplateNode::emitter(
            id,
            name,
            None,
            local_transform,
            EmitterPart::new(template, attachment),
        ))
    }

    /// All nodes in declaration order. Slot indices are this slice's
    /// indices.
    pub fn nodes(&self) -> &[TemplateNode<M, MK, E, S>] {
        &self.nodes
    }

    /// Dense slot for `id`, or `None` if the template has no such node.
    /// The dense slot indexes [`Self::nodes`], [`Self::children`], and
    /// the parallel `RenderProxy::nodes` vec.
    pub fn slot_of(&self, id: NodeId) -> Option<u32> {
        self.id_to_slot.get(id.0 as usize).copied().flatten()
    }

    /// Borrow the node with id `id`, if any.
    pub fn node(&self, id: NodeId) -> Option<&TemplateNode<M, MK, E, S>> {
        let slot = self.slot_of(id)?;
        self.nodes.get(slot as usize)
    }

    /// Slots whose nodes have no parent.
    pub fn roots(&self) -> &[u32] {
        &self.root_slots
    }

    /// Direct children of the node at `slot`, in declaration order.
    pub fn children(&self, slot: u32) -> &[u32] {
        &self.children_slots[slot as usize]
    }

    /// Total number of nodes — also the size of the per-instance
    /// `RenderProxy::nodes` vec built from this template.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
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

    /// Lowest unused [`NodeId`] in this template. Used by the sugar
    /// builders ([`Self::with_mesh_part`] / [`Self::with_emitter_part`])
    /// and by editor / glTF-import paths that don't have author-supplied
    /// ids to assign.
    pub fn next_free_node_id(&self) -> NodeId {
        for (i, slot) in self.id_to_slot.iter().enumerate() {
            if slot.is_none() {
                return NodeId(i as u16);
            }
        }
        let next = self.id_to_slot.len();
        assert!(
            next <= u16::MAX as usize,
            "RenderTemplate `{}`: NodeId space exhausted (u16::MAX nodes)",
            self.label,
        );
        NodeId(next as u16)
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
    fn template_has_no_nodes_by_default() {
        let t: RenderTemplate<TestMesh, TestMat> = RenderTemplate::new("bare");
        assert!(t.nodes().is_empty());
        assert!(t.roots().is_empty());
        assert_eq!(t.node_count(), 0);
    }

    #[test]
    fn with_mesh_part_creates_root_mesh_node() {
        let t: RenderTemplate<TestMesh, TestMat> = RenderTemplate::new("campfire")
            .with_mesh_part(TestMesh::Cube, TestMat::Wood, Mat4::IDENTITY)
            .with_mesh_part(
                TestMesh::Plane,
                TestMat::Stone,
                Mat4::from_translation(Vec3::new(0.0, -0.5, 0.0)),
            );

        let nodes = t.nodes();
        assert_eq!(nodes.len(), 2);
        match &nodes[0].kind {
            NodeKind::Mesh(p) => {
                assert_eq!(p.mesh, TestMesh::Cube);
                assert_eq!(p.material, TestMat::Wood);
            }
            _ => panic!("expected mesh node"),
        }
        assert_eq!(nodes[1].local_transform.col(3).truncate().y, -0.5);
        assert_eq!(t.roots(), &[0, 1]);
    }

    #[test]
    fn with_node_builds_hierarchy() {
        let root = NodeId(0);
        let mesh = NodeId(1);
        let attach = NodeId(2);
        let t: RenderTemplate<TestMesh, TestMat> = RenderTemplate::new("rig")
            .with_node(TemplateNode::empty(root, "root", None, Mat4::IDENTITY))
            .with_node(TemplateNode::mesh(
                mesh,
                "body",
                Some(root),
                Mat4::IDENTITY,
                MeshPart::new(TestMesh::Cube, TestMat::Wood),
            ))
            .with_node(TemplateNode::empty(
                attach,
                "attach",
                Some(root),
                Mat4::from_translation(Vec3::Z),
            ));

        assert_eq!(t.slot_of(root), Some(0));
        assert_eq!(t.slot_of(mesh), Some(1));
        assert_eq!(t.slot_of(attach), Some(2));
        assert_eq!(t.slot_of(NodeId(99)), None);
        assert_eq!(t.roots(), &[0]);
        assert_eq!(t.children(0), &[1, 2]);
        assert!(t.children(1).is_empty());
    }

    #[test]
    #[should_panic(expected = "NodeId(0) is already in use")]
    fn duplicate_node_id_panics() {
        let _ = RenderTemplate::<TestMesh, TestMat>::new("dup")
            .with_node(TemplateNode::empty(NodeId(0), "a", None, Mat4::IDENTITY))
            .with_node(TemplateNode::empty(NodeId(0), "b", None, Mat4::IDENTITY));
    }

    #[test]
    #[should_panic(expected = "parents NodeId(7) which hasn't been added")]
    fn missing_parent_panics() {
        let _ = RenderTemplate::<TestMesh, TestMat>::new("bad").with_node(TemplateNode::empty(
            NodeId(0),
            "orphan",
            Some(NodeId(7)),
            Mat4::IDENTITY,
        ));
    }

    #[test]
    fn next_free_node_id_skips_holes_lowest_first() {
        let t: RenderTemplate<TestMesh, TestMat> = RenderTemplate::new("t")
            .with_node(TemplateNode::empty(NodeId(0), "a", None, Mat4::IDENTITY))
            .with_node(TemplateNode::empty(NodeId(2), "c", None, Mat4::IDENTITY));
        // id 1 wasn't used → first free.
        assert_eq!(t.next_free_node_id(), NodeId(1));
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
    fn with_emitter_part_creates_root_emitter_node() {
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

        let nodes = t.nodes();
        assert_eq!(nodes.len(), 2);
        match &nodes[0].kind {
            NodeKind::Emitter(p) => {
                assert_eq!(p.template, TestEmitter::Flame);
                assert_eq!(p.attachment, TestEmitterSlot::Main);
            }
            _ => panic!("expected emitter node"),
        }
        assert_eq!(nodes[1].local_transform.col(3).truncate().y, 0.85);
    }

    #[test]
    fn template_can_carry_all_node_kinds() {
        let t: RenderTemplate<TestMesh, TestMat, TestEmitter, TestEmitterSlot> =
            RenderTemplate::new("rich")
                .with_mesh_part(TestMesh::Cube, TestMat::Wood, Mat4::IDENTITY)
                .with_emitter_part(TestEmitter::Flame, TestEmitterSlot::Main, Mat4::IDENTITY);

        assert_eq!(t.nodes().len(), 2);
        assert!(matches!(t.nodes()[0].kind, NodeKind::Mesh(_)));
        assert!(matches!(t.nodes()[1].kind, NodeKind::Emitter(_)));
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
