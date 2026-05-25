//! "Add mesh from glb" editor action.
//!
//! Two-phase like [`hot_reload`](crate::hot_reload) because the UI runs
//! in `ui()` (no [`Renderer`] available) but building a
//! [`MeshTemplate`] needs the renderer:
//!
//! - [`LumberEditorView::queue_glb_import`] (called from the scene
//!   panel) validates the path and parks the request in
//!   [`LumberEditorView::pending_glb_imports`].
//! - [`LumberEditorView::apply_pending_glb_imports`] (called from
//!   `shadow_pass` cascade 0) drains the queue, builds the
//!   [`MeshTemplate`], registers the bucket, and appends a
//!   [`NodeKind::Mesh`] node to the target template.
//!
//! The glb is consumed as a single visual unit — the whole glb decodes
//! through the existing flat
//! [`decode_gltf_mesh`](currawong::decode_gltf_mesh) path; its internal
//! node tree is collapsed. Importing a glb as a hierarchy of nodes
//! (`decode_gltf_scene`) is a separate follow-up — this one is the leaf
//! case that fits within the existing single-MeshTemplate-per-key model.

use currawong::data::{KindId, VfsPath};
use currawong::glam::Mat4;
use currawong::{NodeId, Renderer, TemplateNode, build_streamed_pbr_mesh_template};

use crate::{LumberEditorView, MeshKey};

/// Default texture for the PBR fallback material — same lumber atlas
/// the camp's body kinds use, so a grafted glb whose material slot
/// doesn't resolve through the atlas registry still draws with a
/// believable albedo instead of magenta.
const FALLBACK_ALBEDO_PATH: &str = "lumber/gradient_atlas.png";

/// PBR defaults for an editor-grafted glb. Match the
/// [`MeshNodeSpec`](currawong::MeshNodeSpec) defaults so a graft +
/// immediate save round-trips through the spec without changing the
/// material appearance.
const GRAFT_METALLIC: f32 = 0.0;
const GRAFT_ROUGHNESS: f32 = 0.85;

impl LumberEditorView {
    /// Queue a glb-import request for the named kind, allocating a fresh
    /// [`NodeId`] now so the caller knows which node will appear in the
    /// template next frame. The selection in the scene panel jumps to
    /// this id once the GPU phase runs.
    ///
    /// Returns the allocated [`NodeId`], or `None` if the kind has no
    /// registered template (e.g. its own glb failed to load) or the path
    /// is empty.
    pub(crate) fn queue_glb_import(
        &mut self,
        kind: &KindId,
        parent: Option<NodeId>,
        path: VfsPath,
    ) -> Option<NodeId> {
        let template = self.templates.get(kind)?;
        let new_id = template.next_free_node_id();
        // Validate the parent against this template; bad parent → root.
        let parent = parent.filter(|p| template.node(*p).is_some());
        self.pending_glb_imports
            .push((kind.clone(), new_id, parent, path));
        Some(new_id)
    }

    /// Drain [`Self::pending_glb_imports`] and apply each request: build
    /// the [`MeshTemplate`], register the bucket if it's a new key, and
    /// append the [`NodeKind::Mesh`] node to the kind's template. Called
    /// from `shadow_pass` cascade 0 once per frame.
    ///
    /// Idempotent on the no-pending-work path. If a target kind's
    /// template was retired between queue and apply (hot reload), the
    /// request is dropped silently — the editor's "graft is ephemeral"
    /// invariant absorbs this.
    pub(crate) fn apply_pending_glb_imports(&mut self, renderer: &Renderer) {
        if self.pending_glb_imports.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.pending_glb_imports);
        for (kind, node_id, parent, path) in pending {
            self.apply_one_glb_import(renderer, &kind, node_id, parent, path);
        }
    }

    fn apply_one_glb_import(
        &mut self,
        renderer: &Renderer,
        kind: &KindId,
        node_id: NodeId,
        parent: Option<NodeId>,
        path: VfsPath,
    ) {
        let key = MeshKey::Glb(path.clone());

        // Build the MeshTemplate once per unique glb path. Repeat imports
        // of the same path reuse the existing entry — the AssetServer
        // already de-dupes by path so the underlying mesh bytes load once
        // either way; this just avoids a duplicate material instance.
        if !self.mesh_templates.contains_key(&key) {
            let template = build_streamed_pbr_mesh_template(
                renderer,
                &self.material,
                &self.samplers,
                &self.asset_server,
                path.clone(),
                VfsPath::new(FALLBACK_ALBEDO_PATH).expect("valid fallback path"),
                GRAFT_METALLIC,
                GRAFT_ROUGHNESS,
            );
            self.mesh_templates.insert(key.clone(), template);
            self.buckets.register(&renderer.device, key.clone());
        }

        // Append the node to the target template. The id was reserved at
        // queue time; if the user has since deleted nodes such that this
        // id is no longer free, that's an inconsistency from a kind
        // switch / hot reload — drop the request.
        let Some(template) = self.templates.get_mut(kind) else {
            return;
        };
        if template.slot_of(node_id).is_some() {
            return;
        }
        // Use the path's basename as a friendly default name.
        let name = path
            .as_ref()
            .rsplit('/')
            .next()
            .unwrap_or(path.as_ref())
            .to_string();
        // Parent was captured at queue time and validated then; re-check
        // here in case a hot reload retired the id between queue and apply.
        let parent = parent.filter(|p| template.node(*p).is_some());
        template.add_node(TemplateNode::mesh(
            node_id,
            name,
            parent,
            Mat4::IDENTITY,
            currawong::MeshPart::new(key.clone(), key),
        ));
        self.selected_node = Some(node_id);
    }
}
