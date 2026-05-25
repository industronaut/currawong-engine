//! Graft-from-template editor action — deep-copy another kind's node
//! tree into the current kind's template.
//!
//! The Godot-style "make local" model from CLAUDE.md: nodes are copied
//! by value with fresh per-template [`NodeId`]s, parent links remapped
//! through a `src_id → dest_id` table. There's no live reference to the
//! source template — later edits to the source don't propagate. Mesh /
//! Emitter payloads are clone-shared, so a grafted Mesh node referencing
//! the source kind's body mesh draws that body mesh (same
//! [`MeshKey`](crate::MeshKey) flows through the existing
//! `mesh_templates` lookup).
//!
//! Synchronous — no [`Renderer`] needed because no new GPU resources are
//! allocated. The Mesh payload's [`MeshKey`] already resolves through
//! the existing entry the source kind registered at init.
//!
//! Self-graft is a no-op: copying a tree into itself would expand it
//! unboundedly under repeated clicks, so we early-return.

use std::collections::HashMap;

use currawong::data::KindId;
use currawong::{NodeId, TemplateNode};

use crate::{LumberEditorView, MeshKey};

impl LumberEditorView {
    /// Deep-copy `src_kind`'s template nodes into `dest_kind`'s template,
    /// parenting the cloned roots under [`Self::selected_node`] (or as
    /// roots if no node is selected or the selection isn't in the dest
    /// template). Updates the selection to the last root of the copied
    /// subtree.
    ///
    /// Returns the number of nodes added. Zero means the source or dest
    /// template was missing, or the operation was a self-graft.
    pub(crate) fn graft_from_template(&mut self, src_kind: &KindId, dest_kind: &KindId) -> usize {
        if src_kind == dest_kind {
            return 0;
        }

        // Snapshot the source nodes so the immutable borrow on
        // self.templates ends before we take a mutable one for dest.
        let src_nodes: Vec<TemplateNode<MeshKey, MeshKey>> = {
            let Some(src_template) = self.templates.get(src_kind) else {
                return 0;
            };
            src_template.nodes().to_vec()
        };
        if src_nodes.is_empty() {
            return 0;
        }

        let parent_in_dest = self.selected_node;
        let Some(dest_template) = self.templates.get_mut(dest_kind) else {
            return 0;
        };
        // Validate the captured selection against the dest template's
        // live ids — selection may belong to a different kind if the
        // user hasn't clicked into this one yet.
        let parent_in_dest = parent_in_dest.filter(|p| dest_template.node(*p).is_some());

        let mut id_remap: HashMap<NodeId, NodeId> = HashMap::new();
        let mut last_root_dest: Option<NodeId> = None;
        let mut count = 0usize;

        for src_node in &src_nodes {
            let new_id = dest_template.next_free_node_id();
            let new_parent = match src_node.parent {
                None => parent_in_dest,
                Some(src_pid) => id_remap.get(&src_pid).copied(),
            };
            dest_template.add_node(TemplateNode {
                id: new_id,
                name: src_node.name.clone(),
                parent: new_parent,
                local_transform: src_node.local_transform,
                kind: src_node.kind.clone(),
            });
            id_remap.insert(src_node.id, new_id);
            if src_node.parent.is_none() {
                last_root_dest = Some(new_id);
            }
            count += 1;
        }

        if let Some(id) = last_root_dest {
            self.selected_node = Some(id);
            self.tree_view_state.set_selected(vec![Some(id)]);
        }
        count
    }
}
