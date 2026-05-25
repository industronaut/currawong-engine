//! Scene panel — egui tree view of the selected kind's
//! [`RenderTemplate`] node hierarchy plus per-node TRS editing.
//!
//! Slots into [`crate::kind_panel`] as two sections rendered below the
//! existing "Bounding box" controls and above the bottom-anchored Save
//! button: the tree view + add/delete/rename actions (in
//! [`LumberEditorView::scene_section`]) and the selected-node inspector
//! with name + TRS DragValues (in
//! [`LumberEditorView::selected_node_section`]).
//!
//! Edits mutate the template held in [`LumberEditorView::templates`]
//! directly via [`RenderRegistry::get_mut`] / [`RenderTemplate::add_node`]
//! / [`RenderTemplate::remove_node`]. They're **ephemeral** — a hot
//! reload (or any rebuild via
//! [`LumberEditorView::maybe_rebuild_templates`](crate::LumberEditorView::maybe_rebuild_templates))
//! reconstructs the template from disk and discards anything authored
//! here. Persistence is a follow-up PR (Phase 6 — RON schema with
//! `nodes:` block).

use currawong::data::{KindId, VfsPath};
use currawong::egui_ltreeview::{Action, DirPosition, NodeBuilder, TreeView};
use currawong::glam::{EulerRot, Mat4, Quat, Vec3};
use currawong::{InsertPosition, NodeId, NodeKind, RenderTemplate, TemplateNode, egui};
use std::f32::consts::PI;

use crate::LumberEditorView;

/// Deferred mutation queued by the tree-view widgets so the immutable
/// borrow on `self.templates` ends before we call `get_mut` to apply it.
enum SceneAction {
    AddChild {
        parent: Option<NodeId>,
    },
    Delete {
        id: NodeId,
    },
    Reparent {
        id: NodeId,
        new_parent: Option<NodeId>,
        position: InsertPosition,
    },
}

/// Floor for any per-axis scale value passed to
/// [`Mat4::from_scale_rotation_translation`] — keeps the matrix
/// invertible so the decompose round-trip on the next frame can recover
/// the rotation the user is mid-editing.
const MIN_SCALE_AXIS: f32 = 1.0e-4;

impl LumberEditorView {
    /// Scene-tree section: node list + add / delete buttons. Renders
    /// nothing when no kind is selected or the kind has no template
    /// registered (e.g. its glb failed to load).
    pub(crate) fn scene_section(&mut self, current: Option<&KindId>, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        ui.heading("Scene");
        ui.separator();

        let Some(current_kind) = current.cloned() else {
            ui.label("(no kind selected)");
            return;
        };

        // Drop stale selection — happens when the user switched kinds (the
        // previous selection's NodeId doesn't exist in the new template)
        // or when a hot reload rebuilt the template and the id is gone.
        // Pruned on both the convenience mirror (`selected_node`) and the
        // widget's persistent state, so `egui_ltreeview` doesn't keep
        // pointing at a vanished id.
        if let Some(template) = self.templates.get(&current_kind) {
            if self
                .selected_node
                .is_some_and(|id| template.node(id).is_none())
            {
                self.selected_node = None;
            }
            let pruned: Vec<Option<NodeId>> = self
                .tree_view_state
                .selected()
                .iter()
                .copied()
                .filter(|sel| match sel {
                    None => true,
                    Some(id) => template.node(*id).is_some(),
                })
                .collect();
            if pruned.len() != self.tree_view_state.selected().len() {
                self.tree_view_state.set_selected(pruned);
            }
        }

        let mut actions: Vec<SceneAction> = Vec::new();

        // Tree view — immutable borrow on `self.templates` lives only inside
        // this block so the deferred mutations below can call `get_mut`.
        {
            let Some(template) = self.templates.get(&current_kind) else {
                ui.label("(no template registered)");
                return;
            };
            egui::ScrollArea::vertical()
                .max_height(220.0)
                .show(ui, |ui| {
                    let id = ui.make_persistent_id("lumber-editor-scene-tree");
                    let (_response, tree_actions) = TreeView::new(id)
                        .allow_drag_and_drop(true)
                        .allow_multi_selection(false)
                        .show_state(ui, &mut self.tree_view_state, |builder| {
                            // Wrapper "Template" root — gives drag-and-drop a
                            // drop target for "make this a top-level node",
                            // which the real template models with `parent: None`.
                            builder.node(NodeBuilder::dir(None).label("Template"));
                            for &root_slot in template.roots() {
                                build_subtree(builder, template, root_slot);
                            }
                            builder.close_dir();
                        });
                    for action in tree_actions {
                        if let Action::Move(dnd) = action
                            && let Some(scene_action) = translate_move(&dnd)
                        {
                            actions.push(scene_action);
                        }
                    }
                });
        }

        // Mirror the widget's selection into the convenience field the rest
        // of the editor reads (gizmo, inspector, "+ child" parent). First
        // selected `Some(id)` wins; `None` (the wrapper root) maps to no
        // selection.
        self.selected_node = self
            .tree_view_state
            .selected()
            .iter()
            .find_map(|s| s.as_ref().copied());

        ui.horizontal(|ui| {
            let add_label = if self.selected_node.is_some() {
                "+ child"
            } else {
                "+ root"
            };
            if ui.button(add_label).clicked() {
                actions.push(SceneAction::AddChild {
                    parent: self.selected_node,
                });
            }
            let delete_enabled = self.selected_node.is_some();
            let delete = ui.add_enabled(delete_enabled, egui::Button::new("- delete"));
            if delete.clicked()
                && let Some(id) = self.selected_node
            {
                actions.push(SceneAction::Delete { id });
            }
        });

        // Commit deferred mutations. Selection updates land here too so the
        // post-action selection (newly added node id, or cleared on delete)
        // wins over the in-frame click selection.
        let mut mutated = false;
        for action in actions {
            match action {
                SceneAction::AddChild { parent } => {
                    if let Some(template) = self.templates.get_mut(&current_kind) {
                        let new_id = template.next_free_node_id();
                        let name = format!("node_{}", new_id.0);
                        template.add_node(TemplateNode::empty(
                            new_id,
                            name,
                            parent,
                            Mat4::IDENTITY,
                        ));
                        self.selected_node = Some(new_id);
                        self.tree_view_state.set_selected(vec![Some(new_id)]);
                        if let Some(parent_id) = parent {
                            self.tree_view_state.expand_node(&Some(parent_id));
                        }
                        mutated = true;
                    }
                }
                SceneAction::Delete { id } => {
                    if let Some(template) = self.templates.get_mut(&current_kind) {
                        template.remove_node(id);
                        self.selected_node = None;
                        self.tree_view_state.set_selected(vec![]);
                        mutated = true;
                    }
                }
                SceneAction::Reparent {
                    id,
                    new_parent,
                    position,
                } => {
                    if let Some(template) = self.templates.get_mut(&current_kind)
                        && template.reparent_node(id, new_parent, position).is_ok()
                    {
                        mutated = true;
                    }
                }
            }
        }
        if mutated {
            self.dirty_kinds.insert(current_kind.clone());
        }

        // "Add mesh from glb" — VFS path text input + Add button. The
        // request lands as a Mesh node parented under the current
        // selection on the *next* frame's shadow_pass cascade 0, where
        // the renderer is available to build the MeshTemplate.
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("glb:");
            ui.add(
                egui::TextEdit::singleline(&mut self.glb_import_path)
                    .hint_text("e.g. lumber/logs.glb")
                    .desired_width(ui.available_width() - 60.0),
            );
        });
        let path_buffer = self.glb_import_path.trim().to_string();
        let path_valid = !path_buffer.is_empty() && VfsPath::new(&path_buffer).is_ok();
        if ui
            .add_enabled(path_valid, egui::Button::new("Add mesh from glb"))
            .clicked()
            && let Ok(path) = VfsPath::new(&path_buffer)
        {
            let parent = self.selected_node;
            if self.queue_glb_import(&current_kind, parent, path).is_some() {
                self.dirty_kinds.insert(current_kind.clone());
            }
            self.glb_import_path.clear();
        }

        // Graft from another template — deep-copy that kind's node tree
        // into this one with fresh NodeIds, parenting under the current
        // selection. Synchronous (no renderer needed). Picked from a
        // ComboBox listing every kind except the current one (self-graft
        // is a no-op).
        ui.add_space(6.0);
        // Snapshot the kinds list so the immutable borrow on self ends
        // before the show_ui closure runs (which captures the buffer it
        // writes into).
        let mut graft_source: Option<KindId> = None;
        let kinds: Vec<KindId> = self.kind_sources.keys().cloned().collect();
        egui::ComboBox::from_id_salt("graft-source")
            .selected_text("Graft from…")
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for kind in &kinds {
                    if *kind == current_kind {
                        continue;
                    }
                    if ui.selectable_label(false, kind.as_str()).clicked() {
                        graft_source = Some(kind.clone());
                    }
                }
            });
        if let Some(src) = graft_source
            && self.graft_from_template(&src, &current_kind) > 0
        {
            self.dirty_kinds.insert(current_kind.clone());
        }
    }

    /// Selected-node inspector: name field + TRS DragValues. Decomposes
    /// the current `local_transform` for display every frame; recomposes
    /// only on user edit, so untouched matrices don't suffer
    /// decompose→recompose drift.
    pub(crate) fn selected_node_section(&mut self, current: Option<&KindId>, ui: &mut egui::Ui) {
        let Some(current_kind) = current.cloned() else {
            return;
        };
        let Some(selected) = self.selected_node else {
            return;
        };
        let mut mutated = false;
        let Some(template) = self.templates.get_mut(&current_kind) else {
            return;
        };
        let Some(node) = template.node_mut(selected) else {
            return;
        };

        ui.heading("Selected node");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Id:");
            ui.monospace(format!("{}", node.id.0));
            ui.label("Kind:");
            ui.label(node_kind_label(&node.kind));
        });

        ui.horizontal(|ui| {
            ui.label("Name:");
            if ui.text_edit_singleline(&mut node.name).changed() {
                mutated = true;
            }
        });

        // Reset transform — recovery hatch for the degenerate-scale case
        // (the egui DragValue lets a user type 0 in any axis, which
        // collapses the matrix into a rank-deficient one that
        // to_scale_rotation_translation can't decompose back into sane
        // values). One click restores identity.
        ui.horizontal(|ui| {
            if ui.button("Reset transform").clicked() {
                node.local_transform = Mat4::IDENTITY;
                mutated = true;
            }
        });

        // Decompose for display. scale-rotation-translation; Euler XYZ
        // for the rotation surface so a user editing one axis at a time
        // gets predictable behaviour. Branch-flipping under decompose is
        // a known cost we accept until a 3D gizmo lands.
        let (scale, rotation, translation) = node.local_transform.to_scale_rotation_translation();
        let mut t = translation.to_array();
        let (rx, ry, rz) = rotation.to_euler(EulerRot::XYZ);
        let mut rot_deg = [rx * 180.0 / PI, ry * 180.0 / PI, rz * 180.0 / PI];
        let mut s = scale.to_array();

        let mut changed = false;

        ui.vertical(|ui| {
            ui.label("Translation");
            ui.columns(3, |ui| {
                for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
                    changed |= ui[i]
                        .add(
                            egui::DragValue::new(&mut t[i])
                                .speed(0.02)
                                .prefix(format!("{axis} ")),
                        )
                        .changed();
                }
            });
            ui.label("Rotation (deg)");
            ui.columns(3, |ui| {
                for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
                    changed |= ui[i]
                        .add(
                            egui::DragValue::new(&mut rot_deg[i])
                                .speed(1.0)
                                .prefix(format!("{axis} ")),
                        )
                        .changed();
                }
            });
            ui.label("Scale");
            ui.columns(3, |ui| {
                for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
                    changed |= ui[i]
                        .add(
                            egui::DragValue::new(&mut s[i])
                                .speed(0.02)
                                .prefix(format!("{axis} ")),
                        )
                        .changed();
                }
            });
        });

        if changed {
            let translation = Vec3::from(t);
            let rotation = Quat::from_euler(
                EulerRot::XYZ,
                rot_deg[0] * PI / 180.0,
                rot_deg[1] * PI / 180.0,
                rot_deg[2] * PI / 180.0,
            );
            // Clamp scale away from zero — Mat4::from_scale_rotation_translation
            // with a zero axis produces a rank-deficient matrix that
            // to_scale_rotation_translation can't unpack back into the
            // edited rotation, leaving the user stranded with no way to
            // recover via DragValue (the Reset button is the escape hatch
            // for already-broken matrices loaded from disk).
            let scale = Vec3::from(s).max(Vec3::splat(MIN_SCALE_AXIS));
            node.local_transform =
                Mat4::from_scale_rotation_translation(scale, rotation, translation);
            mutated = true;
        }
        if mutated {
            self.dirty_kinds.insert(current_kind);
        }
    }
}

/// Emit one [`egui_ltreeview`] node for the [`TemplateNode`] at `slot`
/// and recurse into its children. Every node is emitted as a `dir`
/// (not `leaf`) because any node in our model can host children
/// (Empty, Mesh, and Emitter alike) — making them all dirs keeps the
/// drop-into-anything affordance the user expects from the widget.
fn build_subtree<M, MK, E, S>(
    builder: &mut currawong::egui_ltreeview::TreeViewBuilder<'_, Option<NodeId>>,
    template: &RenderTemplate<M, MK, E, S>,
    slot: u32,
) {
    let node = &template.nodes()[slot as usize];
    let label = format!("{} {}", node_kind_icon(&node.kind), node.name);
    builder.node(NodeBuilder::dir(Some(node.id)).label(label));
    for &child in template.children(slot) {
        build_subtree(builder, template, child);
    }
    builder.close_dir();
}

/// Translate one [`egui_ltreeview`] drag-and-drop drop into a
/// [`SceneAction::Reparent`]. Returns `None` when the drop carries no
/// real-node source (only the wrapper template root) — that's a no-op,
/// not a tree edit.
///
/// The wrapper root key (`None`) maps to `parent = None` in the real
/// template; every other `Option<NodeId>` is `Some(real_id)`. Drag
/// sources are filtered down to the first real node since the widget
/// is configured for single-selection.
fn translate_move(
    dnd: &currawong::egui_ltreeview::DragAndDrop<Option<NodeId>>,
) -> Option<SceneAction> {
    let source_id = dnd.source.iter().find_map(|s| s.as_ref().copied())?;
    let new_parent = dnd.target;
    // Refuse to drop a node into itself — covered by reparent_node's
    // cycle check too, but cheap to catch early.
    if new_parent == Some(source_id) {
        return None;
    }
    let position = match dnd.position {
        DirPosition::First => InsertPosition::First,
        DirPosition::Last => InsertPosition::Last,
        DirPosition::Before(Some(sib)) => InsertPosition::Before(sib),
        DirPosition::After(Some(sib)) => InsertPosition::After(sib),
        // Sibling slot referenced the wrapper root — that can't happen in a
        // single-root tree; fall back to appending at the end.
        DirPosition::Before(None) | DirPosition::After(None) => InsertPosition::Last,
    };
    Some(SceneAction::Reparent {
        id: source_id,
        new_parent,
        position,
    })
}

fn node_kind_icon<M, MK, E, S>(kind: &NodeKind<M, MK, E, S>) -> &'static str {
    match kind {
        NodeKind::Empty => "○",
        NodeKind::Mesh(_) => "■",
        NodeKind::Emitter(_) => "✷",
    }
}

fn node_kind_label<M, MK, E, S>(kind: &NodeKind<M, MK, E, S>) -> &'static str {
    match kind {
        NodeKind::Empty => "Empty",
        NodeKind::Mesh(_) => "Mesh",
        NodeKind::Emitter(_) => "Emitter",
    }
}
