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

use std::f32::consts::PI;

use currawong::data::{KindId, VfsPath};
use currawong::glam::{EulerRot, Mat4, Quat, Vec3};
use currawong::{NodeId, NodeKind, RenderTemplate, TemplateNode, egui};

use crate::LumberEditorView;

/// Deferred mutation queued by the tree-view widgets so the immutable
/// borrow on `self.templates` ends before we call `get_mut` to apply it.
enum SceneAction {
    AddChild { parent: Option<NodeId> },
    Delete { id: NodeId },
}

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
        if let Some(id) = self.selected_node {
            let exists = self
                .templates
                .get(&current_kind)
                .map(|t| t.node(id).is_some())
                .unwrap_or(false);
            if !exists {
                self.selected_node = None;
            }
        }

        let mut new_selection = self.selected_node;
        let mut action: Option<SceneAction> = None;

        // Tree view — immutable borrow on self.templates, walked in a
        // scope so the borrow ends before the action-buttons block below.
        {
            let Some(template) = self.templates.get(&current_kind) else {
                ui.label("(no template registered)");
                return;
            };
            egui::ScrollArea::vertical()
                .max_height(180.0)
                .show(ui, |ui| {
                    for &root in template.roots() {
                        render_node(ui, template, root, 0, &mut new_selection);
                    }
                    if template.node_count() == 0 {
                        ui.label("(empty)");
                    }
                });
        }

        ui.horizontal(|ui| {
            let add_label = if new_selection.is_some() {
                "+ child"
            } else {
                "+ root"
            };
            if ui.button(add_label).clicked() {
                action = Some(SceneAction::AddChild {
                    parent: new_selection,
                });
            }
            let delete_enabled = new_selection.is_some();
            let delete = ui.add_enabled(delete_enabled, egui::Button::new("- delete"));
            if delete.clicked()
                && let Some(id) = new_selection
            {
                action = Some(SceneAction::Delete { id });
            }
        });

        // Commit the deferred mutation. Selection updates land here too
        // so the post-action selection (newly added node id, or cleared on
        // delete) wins over the in-frame click selection.
        self.selected_node = new_selection;
        match action {
            Some(SceneAction::AddChild { parent }) => {
                if let Some(template) = self.templates.get_mut(&current_kind) {
                    let new_id = template.next_free_node_id();
                    let name = format!("node_{}", new_id.0);
                    template.add_node(TemplateNode::empty(new_id, name, parent, Mat4::IDENTITY));
                    self.selected_node = Some(new_id);
                }
            }
            Some(SceneAction::Delete { id }) => {
                if let Some(template) = self.templates.get_mut(&current_kind) {
                    template.remove_node(id);
                }
                self.selected_node = None;
            }
            None => {}
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
            self.queue_glb_import(&current_kind, parent, path);
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
        if let Some(src) = graft_source {
            self.graft_from_template(&src, &current_kind);
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
        let Some(template) = self.templates.get_mut(&current_kind) else {
            return;
        };
        let Some(node) = template.node_mut(selected) else {
            return;
        };

        ui.add_space(8.0);
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
            ui.text_edit_singleline(&mut node.name);
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
        ui.label("Translation");
        ui.horizontal(|ui| {
            for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut t[i])
                            .speed(0.02)
                            .prefix(format!("{axis} ")),
                    )
                    .changed();
            }
        });
        ui.label("Rotation (deg)");
        ui.horizontal(|ui| {
            for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut rot_deg[i])
                            .speed(1.0)
                            .prefix(format!("{axis} ")),
                    )
                    .changed();
            }
        });
        ui.label("Scale");
        ui.horizontal(|ui| {
            for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut s[i])
                            .speed(0.02)
                            .prefix(format!("{axis} ")),
                    )
                    .changed();
            }
        });

        if changed {
            let translation = Vec3::from(t);
            let rotation = Quat::from_euler(
                EulerRot::XYZ,
                rot_deg[0] * PI / 180.0,
                rot_deg[1] * PI / 180.0,
                rot_deg[2] * PI / 180.0,
            );
            let scale = Vec3::from(s);
            node.local_transform =
                Mat4::from_scale_rotation_translation(scale, rotation, translation);
        }
    }
}

/// Recursive tree row. Indents by depth and renders each child after the
/// parent's own row, in declaration order. Empty children list is fine —
/// produces just the parent's row.
fn render_node<M, MK, E, S>(
    ui: &mut egui::Ui,
    template: &RenderTemplate<M, MK, E, S>,
    slot: u32,
    depth: u8,
    selected: &mut Option<NodeId>,
) {
    let node = &template.nodes()[slot as usize];
    let id = node.id;
    ui.horizontal(|ui| {
        ui.add_space(depth as f32 * 14.0);
        let label = format!("{} {}", node_kind_icon(&node.kind), node.name);
        if ui.selectable_label(*selected == Some(id), label).clicked() {
            *selected = Some(id);
        }
    });
    for &child in template.children(slot) {
        render_node(ui, template, child, depth + 1, selected);
    }
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
