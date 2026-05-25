//! Left-side egui panel: kind dropdown, visibility checkboxes, and the
//! recalc/Save buttons that drive the in-memory bounds edit flow. The
//! `View::ui` trait method in [`crate::main`] delegates to
//! [`LumberEditorView::kind_panel`] here.

use currawong::data::KindId;
use currawong::{CommandQueue, MeshBacking, egui};

use crate::LumberEditorView;
use crate::sim::{Command, Game};

impl LumberEditorView {
    pub(crate) fn kind_panel(
        &mut self,
        sim: &Game,
        cmds: &mut CommandQueue<Command>,
        egui_ctx: &egui::Context,
    ) {
        let current = sim
            .zones
            .get(sim.zone)
            .and_then(|z| z.components().get::<KindId>(sim.subject))
            .cloned();

        // egui 0.34 deprecated top-level `Panel::show` in favour of
        // `show_inside`, which needs an outer `Ui`. View callbacks receive
        // only `&Context`, and the upstream migration story for top-level
        // panel hosting in pure `&Context` callers isn't settled — the
        // deprecated path still works, so we use it.
        #[allow(deprecated)]
        egui::Panel::left("kinds")
            .resizable(false)
            .default_size(260.0)
            .show(egui_ctx, |ui| {
                ui.heading("Kind");
                ui.separator();
                let selected_text = current
                    .as_ref()
                    .map(|k| k.as_str().to_string())
                    .unwrap_or_else(|| "Select…".to_string());
                egui::ComboBox::from_id_salt("kind-select")
                    .selected_text(selected_text)
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        for kind in &sim.available {
                            let selected = current.as_ref() == Some(kind);
                            if ui.selectable_label(selected, kind.as_str()).clicked() && !selected {
                                // Switching away from a kind with an
                                // unsaved bounds edit reverts that edit —
                                // per the editor's "lost if another file
                                // is loaded" model. Done by pushing an
                                // UpdateBounds with the pre-edit pristine
                                // values; the sim applies it on the next
                                // tick, just before SelectKind.
                                if let Some((dirty_kind, pristine)) = self.pending_edit.take() {
                                    cmds.push_now(Command::UpdateBounds {
                                        kind: dirty_kind,
                                        min: pristine.bounds_min,
                                        max: pristine.bounds_max,
                                    });
                                }
                                cmds.push_now(Command::SelectKind(kind.clone()));
                            }
                        }
                    });

                ui.add_space(8.0);
                ui.heading("Visibility");
                ui.separator();
                ui.checkbox(&mut self.show_main_mesh, "Main item mesh");
                ui.checkbox(&mut self.show_bounding_box, "Bounding box");
                ui.checkbox(&mut self.show_interaction_tiles, "Interaction tiles");
                ui.checkbox(&mut self.show_footprint_tiles, "Footprint tiles");
                ui.checkbox(&mut self.show_facing_arrow, "Facing arrow");

                ui.add_space(8.0);
                ui.heading("Bounding box");
                ui.separator();
                // Only enable the button when the mesh is loaded — recalc
                // off the magenta fallback would push its `[-0.5, 0.5]^3`
                // cube into the in-memory spec, which is the wrong
                // direction.
                let mesh_ready = current
                    .as_ref()
                    .and_then(|kind| {
                        self.mesh_templates
                            .get(&crate::MeshKey::KindBody(kind.clone()))
                    })
                    .is_some_and(|tpl| match &tpl.mesh {
                        MeshBacking::Streamed { handle } => handle.is_ready(),
                        MeshBacking::Inline { .. } => true,
                    });
                let recalc_button = egui::Button::new("Recalc bounding box from mesh");
                if ui.add_enabled(mesh_ready, recalc_button).clicked()
                    && let Some(kind) = current.as_ref()
                {
                    self.recalc_bounds_for(kind, sim, cmds);
                }

                // Scene-tree section (Phase 4 — scene_panel.rs). The
                // selected-node inspector lives in its own right-side
                // panel below so the left panel doesn't change height
                // when the user clicks into a node.
                self.scene_section(current.as_ref(), ui);

                // Gizmo enable + mode toggle. Lives in the left panel
                // alongside the visibility toggles so all view-side
                // overlays cluster in one place.
                self.gizmo_section(ui);

                // Save button anchored to the bottom of the panel. Bottom-up
                // layout reverses the natural top-down flow, so the button
                // sits flush against the panel's lower edge regardless of
                // how much vertical space the sections above consumed.
                // Enabled when the selected kind has any unsaved edit —
                // bounds recalc (legacy `pending_edit`) or scene-tree
                // mutation tracked in `dirty_kinds`.
                let dirty = current.as_ref().is_some_and(|k| self.is_kind_dirty(k));
                ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                    let save_button = egui::Button::new("Save");
                    if ui
                        .add_enabled(dirty, save_button)
                        .on_hover_text(
                            "Rewrite the selected kind's `render` block on \
                             disk with the current bounds + scene tree.",
                        )
                        .clicked()
                        && let Some(kind) = current.as_ref()
                    {
                        self.save_template_for(kind, sim);
                    }
                    ui.separator();
                });
            });

        // Right-side panel — only shown when a node is selected. Keeps
        // the selected-node inspector visually separated from the
        // scene-tree controls and avoids the left panel's height
        // jumping every time the user clicks into a node.
        if self.selected_node.is_some() {
            #[allow(deprecated)]
            egui::Panel::right("selected-node")
                .resizable(false)
                .default_size(280.0)
                .show(egui_ctx, |ui| {
                    self.selected_node_section(current.as_ref(), ui);
                });
        }
    }
}
