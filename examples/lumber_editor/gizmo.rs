//! Transform gizmo overlay — drags the selected node's local transform
//! via the [`transform-gizmo-egui`] crate.
//!
//! Edits the same `local_transform` the scene-panel DragValues edit, so
//! the existing dirty-tracking + Save machinery picks the changes up
//! without extra plumbing.
//!
//! ## Local↔world plumbing
//!
//! The gizmo lives in world space; the node we mutate stores a *local*
//! matrix relative to its parent (or the subject's
//! [`WorldTransform`](currawong::WorldTransform) at the root). On every
//! frame we compose:
//!
//! ```text
//!   node_world = object_world * parent₀.local * parent₁.local * ... * node.local
//!   parent_world = node_world / node.local
//! ```
//!
//! drive the gizmo with `node_world`, then bake the user-edited
//! `node_world'` back as `node.local = parent_world⁻¹ * node_world'`.
//! Same parent-world cancels out, so non-root children move identically
//! to roots from the user's perspective.
//!
//! ## Mode toggle
//!
//! The left panel exposes Translate / Rotate / Scale / All radio
//! buttons. "All" gates every sub-gizmo on the same screen — the
//! crate's default and easiest to land on a single click.

use currawong::data::KindId;
use currawong::egui;
use currawong::glam::{Mat4, Quat, Vec3};
use currawong::transform_gizmo_egui::GizmoInteraction;
use currawong::transform_gizmo_egui::config::GizmoModeKind;
use currawong::transform_gizmo_egui::math::Transform;
use currawong::transform_gizmo_egui::{
    EnumSet, Gizmo, GizmoConfig, GizmoMode, GizmoOrientation, mint,
};
use currawong::{NodeId, RenderTemplate};

use crate::LumberEditorView;
use crate::sim::Game;

/// Closed set of gizmo "modes" surfaced in the left-panel radio group.
/// One enum instead of a four-element `EnumSet<GizmoMode>` field so the
/// radio buttons map to a single egui selection without juggling sets.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum GizmoModeChoice {
    /// All three sub-gizmos visible — the crate's `GizmoMode::all()`.
    All,
    Translate,
    Rotate,
    Scale,
}

impl GizmoModeChoice {
    fn to_modes(self) -> EnumSet<GizmoMode> {
        match self {
            GizmoModeChoice::All => GizmoMode::all(),
            GizmoModeChoice::Translate => GizmoMode::all_from_kind(GizmoModeKind::Translate),
            GizmoModeChoice::Rotate => GizmoMode::all_from_kind(GizmoModeKind::Rotate),
            GizmoModeChoice::Scale => GizmoMode::all_from_kind(GizmoModeKind::Scale),
        }
    }
}

/// Per-view gizmo state. One [`Gizmo`] instance plus the current mode
/// selection — both live across frames so the gizmo can track its own
/// drag state internally between calls.
pub(crate) struct GizmoState {
    pub(crate) gizmo: Gizmo,
    pub(crate) mode: GizmoModeChoice,
    /// Hidden when the user wants the scene unobstructed. Independent of
    /// the existing `show_*` visibility checkboxes since the gizmo isn't
    /// a draw overlay in the same sense.
    pub(crate) enabled: bool,
}

impl GizmoState {
    pub(crate) fn new() -> Self {
        Self {
            gizmo: Gizmo::default(),
            mode: GizmoModeChoice::All,
            enabled: true,
        }
    }
}

impl LumberEditorView {
    /// Mode + enable radio group, intended to slot in below the existing
    /// scene-panel sections in the left panel.
    pub(crate) fn gizmo_section(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        ui.heading("Gizmo");
        ui.separator();
        ui.checkbox(&mut self.gizmo.enabled, "Show transform gizmo");
        ui.add_enabled_ui(self.gizmo.enabled, |ui| {
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.gizmo.mode, GizmoModeChoice::All, "All");
                ui.radio_value(&mut self.gizmo.mode, GizmoModeChoice::Translate, "T");
                ui.radio_value(&mut self.gizmo.mode, GizmoModeChoice::Rotate, "R");
                ui.radio_value(&mut self.gizmo.mode, GizmoModeChoice::Scale, "S");
            });
        });
    }

    /// Per-frame gizmo draw. Bypasses [`GizmoExt::interact`] so the
    /// gizmo doesn't register a click-and-drag widget with egui — that
    /// would cause egui to consume **every** pointer event including
    /// the RMB drag the [`OrbitRig`](currawong::OrbitRig) relies on for
    /// camera rotation.
    ///
    /// Instead we drive the lower-level
    /// [`Gizmo::update`] / [`Gizmo::draw`] / [`Gizmo::pick_preview`]
    /// directly: `pick_preview` tells us whether the cursor is over a
    /// handle, the pointer state comes from `egui_ctx.input(...)`, and
    /// we paint the resulting mesh into a `Foreground`-ordered layer
    /// clipped to the central area so we don't draw over the side
    /// panels.
    ///
    /// The configured viewport is the **full window**, matching the 3D
    /// scene's aspect — the camera's projection is built with
    /// `width / height`, so a smaller viewport here would squash the
    /// gizmo into an oval (the bug the first cut shipped with).
    pub(crate) fn draw_gizmo(&mut self, sim: &Game, egui_ctx: &egui::Context) {
        if !self.gizmo.enabled {
            return;
        }
        let Some(selected) = self.selected_node else {
            return;
        };
        let Some(current_kind) = sim
            .zones
            .get(sim.zone)
            .and_then(|z| z.components().get::<KindId>(sim.subject))
            .cloned()
        else {
            return;
        };

        // Subject world matrix — drives roots, and the leftmost factor in
        // any deeper chain. The editor's subject sits at origin with
        // default facing today, but compose it correctly anyway so the
        // gizmo doesn't lie if a future feature offsets the subject.
        let object_world = sim
            .zones
            .get(sim.zone)
            .and_then(|z| z.get(sim.subject))
            .map(|t| Mat4::from_rotation_translation(t.facing.to_quat(), t.position.to_vec3()))
            .unwrap_or(Mat4::IDENTITY);

        let template = match self.templates.get(&current_kind) {
            Some(t) => t,
            None => return,
        };
        let Some((parent_world, node_world)) = world_for_node(template, object_world, selected)
        else {
            return;
        };

        // Full window — matches the 3D scene's projection aspect. The
        // camera's `aspect` is set from the full surface size, so feeding
        // the gizmo any smaller rect would distort its NDC→screen mapping
        // (the bug that showed handles squashed into ovals).
        let viewport_rect = egui_ctx.viewport_rect();
        // Rect left over after the side panels reserved their space —
        // we clip the gizmo painter to it so we don't bleed onto the
        // panels.
        //
        // We deliberately don't mount a `CentralPanel` here: doing so
        // makes `Context::is_pointer_over_egui` true for the whole 3D
        // viewport, which in turn flips `wants_pointer_input` true and
        // causes the engine's `egui_consumed` gate to swallow every
        // `WindowEvent::MouseInput` — including the RMB press the
        // [`OrbitRig`](currawong::OrbitRig) needs to start a camera
        // rotation. `available_rect` reports the same rect without
        // registering anything with egui's interaction system.
        #[allow(deprecated)]
        let central_rect = egui_ctx.available_rect();
        let modes = self.gizmo.mode.to_modes();
        let view_matrix = mat4_to_row_matrix4(self.camera.view_matrix());
        let projection_matrix = mat4_to_row_matrix4(self.camera.projection_matrix());
        self.gizmo.gizmo.update_config(GizmoConfig {
            view_matrix,
            projection_matrix,
            viewport: viewport_rect,
            pixels_per_point: egui_ctx.pixels_per_point(),
            modes,
            orientation: GizmoOrientation::Local,
            ..Default::default()
        });

        let target = mat4_to_transform(node_world);

        let (cursor_pos, drag_started, dragging) = egui_ctx.input(|input| {
            let pos = input.pointer.hover_pos().unwrap_or_default();
            (
                (pos.x, pos.y),
                input.pointer.button_pressed(egui::PointerButton::Primary),
                input.pointer.button_down(egui::PointerButton::Primary),
            )
        });
        // Treat the cursor as "over" the gizmo only when it's actually
        // hovering a handle. Restrict to the central area so dragging the
        // side panels doesn't accidentally start a gizmo interaction.
        let cursor_in_central = central_rect.contains(egui::Pos2::new(cursor_pos.0, cursor_pos.1));
        let hovered = cursor_in_central && self.gizmo.gizmo.pick_preview(cursor_pos);

        let gizmo_result = self.gizmo.gizmo.update(
            GizmoInteraction {
                cursor_pos,
                hovered,
                drag_started,
                dragging,
            },
            std::slice::from_ref(&target),
        );

        // Paint into a foreground layer so we land on top of the
        // background-layer panel fills, but clip to the central area so
        // we don't bleed over the side panels.
        let draw_data = self.gizmo.gizmo.draw();
        let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("lumber-gizmo"));
        let painter = egui::Painter::new(egui_ctx.clone(), layer, central_rect);
        painter.add(egui::Mesh {
            indices: draw_data.indices,
            vertices: draw_data
                .vertices
                .into_iter()
                .zip(draw_data.colors)
                .map(|(pos, [r, g, b, a])| egui::epaint::Vertex {
                    pos: egui::Pos2::new(pos[0], pos[1]),
                    uv: egui::Pos2::ZERO,
                    color: egui::Rgba::from_rgba_premultiplied(r, g, b, a).into(),
                })
                .collect(),
            ..Default::default()
        });

        let Some((_result, transforms)) = gizmo_result else {
            return;
        };
        let Some(new_world) = transforms.into_iter().next().map(|t| transform_to_mat4(&t)) else {
            return;
        };
        // Bake back to local. Inverse is safe here — `parent_world`
        // composes only rotation + translation + (template-authored)
        // scale, and we clamp scale away from zero on the inspector
        // edit path; the gizmo itself can't author a zero scale unless
        // the user drags one to collapse.
        let new_local = parent_world.inverse() * new_world;
        if let Some(template) = self.templates.get_mut(&current_kind)
            && let Some(node) = template.node_mut(selected)
            && (node.local_transform - new_local)
                .to_cols_array()
                .iter()
                .any(|d| d.abs() > 1.0e-6)
        {
            node.local_transform = new_local;
            self.dirty_kinds.insert(current_kind);
        }
    }
}

/// Walks the parent chain root-to-leaf, composing `local_transform`s on
/// top of the subject's world matrix. Returns `(parent_world,
/// node_world)` so the caller can bake the gizmo result back into local
/// space without recomputing.
fn world_for_node<M, MK, E, S>(
    template: &RenderTemplate<M, MK, E, S>,
    object_world: Mat4,
    target: NodeId,
) -> Option<(Mat4, Mat4)> {
    // Collect leaf→root, then iterate in reverse for the matrix walk.
    let mut chain: Vec<&_> = Vec::new();
    let mut cursor = Some(target);
    while let Some(id) = cursor {
        let node = template.node(id)?;
        chain.push(node);
        cursor = node.parent;
    }
    let mut parent_world = object_world;
    let mut node_world = object_world;
    for node in chain.iter().rev() {
        parent_world = node_world;
        node_world *= node.local_transform;
    }
    Some((parent_world, node_world))
}

fn mat4_to_row_matrix4(m: Mat4) -> mint::RowMatrix4<f64> {
    // glam's `to_cols_array` is column-major; transposing first makes
    // the same `[f32; 16]` read as row-major, matching `RowMatrix4`.
    let c = m.transpose().to_cols_array();
    let row = |i: usize| mint::Vector4 {
        x: c[i] as f64,
        y: c[i + 1] as f64,
        z: c[i + 2] as f64,
        w: c[i + 3] as f64,
    };
    mint::RowMatrix4 {
        x: row(0),
        y: row(4),
        z: row(8),
        w: row(12),
    }
}

fn mat4_to_transform(m: Mat4) -> Transform {
    let (scale, rotation, translation) = m.to_scale_rotation_translation();
    Transform {
        scale: mint::Vector3 {
            x: scale.x as f64,
            y: scale.y as f64,
            z: scale.z as f64,
        },
        rotation: mint::Quaternion {
            v: mint::Vector3 {
                x: rotation.x as f64,
                y: rotation.y as f64,
                z: rotation.z as f64,
            },
            s: rotation.w as f64,
        },
        translation: mint::Vector3 {
            x: translation.x as f64,
            y: translation.y as f64,
            z: translation.z as f64,
        },
    }
}

fn transform_to_mat4(t: &Transform) -> Mat4 {
    Mat4::from_scale_rotation_translation(
        Vec3::new(t.scale.x as f32, t.scale.y as f32, t.scale.z as f32),
        Quat::from_xyzw(
            t.rotation.v.x as f32,
            t.rotation.v.y as f32,
            t.rotation.v.z as f32,
            t.rotation.s as f32,
        ),
        Vec3::new(
            t.translation.x as f32,
            t.translation.y as f32,
            t.translation.z as f32,
        ),
    )
}
