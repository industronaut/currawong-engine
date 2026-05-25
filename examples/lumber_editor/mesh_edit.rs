//! Bounding-box recalc + camera auto-frame.
//!
//! - [`LumberEditorView::recalc_bounds_for`] reads the kind's loaded mesh,
//!   snapshots the pre-edit `RenderSpec` into `pending_edit`, and pushes a
//!   [`Command::UpdateBounds`] so the sim's `render_specs[kind]` reflects
//!   the new value on the next tick. Save lives in [`crate::save`] now —
//!   the unified rewriter handles both bounds edits and scene-tree
//!   mutations through the same render-block-replacement path.
//! - [`LumberEditorView::maybe_auto_frame`] snaps the orbit rig to fit the
//!   newly-selected (or freshly-edited) kind's AABB.

use currawong::data::KindId;
use currawong::{CommandQueue, HandleState, MeshBacking};

use crate::LumberEditorView;
use crate::sim::{Command, Game};

impl LumberEditorView {
    /// Recalculate the visual AABB for `kind` from its loaded mesh and
    /// push an [`Command::UpdateBounds`] so the sim's `render_specs[kind]`
    /// reflects the new value on the next tick. The edit stays in memory
    /// — the Save button is what mirrors it out to disk.
    ///
    /// Snapshots the pre-edit [`RenderSpec`](currawong::RenderSpec) into
    /// `pending_edit` on the first recalc for a given kind so the edit
    /// can be reverted if the user switches kinds without saving.
    /// Subsequent recalcs on the same kind reuse the same pristine
    /// snapshot — clicking recalc N times in a row is still "one edit"
    /// from the dirty-tracking perspective.
    ///
    /// Quietly no-ops if the mesh isn't `Ready` (button is gated in `ui`
    /// too, but state can change between the check and the click).
    pub(crate) fn recalc_bounds_for(
        &mut self,
        kind: &KindId,
        sim: &Game,
        cmds: &mut CommandQueue<Command>,
    ) {
        let Some(template) = self
            .mesh_templates
            .get(&crate::MeshKey::KindBody(kind.clone()))
        else {
            eprintln!("lumber_editor: recalc — no mesh template for {kind}");
            return;
        };
        let bounds = match &template.mesh {
            MeshBacking::Streamed { handle } => match handle.peek() {
                HandleState::Ready(mesh) => mesh.bounds,
                HandleState::Loading => {
                    eprintln!("lumber_editor: recalc — mesh for {kind} is still loading");
                    return;
                }
                HandleState::Failed(err) => {
                    eprintln!("lumber_editor: recalc — mesh for {kind} failed: {err}");
                    return;
                }
            },
            // Inline templates don't carry a runtime `Mesh.bounds`; not
            // produced by the editor today, but skip cleanly if one ever
            // appears rather than silently writing a degenerate AABB.
            MeshBacking::Inline { .. } => {
                eprintln!("lumber_editor: recalc — {kind} uses an inline mesh, no recalc");
                return;
            }
        };
        if self.pending_edit.as_ref().is_none_or(|(k, _)| k != kind)
            && let Some(spec) = sim.render_specs.get(kind)
        {
            self.pending_edit = Some((kind.clone(), spec.clone()));
        }
        cmds.push_now(Command::UpdateBounds {
            kind: kind.clone(),
            min: (bounds.min.x, bounds.min.y, bounds.min.z),
            max: (bounds.max.x, bounds.max.y, bounds.max.z),
        });
        // The new bounds may differ enough from the disk-loaded ones that
        // the camera should refit. Invalidate the auto-frame cache so
        // `maybe_auto_frame` runs again on the next `update`.
        self.last_selected = None;
    }

    /// Snap the orbit rig to fit the newly-selected kind's bounds. No-op
    /// when the selection hasn't changed since the previous frame.
    /// (Bounds overlay re-upload lives in `render` where the queue is
    /// available — `EngineCtx` deliberately doesn't expose `Renderer`.)
    pub(crate) fn maybe_auto_frame(&mut self, sim: &Game) {
        let current = sim
            .zones
            .get(sim.zone)
            .and_then(|z| z.components().get::<KindId>(sim.subject))
            .cloned();
        if current == self.last_selected {
            return;
        }
        if let Some(kind) = &current
            && let Some(spec) = sim.render_specs.get(kind)
        {
            let aabb = spec.visual_bounds();
            let centre = (aabb.min + aabb.max) * 0.5;
            let extent = (aabb.max - aabb.min).max_element();
            // `extent * 2` keeps the AABB comfortably inside the 45°-ish
            // FOV with margin for surrounding emitter reach; floor at 1 m
            // so a tiny mesh doesn't degenerate to the rig's
            // distance_min clamp.
            self.rig.focus = centre;
            self.rig.distance = (extent * 2.0).max(1.0);
        }
        self.last_selected = current;
    }
}
