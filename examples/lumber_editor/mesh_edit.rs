//! In-memory bounding-box edits + Save flow + camera auto-frame. All three
//! share the same lifecycle:
//!
//! - [`LumberEditorView::recalc_bounds_for`] reads the kind's loaded mesh,
//!   snapshots the pre-edit `RenderSpec` into `pending_edit`, and pushes a
//!   [`Command::UpdateBounds`] so the sim's `render_specs[kind]` reflects
//!   the new value on the next tick.
//! - [`LumberEditorView::save_bounds_for`] mirrors the sim's current
//!   in-memory bounds out to the kind's `.ron` file, clearing
//!   `pending_edit`. The file watcher then triggers a hot reload that
//!   no-ops on bounds (disk now matches in-memory).
//! - [`LumberEditorView::maybe_auto_frame`] snaps the orbit rig to fit the
//!   newly-selected (or freshly-edited) kind's AABB.
//!
//! [`rewrite_bounds_in_ron`] is the line-level rewriter the Save button
//! depends on — preserves comments, indentation, and unrelated fields.

use std::path::Path;

use currawong::data::KindId;
use currawong::{Aabb, CommandQueue, HandleState, MeshBacking};

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

    /// Mirror the sim's current in-memory bounds for `kind` out to its
    /// source `.ron` file and clear `pending_edit`. The file watcher
    /// picks up the write and triggers a hot reload through
    /// [`Self::maybe_hot_reload`]; since disk now matches in-memory the
    /// resulting `ReloadDefinitions` is a no-op for the bounds.
    pub(crate) fn save_bounds_for(&mut self, kind: &KindId, bounds: Aabb) {
        let Some(source) = self.kind_sources.get(kind) else {
            eprintln!("lumber_editor: save — no source path known for {kind}");
            return;
        };
        let on_disk = self.assets_root.join(source.as_str());
        if let Err(e) = rewrite_bounds_in_ron(&on_disk, bounds) {
            eprintln!(
                "lumber_editor: save — failed to rewrite {}: {e}",
                on_disk.display()
            );
            return;
        }
        eprintln!(
            "lumber_editor: save — wrote bounds_min={:?}, bounds_max={:?} to {}",
            bounds.min,
            bounds.max,
            on_disk.display()
        );
        self.pending_edit = None;
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

// --- Bounds rewrite ----------------------------------------------------

/// Rewrite the `bounds_min:` and `bounds_max:` lines of a kind `.ron` file
/// in place from `bounds`. Walks the file line by line so comments,
/// formatting, and unrelated fields are preserved — the only edits are
/// the two tuple values inside the `render:` block. Whitespace at the
/// start of each line is kept so the indentation matches the source.
///
/// A degenerate match (a kind file with two `bounds_min:` lines, or one
/// inside a string literal) would rewrite both. Acceptable for the
/// editor's authored content; not robust against adversarial RON.
fn rewrite_bounds_in_ron(path: &Path, bounds: Aabb) -> std::io::Result<()> {
    let original = std::fs::read_to_string(path)?;
    let mut out = String::with_capacity(original.len());
    for line in original.split_inclusive('\n') {
        if let Some(rewritten) = rewrite_bounds_line(line, bounds) {
            out.push_str(&rewritten);
        } else {
            out.push_str(line);
        }
    }
    std::fs::write(path, out)
}

/// Return a replacement for `line` if it's a `bounds_min:` / `bounds_max:`
/// line; otherwise `None`. Preserves leading indentation and the original
/// line ending. `Debug` formatting on the f32 components matches the
/// existing RON style (`0.0` not `0`, `-0.5` not `-0.50000`).
fn rewrite_bounds_line(line: &str, bounds: Aabb) -> Option<String> {
    let stripped = line.strip_suffix("\r\n").or(line.strip_suffix('\n'));
    let body = stripped.unwrap_or(line);
    let line_end = &line[body.len()..];
    let trimmed = body.trim_start();
    let indent = &body[..body.len() - trimmed.len()];
    let (field, v) = if trimmed.starts_with("bounds_min:") {
        ("bounds_min", bounds.min)
    } else if trimmed.starts_with("bounds_max:") {
        ("bounds_max", bounds.max)
    } else {
        return None;
    };
    Some(format!(
        "{indent}{field}: ({:?}, {:?}, {:?}),{line_end}",
        v.x, v.y, v.z,
    ))
}
