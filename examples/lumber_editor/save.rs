//! Template save — write the in-memory [`RenderTemplate`] for one kind
//! back into its source `.ron` file, replacing the entire `render: (...)`
//! block with a freshly-serialised version containing the current
//! hierarchical `nodes:` array plus the flat fields preserved from
//! [`sim.render_specs`](crate::sim::Game::render_specs).
//!
//! Single source of truth on save:
//! [`Game::render_specs`](crate::sim::Game::render_specs) holds the
//! current flat fields (refreshed by `Command::UpdateBounds` after a
//! recalc); the editor's
//! [`templates`](crate::LumberEditorView::templates) holds the
//! hierarchical tree as currently authored. The save merges them — flat
//! fields from the sim, `nodes` from the template.
//!
//! Caveats for the initial implementation:
//! - Comments inside the `render: (...)` block are not preserved across
//!   save — the block is rebuilt from scratch. Comments outside the
//!   block (other top-level fields, file header) are kept verbatim.
//! - Per-node material parameters (metallic / roughness / albedo) for
//!   editor-added Mesh nodes are written with PBR defaults. Hand-authored
//!   non-default values for a node load fine but round-trip back to
//!   defaults after the next save. Persisting them needs the editor to
//!   either expose per-node material editing or read the original spec
//!   on save — neither is in scope for Phase 6.
//! - The render-block finder is a paren-counter with string-literal
//!   awareness; it doesn't track comments. A `// render:` inside a comment
//!   above the real `render:` block would still find the real one because
//!   the matcher prefers the first whole-word `render:` hit, but
//!   pathological cases are possible.

use std::path::Path;

use currawong::data::KindId;
use currawong::{MeshNodeSpec, NodeKind, NodeSpec, RenderTemplate, TransformSpec, node_kind};

use crate::sim::Game;
use crate::{LumberEditorView, MeshKey};

impl LumberEditorView {
    /// Mirror the current in-memory template + flat-spec fields for
    /// `kind` to its source `.ron` file. Clears the dirty marker and
    /// the legacy bounds-recalc `pending_edit` on success — both kinds
    /// of edit are now persisted.
    ///
    /// Logs and bails on any I/O or serialisation error rather than
    /// panicking; the editor stays usable so the user can retry.
    pub(crate) fn save_template_for(&mut self, kind: &KindId, sim: &Game) {
        let Some(source) = self.kind_sources.get(kind).cloned() else {
            eprintln!("lumber_editor: save — no source path known for {kind}");
            return;
        };
        let Some(spec) = sim.render_specs.get(kind) else {
            eprintln!("lumber_editor: save — no render_spec for {kind}");
            return;
        };
        let Some(template) = self.templates.get(kind) else {
            eprintln!("lumber_editor: save — no template for {kind}");
            return;
        };

        // Build the spec we'll write to disk: clone the current flat
        // fields from the sim, replace the `nodes` array with the
        // current template's serialised form.
        let mut to_write = spec.clone();
        to_write.nodes = node_specs_from_template(template, &spec.mesh);

        let serialised = match ron::ser::to_string_pretty(
            &to_write,
            ron::ser::PrettyConfig::new()
                .indentor("    ".to_string())
                .depth_limit(8),
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("lumber_editor: save — RON serialise failed for {kind}: {e}");
                return;
            }
        };
        // ron::ser emits the struct as `( shape: ..., ... )`. The render
        // block wants `render: ( ... )` with leading whitespace + trailing
        // comma to match RON style.
        let new_block = format!("render: {serialised}");

        let on_disk = self.assets_root.join(source.as_str());
        if let Err(e) = replace_render_block(&on_disk, &new_block) {
            eprintln!(
                "lumber_editor: save — failed to rewrite {}: {e}",
                on_disk.display()
            );
            return;
        }
        eprintln!(
            "lumber_editor: save — wrote {} nodes to {}",
            to_write.nodes.len(),
            on_disk.display()
        );
        self.dirty_kinds.remove(kind);
        // Bounds recalc edits subsume into the rewrite; clear the legacy
        // pending state so the Save-button gating reflects "nothing
        // pending."
        if self.pending_edit.as_ref().is_some_and(|(k, _)| k == kind) {
            self.pending_edit = None;
        }
        let _ = source; // moved into clone above, silence unused for stable
    }

    /// True if `kind` has any unsaved edits — either a scene-tree
    /// mutation tracked in [`Self::dirty_kinds`], or a bounds-recalc
    /// edit tracked in [`Self::pending_edit`].
    pub(crate) fn is_kind_dirty(&self, kind: &KindId) -> bool {
        self.dirty_kinds.contains(kind)
            || self.pending_edit.as_ref().is_some_and(|(k, _)| k == kind)
    }
}

/// Walk the template's nodes and project each one back into a
/// [`NodeSpec`]. Mesh nodes resolve their VFS path from the [`MeshKey`]
/// — `KindBody` falls back to `body_mesh_path` (the kind's flat
/// `render.mesh`), `Glb` uses the path baked into the key. Emitter
/// nodes are skipped — the editor doesn't author them today and the
/// schema doesn't have a serialisation form yet.
fn node_specs_from_template(
    template: &RenderTemplate<MeshKey, MeshKey>,
    body_mesh_path: &str,
) -> Vec<NodeSpec> {
    template
        .nodes()
        .iter()
        .filter_map(|node| {
            let (kind_tag, mesh) = match &node.kind {
                NodeKind::Empty => (node_kind::EMPTY.to_string(), None),
                NodeKind::Mesh(part) => {
                    let mesh_path = match &part.mesh {
                        MeshKey::KindBody(_) => body_mesh_path.to_string(),
                        MeshKey::Glb(p) => p.as_ref().to_string(),
                    };
                    (
                        node_kind::MESH.to_string(),
                        Some(MeshNodeSpec {
                            mesh: mesh_path,
                            albedo: None,
                            metallic: 0.0,
                            roughness: 0.85,
                        }),
                    )
                }
                NodeKind::Emitter(_) => return None,
            };
            Some(NodeSpec {
                id: node.id.0,
                name: node.name.clone(),
                parent: node.parent.map(|p| p.0),
                transform: TransformSpec::from_mat4(node.local_transform),
                kind: kind_tag,
                mesh,
            })
        })
        .collect()
}

/// Find the byte span of the top-level `render: (...)` block in `src`
/// using paren-matching. Returns `None` if no well-formed block is
/// found.
///
/// String-literal aware (a `(` or `)` inside a `"..."` doesn't count
/// toward the depth). Not comment-aware — a `// render:` line above the
/// real block could shadow the search in pathological files.
fn find_render_block_span(src: &str) -> Option<(usize, usize)> {
    let needle = "render:";
    let bytes = src.as_bytes();
    let mut from = 0usize;
    let block_start;
    loop {
        let pos = src[from..].find(needle)? + from;
        // Word-boundary check on the byte before: must not be alphanumeric
        // or underscore, otherwise we'd match something like `prerender:`.
        let prev_ok = pos == 0 || {
            let p = bytes[pos - 1];
            !(p.is_ascii_alphanumeric() || p == b'_')
        };
        if prev_ok {
            block_start = pos;
            break;
        }
        from = pos + needle.len();
    }

    // Skip the `:` and any whitespace, then expect a `(`.
    let mut i = block_start + needle.len();
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'(' {
        return None;
    }

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    let mut j = i;
    while j < bytes.len() {
        let c = bytes[j];
        if in_string {
            if escape {
                escape = false;
            } else if c == b'\\' {
                escape = true;
            } else if c == b'"' {
                in_string = false;
            }
        } else {
            match c {
                b'"' => in_string = true,
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((block_start, j + 1));
                    }
                }
                _ => {}
            }
        }
        j += 1;
    }
    None
}

/// Read the file, splice `new_block` into the existing
/// `render: (...)` span, write it back.
fn replace_render_block(path: &Path, new_block: &str) -> std::io::Result<()> {
    let original = std::fs::read_to_string(path)?;
    let Some((start, end)) = find_render_block_span(&original) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "no top-level `render:` block found",
        ));
    };
    let mut out = String::with_capacity(original.len() + new_block.len());
    out.push_str(&original[..start]);
    out.push_str(new_block);
    out.push_str(&original[end..]);
    std::fs::write(path, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_render_block_at_top_level() {
        let src = "(name: \"x\", render: (a: 1, b: (2, 3)), more: 42)";
        let (start, end) = find_render_block_span(src).expect("found");
        assert_eq!(&src[start..end], "render: (a: 1, b: (2, 3))");
    }

    #[test]
    fn paren_in_string_does_not_break_matching() {
        let src = "(render: (path: \"a) b (c\", n: 1))";
        let (start, end) = find_render_block_span(src).expect("found");
        // Outer paren close at the very end of the test string.
        assert_eq!(&src[start..end], "render: (path: \"a) b (c\", n: 1)");
    }

    #[test]
    fn word_boundary_skips_substring_match() {
        // "prerender:" must not match.
        let src = "(prerender: 1, render: (n: 2))";
        let (start, end) = find_render_block_span(src).expect("found");
        assert_eq!(&src[start..end], "render: (n: 2)");
    }

    #[test]
    fn replace_block_round_trip() {
        let tmp = std::env::temp_dir().join("lumber_editor_save_test.ron");
        let original =
            "(name: \"x\",\n    render: (\n        shape: \"tree\",\n    ),\n    other: 1)\n";
        std::fs::write(&tmp, original).unwrap();
        replace_render_block(&tmp, "render: (shape: \"replaced\")").unwrap();
        let after = std::fs::read_to_string(&tmp).unwrap();
        assert!(after.contains("render: (shape: \"replaced\")"));
        assert!(after.contains("name: \"x\""));
        assert!(after.contains("other: 1"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn missing_render_block_is_an_error() {
        let tmp = std::env::temp_dir().join("lumber_editor_save_test_norender.ron");
        std::fs::write(&tmp, "(no_render: true)\n").unwrap();
        assert!(replace_render_block(&tmp, "render: ()").is_err());
        let _ = std::fs::remove_file(&tmp);
    }
}
