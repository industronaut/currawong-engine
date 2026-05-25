//! Hot-reload pipeline. Split into two halves because they want different
//! state:
//!
//! - [`LumberEditorView::maybe_hot_reload`] runs in `update`, where only
//!   the [`CommandQueue`] is available. It drains VFS-side change events
//!   from the [`AssetServer`], re-parses [`Definitions`] off the watcher
//!   tick, ships the typed digest to the sim via
//!   [`Command::ReloadDefinitions`], and parks the parsed registry in
//!   `self.pending_defs` for the next frame.
//! - [`LumberEditorView::maybe_rebuild_templates`] runs in `shadow_pass`
//!   (cascade 0), where the [`Renderer`] is available. It consumes
//!   `pending_defs` to rebuild the per-kind `mesh_templates` and
//!   `templates`.

use std::collections::HashMap;

use currawong::data::{Definitions, KindId, VfsPath};
use currawong::glam::Mat4;
use currawong::{
    CommandQueue, Footprint, Interaction, MeshTemplate, PbrMaterialInstance, RenderRegistry,
    RenderSpec, RenderTemplate, Renderer, pollster,
};

use crate::LumberEditorView;
use crate::sim::Command;
use crate::{MeshKey, Templates};

impl LumberEditorView {
    /// Drain VFS-side change events from the [`AssetServer`]; if anything
    /// changed, re-parse [`Definitions`], ship the typed digest to the sim
    /// via [`Command::ReloadDefinitions`], and park the parsed registry in
    /// [`Self::pending_defs`] for `shadow_pass` to consume.
    ///
    /// Catches three kinds of edit in one flow:
    /// - **Asset bytes** (glb / png) — `pump` evicts the cache entry; the
    ///   template rebuild in `shadow_pass` re-requests fresh handles.
    /// - **`.ron` def fields** (bounds, metallic, footprint…) — re-parsed
    ///   defs feed both the sim's typed caches and the next template
    ///   rebuild.
    /// - **`.ron` adds/removes** (new kind file, deleted kind file) — the
    ///   typed digest's `available` list changes; sim swaps it and resets
    ///   the subject if its KindId was deleted.
    ///
    /// A parse error during reload is logged and dropped — the editor
    /// keeps its previous defs rather than crashing on a half-saved file.
    pub(crate) fn maybe_hot_reload(&mut self, cmds: &mut CommandQueue<Command>) {
        let changed = self.asset_server.pump();
        if changed.is_empty() {
            return;
        }
        // A hot reload replaces sim.render_specs wholesale from disk, so
        // any in-memory bounds edit (this kind's or another's) is gone
        // regardless. Clear the dirty marker to match.
        self.pending_edit = None;
        let defs = match pollster::block_on(Definitions::load(
            self.asset_server.vfs(),
            &VfsPath::new("kinds").expect("valid VFS path"),
        )) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("lumber_editor: hot reload — definitions re-parse failed: {e}");
                return;
            }
        };
        let mut available: Vec<KindId> = Vec::new();
        let mut render_specs: HashMap<KindId, RenderSpec> = HashMap::new();
        let mut interactions: HashMap<KindId, Interaction> = HashMap::new();
        let mut footprints: HashMap<KindId, Footprint> = HashMap::new();
        for (kind_id, def) in defs.iter() {
            let Ok(spec) = RenderSpec::from_def(def) else {
                continue;
            };
            available.push(kind_id.clone());
            render_specs.insert(kind_id.clone(), spec);
            interactions.insert(
                kind_id.clone(),
                Interaction::from_def(def).unwrap_or(Interaction::None),
            );
            footprints.insert(
                kind_id.clone(),
                Footprint::from_def(def).unwrap_or_default(),
            );
        }
        available.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        eprintln!(
            "lumber_editor: hot reload — {} change(s), {} kind(s) after re-parse",
            changed.len(),
            available.len()
        );
        cmds.push_now(Command::ReloadDefinitions {
            available,
            render_specs,
            interactions,
            footprints,
        });
        self.pending_defs = Some(defs);
    }

    /// Rebuild `mesh_templates` + `templates` (and register any new bucket
    /// keys) from `self.pending_defs`. No-op when no reload is pending.
    /// Runs once per reload, at the top of `shadow_pass` cascade 0 where
    /// the `Renderer` is available.
    ///
    /// Templates for kinds removed in the new defs aren't explicitly
    /// dropped from `buckets` (no API for that today) — they simply stop
    /// getting `push`ed because nothing in the cull walk references them.
    /// The bucket buffer lingers as a small leak; fine for editor scope.
    pub(crate) fn maybe_rebuild_templates(&mut self, renderer: &Renderer) {
        let Some(defs) = self.pending_defs.take() else {
            return;
        };
        let mut mesh_templates: HashMap<MeshKey, MeshTemplate<PbrMaterialInstance>> =
            HashMap::new();
        let mut templates: Templates = RenderRegistry::new();
        let mut kind_sources: HashMap<KindId, VfsPath> = HashMap::new();
        for (kind_id, def) in defs.iter() {
            kind_sources.insert(kind_id.clone(), def.source.clone());
        }
        for (kind_id, _spec, body) in self.material.streamed_kind_body_templates(
            renderer,
            &self.samplers,
            &self.asset_server,
            &defs,
            |_, _| {},
        ) {
            let bounds = body.visual_bounds;
            let body_key = MeshKey::KindBody(kind_id.clone());
            mesh_templates.insert(body_key.clone(), body);
            let template = RenderTemplate::new(kind_id.as_str())
                .with_mesh_part(body_key.clone(), body_key, Mat4::IDENTITY)
                .with_visual_bounds(bounds);
            templates.register(kind_id, template);
        }
        for key in mesh_templates.keys().cloned().collect::<Vec<_>>() {
            self.buckets.register(&renderer.device, key);
        }
        self.mesh_templates = mesh_templates;
        self.templates = templates;
        self.kind_sources = kind_sources;
        // Invalidate the auto-frame cache so a kind whose visual bounds
        // changed reframes the camera on the next `update` tick.
        self.last_selected = None;
    }
}
