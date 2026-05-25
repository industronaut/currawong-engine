//! Build the runtime [`RenderTemplate`] + per-glb [`MeshTemplate`]
//! entries for one kind, dispatching on whether the kind's
//! [`RenderSpec`](currawong::RenderSpec) declares a hierarchical
//! [`nodes`](currawong::NodeSpec) block or sticks with the legacy flat
//! fields.
//!
//! Shared between
//! [`LumberEditorView::init`](crate::LumberEditorView::init) and
//! [`maybe_rebuild_templates`](crate::LumberEditorView::maybe_rebuild_templates)
//! so both paths produce structurally identical templates from the same
//! spec.
//!
//! The hierarchical branch delegates to the engine's
//! [`build_hierarchical_render_template`] — the same helper
//! [`examples/lumber_camp`](../lumber_camp) calls. The editor only owns
//! the flat-schema branch (its `MeshKey::KindBody(KindId)` slot has no
//! equivalent in the hierarchical schema) plus the per-glb fallback
//! albedo choice.

use std::collections::HashMap;

use currawong::data::{KindId, VfsPath};
use currawong::glam::Mat4;
use currawong::{
    AssetServer, MeshTemplate, PbrMaterial, PbrMaterialInstance, RenderSpec, RenderTemplate,
    Renderer, SamplerRegistry, build_hierarchical_render_template,
};

use crate::{MeshKey, Templates};

/// Fallback PBR albedo texture used by Mesh nodes whose
/// [`MeshNodeSpec::albedo`](currawong::MeshNodeSpec) is `None`. Reusing
/// the lumber atlas keeps the colour believable when a node's glb has
/// no material assignment.
const FALLBACK_ALBEDO_PATH: &str = "lumber/gradient_atlas.png";

/// Build the [`RenderTemplate`] for `kind_id` and register any
/// per-glb [`MeshTemplate`]s the hierarchical schema introduces. The
/// caller's `templates` registry gets one fresh entry; `mesh_templates`
/// gets one entry per unique mesh handle the template references.
///
/// `body` is the `streamed_kind_body_templates`-produced flat fallback;
/// it's consumed unconditionally for the flat schema and ignored when
/// the spec carries a hierarchical `nodes` block — that branch streams
/// per-Mesh-node templates by glb path through
/// [`build_hierarchical_render_template`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_template_for_kind(
    kind_id: &KindId,
    spec: &RenderSpec,
    body: MeshTemplate<PbrMaterialInstance>,
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
    mesh_templates: &mut HashMap<MeshKey, MeshTemplate<PbrMaterialInstance>>,
    templates: &mut Templates,
) {
    if spec.nodes.is_empty() {
        // Flat schema — the previous one-mesh-per-kind path.
        let bounds = body.visual_bounds;
        let body_key = MeshKey::KindBody(kind_id.clone());
        mesh_templates.insert(body_key.clone(), body);
        let template = RenderTemplate::new(kind_id.as_str())
            .with_mesh_part(body_key.clone(), body_key, Mat4::IDENTITY)
            .with_visual_bounds(bounds);
        templates.register(kind_id.clone(), template);
        return;
    }

    let fallback_albedo = VfsPath::new(FALLBACK_ALBEDO_PATH).expect("valid fallback albedo path");
    let label = format!("lumber_editor: kind {kind_id}");
    let template = build_hierarchical_render_template(
        &label,
        spec,
        MeshKey::Glb,
        &fallback_albedo,
        renderer,
        material,
        samplers,
        asset_server,
        mesh_templates,
    );
    templates.register(kind_id.clone(), template);
}
