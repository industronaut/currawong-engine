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
//! Free function rather than a method because `init` is composing the
//! eventual `self` from a stack of local variables that aren't yet
//! fields — the function takes the underlying registries by `&mut`
//! reference and works against any combination of locals or fields.

use std::collections::HashMap;

use currawong::data::{KindId, VfsPath};
use currawong::glam::{Mat4, Vec3, Vec4};
use currawong::{
    Aabb, AssetServer, MeshBacking, MeshPart, MeshTemplate, NodeId, NodeKind, NodeSpec,
    PbrMaterial, PbrMaterialInstance, PbrMaterialParams, RenderSpec, RenderTemplate, Renderer,
    SamplerKind, SamplerRegistry, TemplateNode, TextureColorSpace, node_kind,
};

use crate::{MeshKey, Templates};

/// Fallback bounds for a streamed glb [`MeshTemplate`] while the real
/// mesh loads. The real bounds take over once the
/// [`Handle`](currawong::Handle) resolves; the fallback only matters in
/// the first few frames after registration.
const FALLBACK_BOUNDS_HALF_EXTENT: f32 = 0.5;

/// Fallback PBR albedo texture used by Mesh nodes that don't supply
/// their own. Reusing the lumber atlas keeps the colour believable
/// when a node's glb has no material assignment.
const FALLBACK_ALBEDO_PATH: &str = "lumber/gradient_atlas.png";

/// Build the [`RenderTemplate`] for `kind_id` and register any
/// per-glb [`MeshTemplate`]s the hierarchical schema introduces. The
/// caller's `templates` registry gets one fresh entry; `mesh_templates`
/// gets one entry per unique mesh handle the template references.
///
/// `body` is the `streamed_kind_body_templates`-produced flat fallback;
/// it's consumed unconditionally for the flat schema and ignored when
/// the spec carries a hierarchical `nodes` block — that branch builds
/// its own per-Mesh-node templates.
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

    let mut template: RenderTemplate<MeshKey, MeshKey> = RenderTemplate::new(kind_id.as_str());
    // Preserve the flat-schema visual_bounds so the engine cull and the
    // editor's bounding-box overlay still have a region to test against.
    template = template.with_visual_bounds(spec.visual_bounds());

    for node_spec in &spec.nodes {
        let kind = build_node_kind(
            node_spec,
            kind_id,
            renderer,
            material,
            samplers,
            asset_server,
            mesh_templates,
        );
        template.add_node(TemplateNode {
            id: NodeId(node_spec.id),
            name: node_spec.name.clone(),
            parent: node_spec.parent.map(NodeId),
            local_transform: node_spec.transform.to_mat4(),
            kind,
        });
    }
    templates.register(kind_id.clone(), template);
}

/// Translate one [`NodeSpec`] into a runtime
/// [`NodeKind<MeshKey, MeshKey>`], registering its `MeshTemplate` in
/// `mesh_templates` if the kind is `"mesh"` and we haven't already
/// streamed the glb.
///
/// Returns [`NodeKind::Empty`] on any failure (unknown `kind` tag,
/// invalid glb path, missing payload). Errors are logged via
/// `eprintln!` so a malformed kind def doesn't crash the editor.
fn build_node_kind(
    node_spec: &NodeSpec,
    kind_id: &KindId,
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
    mesh_templates: &mut HashMap<MeshKey, MeshTemplate<PbrMaterialInstance>>,
) -> NodeKind<MeshKey, MeshKey> {
    match node_spec.kind.as_str() {
        node_kind::EMPTY => NodeKind::Empty,
        node_kind::MESH => {
            let Some(mesh_spec) = node_spec.mesh.as_ref() else {
                eprintln!(
                    "lumber_editor: kind {kind_id} node {} declares kind \"mesh\" but has no payload; treating as empty",
                    node_spec.id,
                );
                return NodeKind::Empty;
            };
            let path = match VfsPath::new(&mesh_spec.mesh) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!(
                        "lumber_editor: kind {kind_id} node {} bad mesh path `{}`: {e}",
                        node_spec.id, mesh_spec.mesh,
                    );
                    return NodeKind::Empty;
                }
            };
            let key = MeshKey::Glb(path.clone());
            mesh_templates.entry(key.clone()).or_insert_with(|| {
                build_glb_mesh_template(
                    renderer,
                    material,
                    samplers,
                    asset_server,
                    path,
                    mesh_spec.albedo.as_deref().unwrap_or(FALLBACK_ALBEDO_PATH),
                    mesh_spec.metallic,
                    mesh_spec.roughness,
                )
            });
            NodeKind::Mesh(MeshPart::new(key.clone(), key))
        }
        other => {
            eprintln!(
                "lumber_editor: kind {kind_id} node {} unknown kind tag `{other}`; treating as empty",
                node_spec.id,
            );
            NodeKind::Empty
        }
    }
}

/// Build a streamed [`MeshTemplate`] for one glb path with PBR material
/// parameters. Shared between this module and
/// [`glb_import`](crate::glb_import) (which uses an inline copy for the
/// scene-panel "Add mesh from glb" action — same shape, different
/// trigger).
#[allow(clippy::too_many_arguments)]
fn build_glb_mesh_template(
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
    mesh_path: VfsPath,
    albedo_path: &str,
    metallic: f32,
    roughness: f32,
) -> MeshTemplate<PbrMaterialInstance> {
    let mesh_handle = asset_server.mesh(mesh_path);
    let albedo_handle = asset_server.texture(
        VfsPath::new(albedo_path).expect("valid fallback albedo path"),
        TextureColorSpace::Srgb,
    );
    let material_instance = material.create_instance(
        renderer,
        samplers,
        asset_server,
        PbrMaterialParams {
            albedo: albedo_handle,
            sampler: SamplerKind::LinearRepeat,
            albedo_factor: Vec4::ONE,
            metallic,
            roughness,
        },
    );
    MeshTemplate {
        mesh: MeshBacking::Streamed {
            handle: mesh_handle,
        },
        visual_bounds: Aabb::new(
            Vec3::splat(-FALLBACK_BOUNDS_HALF_EXTENT),
            Vec3::splat(FALLBACK_BOUNDS_HALF_EXTENT),
        ),
        material: material_instance,
    }
}
