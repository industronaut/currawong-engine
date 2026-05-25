//! [`MeshTemplate`] — bundled per-part GPU resources for the render-object
//! pipeline, plus the [`RenderSpec`] projection that lets a kind def's
//! `render:` block be the single source of truth for a streamed body's mesh +
//! texture + visual bounds.
//!
//! ## Where this fits
//!
//! Examples register one [`MeshTemplate`] per drawable *part* (the smallest
//! unit a [`RenderTemplate`](super::RenderTemplate) references through its
//! mesh-part list), keyed by some example-defined `PartKey`. The draw loop
//! looks up the template, calls [`MeshTemplate::resolve`] to get a primitive
//! slice + fallback adjustment, and binds the material.
//!
//! The two PBR-flavoured constructors —
//! [`PbrMaterial::streamed_template`] and [`PbrMaterial::inline_template`] —
//! are the conventional way to build one. Future material families can add
//! their own constructors with the same shape.

use std::collections::HashMap;
use std::hash::Hash;

use bytemuck::cast_slice;
use glam::{Mat4, Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;

use crate::data::{Definitions, KindDef, KindId, VfsPath};

use super::asset_server::{AssetServer, MeshSource, ResolvedMesh};
use super::handle::Handle;
use super::mesh::{Mesh, MeshPrimitive};
use super::mesh_primitives::PrimitiveMesh;
use super::pbr::{PbrMaterial, PbrMaterialInstance, PbrMaterialParams};
use super::render_object::{MeshPart, NodeId, NodeKind, RenderTemplate, TemplateNode};
use super::renderer::Renderer;
use super::texture::{SamplerKind, SamplerRegistry, Texture, TextureColorSpace};
use super::visibility::Aabb;

/// Bundled GPU resources for one drawable part: mesh buffers (streamed or
/// inline), the visual AABB used for fallback sizing and culling, and a live
/// material instance to draw them with. One per `PartKey` in the render-object
/// pipeline.
///
/// Generic over the material instance type so the same shape works for PBR,
/// stylized PBR, unlit, or any other material with a per-part instance.
pub struct MeshTemplate<M> {
    pub mesh: MeshBacking,
    pub visual_bounds: Aabb,
    pub material: M,
}

/// Where a [`MeshTemplate`] gets its mesh buffers from.
///
/// `Streamed` defers buffer ownership to the [`AssetServer`] — bound through a
/// [`Handle<Mesh>`] that resolves to magenta-flavoured fallback geometry while
/// loading. `Inline` owns the buffers directly: no streaming, no fallback. Use
/// inline for procedural ancillary parts (markers, carried items) that don't
/// go through the asset pipeline.
pub enum MeshBacking {
    Streamed { handle: Handle<Mesh> },
    Inline { primitives: Vec<MeshPrimitive> },
}

impl<M> MeshTemplate<M> {
    /// Resolve the template's mesh buffers + fallback adjustment for this
    /// frame's draw. Equivalent shape regardless of backing: a non-empty
    /// [`MeshPrimitive`] slice + a [`Mat4`] to compose inside the per-instance
    /// world transform. For inline templates the adjustment is identity and
    /// the source tag is always [`MeshSource::Real`].
    pub fn resolve<'a>(&'a self, asset_server: &'a AssetServer) -> ResolvedMesh<'a> {
        match &self.mesh {
            MeshBacking::Streamed { handle } => {
                asset_server.resolve_mesh(handle, Some(self.visual_bounds))
            }
            MeshBacking::Inline { primitives } => ResolvedMesh {
                primitives,
                source: MeshSource::Real,
                fallback_adjustment: Mat4::IDENTITY,
            },
        }
    }
}

/// View-side projection of a kind def's `render:` block. The single source of
/// truth for what to draw for a given [`KindId`]: mesh path, albedo path, PBR
/// factors, and the visual AABB used for culling + fallback sizing.
///
/// Sim-side kind body structs typically pick out their own sim-relevant fields
/// with a separate `Deserialize` shim — serde silently drops the rest, so the
/// two projections stay independent.
///
/// ```ron
/// (
///     id: "currawong:oak_tree",
///     render: (
///         shape: "tree",
///         mesh: "trees/oak.glb",
///         albedo: "trees/oak_bark.png",
///         metallic: 0.0,
///         roughness: 0.9,
///         bounds_min: (-0.6, -0.6, 0.0),
///         bounds_max: (0.6, 0.6, 3.4),
///     ),
///     // ... sim fields ...
/// )
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RenderSpec {
    /// View-side dispatch tag. The example chooses a closed set (e.g. `"tree"
    /// | "pawn" | "building"`) and uses it to pick a factory that decides the
    /// template's structural layout. Stored as a `String` so the engine
    /// doesn't constrain the example's tag set.
    pub shape: String,
    /// VFS path to the streamed glTF body.
    pub mesh: String,
    /// VFS path to the streamed albedo texture (sRGB).
    pub albedo: String,
    /// `0.0` = dielectric (plastic, wood), `1.0` = metal.
    pub metallic: f32,
    /// `0.04..1.0` (clamped in shader). Lower = sharper highlights.
    pub roughness: f32,
    pub bounds_min: (f32, f32, f32),
    pub bounds_max: (f32, f32, f32),
    /// Optional hierarchical node tree authored in the kind def, parallel
    /// to the flat `mesh`/`albedo`/`bounds_*` fields above. When
    /// non-empty, view-side template construction walks these nodes
    /// instead of synthesising a one-mesh template from the flat fields.
    /// The flat fields stay populated either way — sim consumers
    /// (`Game::render_specs`, recalc-bounds, overlays) read them
    /// directly. New on the hierarchical-render-templates branch
    /// (Phase 6); existing kind files omit the field and parse with an
    /// empty vec.
    #[serde(default)]
    pub nodes: Vec<NodeSpec>,
}

/// Translation / rotation (xyzw quaternion) / scale tuple — the
/// serialisable form of an [`Mat4`] local-transform that round-trips
/// through `Mat4::from_scale_rotation_translation` /
/// `Mat4::to_scale_rotation_translation`.
///
/// Defaulted to identity so kind defs can omit unchanged transforms.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct TransformSpec {
    pub translation: (f32, f32, f32),
    /// Quaternion as `(x, y, z, w)`. Defaults to identity `(0, 0, 0, 1)`.
    pub rotation: (f32, f32, f32, f32),
    pub scale: (f32, f32, f32),
}

impl Default for TransformSpec {
    fn default() -> Self {
        Self {
            translation: (0.0, 0.0, 0.0),
            rotation: (0.0, 0.0, 0.0, 1.0),
            scale: (1.0, 1.0, 1.0),
        }
    }
}

impl TransformSpec {
    /// Decompose a [`Mat4`] for serialisation. Inverse of [`Self::to_mat4`].
    pub fn from_mat4(m: Mat4) -> Self {
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        Self {
            translation: translation.into(),
            rotation: (rotation.x, rotation.y, rotation.z, rotation.w),
            scale: scale.into(),
        }
    }

    /// Compose the local transform back into a [`Mat4`].
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::from(self.scale),
            Quat::from_xyzw(
                self.rotation.0,
                self.rotation.1,
                self.rotation.2,
                self.rotation.3,
            ),
            Vec3::from(self.translation),
        )
    }
}

/// Material parameters for a [`NodeKindSpec::Mesh`] node — the
/// per-node analogue of the flat `mesh` / `albedo` / `metallic` /
/// `roughness` fields on [`RenderSpec`]. `albedo` is optional so a node
/// can fall back to the example's standard atlas; `metallic` and
/// `roughness` have PBR-default values when omitted.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MeshNodeSpec {
    /// VFS path to the streamed glTF.
    pub mesh: String,
    /// VFS path to the streamed albedo texture (sRGB). Optional —
    /// editor's "Add mesh from glb" leaves this `None` and the consumer
    /// falls back to whatever default it uses for grafted geometry.
    #[serde(default)]
    pub albedo: Option<String>,
    /// `0.0` = dielectric, `1.0` = metal. Defaults to `0.0`.
    #[serde(default)]
    pub metallic: f32,
    /// `0.04..1.0`. Defaults to `0.85`.
    #[serde(default = "default_roughness")]
    pub roughness: f32,
}

fn default_roughness() -> f32 {
    0.85
}

/// Tag string for the [`NodeSpec`] payload — `"empty"` for an
/// attachment-only node, `"mesh"` for a [`MeshNodeSpec`] payload.
/// `"emitter"` is reserved for the future emitter-in-editor pass.
///
/// Modelled as a tag string rather than an enum so the schema
/// round-trips through `ron::Value::into_rust` — ron 0.8's `Value` has
/// no enum-payload variant, so a Rust `enum NodeKindSpec { Empty,
/// Mesh(MeshNodeSpec) }` would fail to deserialise via the existing
/// `KindDef.value` path.
pub mod node_kind {
    pub const EMPTY: &str = "empty";
    pub const MESH: &str = "mesh";
}

/// Serialisable form of a [`TemplateNode`](super::TemplateNode) authored
/// in a kind def's hierarchical `render.nodes` block. Carries the
/// stable [`NodeId`](super::NodeId) (`u16`), a display name, an optional
/// parent id (`None` = root), a local transform (defaults to identity),
/// and a [`node_kind`]-tagged payload.
///
/// `mesh` is `Some` iff `kind == "mesh"`. Validation lives at the
/// editor's template-build seam — schema-level checks would otherwise
/// be duplicated across every consumer.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NodeSpec {
    /// Stable [`NodeId`](super::NodeId) within this template. Authored
    /// at editor time and persisted verbatim across saves.
    pub id: u16,
    pub name: String,
    /// Parent id; `None` for roots. Parents must precede children in
    /// the `nodes` vec (matches the
    /// [`RenderTemplate`](super::RenderTemplate) parent-first
    /// invariant).
    #[serde(default)]
    pub parent: Option<u16>,
    #[serde(default)]
    pub transform: TransformSpec,
    /// One of [`node_kind::EMPTY`] or [`node_kind::MESH`].
    pub kind: String,
    /// Mesh-node payload — present iff [`kind`](Self::kind) is
    /// `"mesh"`. Editor consumers panic / log on a `"mesh"` kind with
    /// no payload.
    #[serde(default)]
    pub mesh: Option<MeshNodeSpec>,
}

impl RenderSpec {
    pub fn visual_bounds(&self) -> Aabb {
        Aabb::new(
            Vec3::new(self.bounds_min.0, self.bounds_min.1, self.bounds_min.2),
            Vec3::new(self.bounds_max.0, self.bounds_max.1, self.bounds_max.2),
        )
    }

    /// Parse the `render` block out of a kind def's raw value.
    ///
    /// Returns `Err` for kinds whose def doesn't have a `render` field, or
    /// whose `render` field is malformed. Examples typically `eprintln!` +
    /// skip so the rest of the world loads — rules-only kinds (recipes,
    /// faction markers) are the expected reason to be skipped.
    pub fn from_def(def: &KindDef) -> Result<Self, ron::Error> {
        #[derive(Deserialize)]
        struct Body {
            render: RenderSpec,
        }
        let body: Body = def.value.clone().into_rust()?;
        Ok(body.render)
    }
}

/// Parameters for [`PbrMaterial::inline_template`] — bundled into a struct
/// because the positional arg list otherwise crosses clippy's
/// `too_many_arguments` threshold.
pub struct InlineTemplate<'a> {
    pub label: &'static str,
    pub mesh: &'a PrimitiveMesh,
    pub bounds: Aabb,
    /// Flat colour multiplier — inline parts don't stream a texture so this
    /// is the only place their colour comes from.
    pub albedo_factor: Vec4,
    pub metallic: f32,
    pub roughness: f32,
}

impl PbrMaterial {
    /// Build a streamed [`MeshTemplate`] for the kind named by `kind_id`,
    /// taking its mesh path + albedo path + PBR factors + visual bounds from
    /// `spec`. The mesh and texture stream through `asset_server`; while
    /// loading, the template draws the magenta-flavoured fallback sized to
    /// `spec.visual_bounds()`.
    ///
    /// Panics if `spec.mesh` or `spec.albedo` isn't a valid [`VfsPath`] — that
    /// is a content authoring bug rather than a runtime condition, and silent
    /// fallback would hide it.
    pub fn streamed_template(
        &self,
        renderer: &Renderer,
        samplers: &SamplerRegistry,
        asset_server: &AssetServer,
        kind_id: &KindId,
        spec: &RenderSpec,
    ) -> MeshTemplate<PbrMaterialInstance> {
        let mesh_path = VfsPath::new(spec.mesh.clone())
            .unwrap_or_else(|e| panic!("kind {kind_id}: invalid render.mesh path: {e}"));
        let albedo_path = VfsPath::new(spec.albedo.clone())
            .unwrap_or_else(|e| panic!("kind {kind_id}: invalid render.albedo path: {e}"));
        let mesh_handle = asset_server.mesh(mesh_path);
        let albedo_handle = asset_server.texture(albedo_path, TextureColorSpace::Srgb);
        let material = self.create_instance(
            renderer,
            samplers,
            asset_server,
            PbrMaterialParams {
                albedo: albedo_handle,
                sampler: SamplerKind::LinearRepeat,
                albedo_factor: Vec4::ONE,
                metallic: spec.metallic,
                roughness: spec.roughness,
            },
        );
        MeshTemplate {
            mesh: MeshBacking::Streamed {
                handle: mesh_handle,
            },
            visual_bounds: spec.visual_bounds(),
            material,
        }
    }

    /// Walk every kind in `defs` and build a streamed body
    /// [`MeshTemplate`] for each one that has a parseable `render:` block.
    /// Skipped kinds (no `render` block, or a malformed one) are reported
    /// through `on_skip` and absent from the result.
    ///
    /// Returns `(KindId, RenderSpec, MeshTemplate)` triples in iteration
    /// order over `defs`. Callers typically:
    /// - Collect the templates into the `HashMap<PartKey, MeshTemplate>` the
    ///   draw loop consumes.
    /// - Use the spec to populate a per-kind [`RenderTemplate`](super::RenderTemplate)
    ///   (visual bounds, shape-specific extra parts, etc).
    ///
    /// Pass `|_, _| {}` for `on_skip` to ignore parse errors silently — the
    /// right shape when the sim side already validated and logged them.
    pub fn streamed_kind_body_templates(
        &self,
        renderer: &Renderer,
        samplers: &SamplerRegistry,
        asset_server: &AssetServer,
        defs: &Definitions,
        mut on_skip: impl FnMut(&KindId, ron::Error),
    ) -> Vec<(KindId, RenderSpec, MeshTemplate<PbrMaterialInstance>)> {
        defs.iter()
            .filter_map(|(kind_id, def)| match RenderSpec::from_def(def) {
                Ok(spec) => {
                    let body =
                        self.streamed_template(renderer, samplers, asset_server, kind_id, &spec);
                    Some((kind_id.clone(), spec, body))
                }
                Err(e) => {
                    on_skip(kind_id, e);
                    None
                }
            })
            .collect()
    }

    /// Build an inline [`MeshTemplate`] from a [`PrimitiveMesh`] + flat albedo
    /// factor. Shared helper for procedural ancillary parts (markers, carried
    /// items, gizmos) that don't go through the asset pipeline — they still
    /// plug into the same PBR material surface streamed bodies use, via a 1×1
    /// white texture wrapped in a ready [`Handle`].
    pub fn inline_template(
        &self,
        renderer: &Renderer,
        samplers: &SamplerRegistry,
        asset_server: &AssetServer,
        params: InlineTemplate<'_>,
    ) -> MeshTemplate<PbrMaterialInstance> {
        let vertex_buffer = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(params.label),
                contents: cast_slice(&params.mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(params.label),
                contents: cast_slice(&params.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let primitive = MeshPrimitive {
            vertex_buffer,
            index_buffer,
            index_count: params.mesh.index_count(),
            material_name: None,
        };
        let white = Texture::from_rgba8(renderer, params.label, 1, 1, &[255; 4], true);
        let material = self.create_instance(
            renderer,
            samplers,
            asset_server,
            PbrMaterialParams {
                albedo: Handle::ready(white),
                sampler: SamplerKind::LinearClamp,
                albedo_factor: params.albedo_factor,
                metallic: params.metallic,
                roughness: params.roughness,
            },
        );
        MeshTemplate {
            mesh: MeshBacking::Inline {
                primitives: vec![primitive],
            },
            visual_bounds: params.bounds,
            material,
        }
    }
}

// --- Hierarchical template builder -------------------------------------
//
// Shared core between examples that adopt the hierarchical
// `render.nodes` schema. Today: lumber_camp and lumber_editor — both
// walk the spec's NodeSpecs, dedupe streamed MeshTemplates by glb path,
// and produce a parent-linked RenderTemplate. Example-specific concerns
// (the flat-schema fallback, ancillary parts like markers / carried
// logs, editor grafting state) stay in the example.

/// Fallback half-extent for a per-glb [`MeshTemplate`]'s `visual_bounds`
/// while the real mesh streams in. Only matters for the few frames
/// between handle creation and the [`AssetServer`] publishing real
/// geometry — the composite render object's visual bounds come from the
/// spec's flat `bounds_min/max`, so this number doesn't affect culling.
const HIERARCHICAL_FALLBACK_BOUNDS_HALF_EXTENT: f32 = 0.5;

/// Build a hierarchical [`RenderTemplate`] from `spec.nodes`, registering
/// one streamed [`MeshTemplate`] per unique glb path in `mesh_templates`
/// (deduped — multiple nodes referencing the same path share storage).
///
/// `glb_key` converts a node's glb [`VfsPath`] into the caller's part-key
/// type, e.g. `|p| PartKey::Glb(p)`. `fallback_albedo` is bound on Mesh
/// nodes whose [`MeshNodeSpec::albedo`] is `None` (or malformed); the
/// engine doesn't ship a default texture, so the caller picks one
/// appropriate to its example.
///
/// `label` prefixes any malformed-node warnings — typically
/// `"lumber_camp: kind {kind_id}"` so the error message points at the
/// caller and the kind.
///
/// `visual_bounds` is taken from `spec.visual_bounds()` (the flat
/// `bounds_min/max` fields) so the engine cull sees the same region
/// whether the kind uses the flat or hierarchical schema.
///
/// Returns the assembled template; mutates `mesh_templates` with any
/// newly-streamed glb entries.
#[allow(clippy::too_many_arguments)]
pub fn build_hierarchical_render_template<K>(
    label: &str,
    spec: &RenderSpec,
    glb_key: impl Fn(VfsPath) -> K,
    fallback_albedo: &VfsPath,
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
    mesh_templates: &mut HashMap<K, MeshTemplate<PbrMaterialInstance>>,
) -> RenderTemplate<K, K>
where
    K: Clone + Eq + Hash,
{
    let mut template: RenderTemplate<K, K> =
        RenderTemplate::new(label).with_visual_bounds(spec.visual_bounds());
    for node_spec in &spec.nodes {
        let kind = match node_spec.kind.as_str() {
            node_kind::EMPTY => NodeKind::Empty,
            node_kind::MESH => build_mesh_node_kind(
                label,
                node_spec,
                &glb_key,
                fallback_albedo,
                renderer,
                material,
                samplers,
                asset_server,
                mesh_templates,
            )
            .unwrap_or(NodeKind::Empty),
            other => {
                eprintln!(
                    "{label} node {} unknown kind tag `{other}`; treating as empty",
                    node_spec.id
                );
                NodeKind::Empty
            }
        };
        template.add_node(TemplateNode {
            id: NodeId(node_spec.id),
            name: node_spec.name.clone(),
            parent: node_spec.parent.map(NodeId),
            local_transform: node_spec.transform.to_mat4(),
            kind,
        });
    }
    template
}

/// Resolve one Mesh node's payload into a [`NodeKind::Mesh`], populating
/// `mesh_templates` on cache miss. Returns `None` when the node's
/// payload is missing or malformed — the caller substitutes
/// [`NodeKind::Empty`] so the rest of the tree still loads.
#[allow(clippy::too_many_arguments)]
fn build_mesh_node_kind<K>(
    label: &str,
    node_spec: &NodeSpec,
    glb_key: &impl Fn(VfsPath) -> K,
    fallback_albedo: &VfsPath,
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
    mesh_templates: &mut HashMap<K, MeshTemplate<PbrMaterialInstance>>,
) -> Option<NodeKind<K, K>>
where
    K: Clone + Eq + Hash,
{
    let Some(mesh_spec) = node_spec.mesh.as_ref() else {
        eprintln!(
            "{label} node {} declares \"mesh\" with no payload; treating as empty",
            node_spec.id
        );
        return None;
    };
    let mesh_path = match VfsPath::new(&mesh_spec.mesh) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "{label} node {} bad mesh path `{}`: {e}",
                node_spec.id, mesh_spec.mesh
            );
            return None;
        }
    };
    let key = glb_key(mesh_path.clone());
    mesh_templates.entry(key.clone()).or_insert_with(|| {
        let albedo_path = mesh_spec
            .albedo
            .as_deref()
            .and_then(|a| match VfsPath::new(a) {
                Ok(p) => Some(p),
                Err(e) => {
                    eprintln!(
                        "{label} node {} bad albedo path `{a}`: {e}; using fallback",
                        node_spec.id
                    );
                    None
                }
            })
            .unwrap_or_else(|| fallback_albedo.clone());
        build_streamed_pbr_mesh_template(
            renderer,
            material,
            samplers,
            asset_server,
            mesh_path,
            albedo_path,
            mesh_spec.metallic,
            mesh_spec.roughness,
        )
    });
    Some(NodeKind::Mesh(MeshPart::new(key.clone(), key)))
}

/// Build a streamed-PBR [`MeshTemplate`] for one glb path. Used by the
/// hierarchical walk; exposed for examples that want to register
/// glb-keyed templates outside the hierarchical builder (e.g. the
/// editor's "Add mesh from glb" action that lands a new node
/// imperatively rather than through a spec walk).
#[allow(clippy::too_many_arguments)]
pub fn build_streamed_pbr_mesh_template(
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
    mesh_path: VfsPath,
    albedo_path: VfsPath,
    metallic: f32,
    roughness: f32,
) -> MeshTemplate<PbrMaterialInstance> {
    let mesh_handle = asset_server.mesh(mesh_path);
    let albedo_handle = asset_server.texture(albedo_path, TextureColorSpace::Srgb);
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
            Vec3::splat(-HIERARCHICAL_FALLBACK_BOUNDS_HALF_EXTENT),
            Vec3::splat(HIERARCHICAL_FALLBACK_BOUNDS_HALF_EXTENT),
        ),
        material: material_instance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_spec_visual_bounds_round_trips() {
        let spec = RenderSpec {
            shape: "tree".into(),
            mesh: "trees/oak.glb".into(),
            albedo: "trees/oak_bark.png".into(),
            metallic: 0.0,
            roughness: 0.9,
            bounds_min: (-0.6, -0.6, 0.0),
            bounds_max: (0.6, 0.6, 3.4),
            nodes: Vec::new(),
        };
        let bounds = spec.visual_bounds();
        assert_eq!(bounds.min, Vec3::new(-0.6, -0.6, 0.0));
        assert_eq!(bounds.max, Vec3::new(0.6, 0.6, 3.4));
    }

    #[test]
    fn render_spec_deserialises_from_kind_def_value() {
        use crate::data::{KindDef, KindId, VfsPath};
        let ron_text = r#"(
            id: "currawong:oak_tree",
            render: (
                shape: "tree",
                mesh: "trees/oak.glb",
                albedo: "trees/oak_bark.png",
                metallic: 0.0,
                roughness: 0.9,
                bounds_min: (-0.6, -0.6, 0.0),
                bounds_max: (0.6, 0.6, 3.4),
            ),
            extra_sim_field: 42,
        )"#;
        let value: ron::Value = ron::from_str(ron_text).expect("parse");
        let def = KindDef {
            id: KindId::new("currawong:oak_tree").expect("valid id"),
            source: VfsPath::new("kinds/oak.ron").expect("valid path"),
            value,
        };
        let spec = RenderSpec::from_def(&def).expect("render block parses");
        assert_eq!(spec.shape, "tree");
        assert_eq!(spec.mesh, "trees/oak.glb");
        assert_eq!(spec.roughness, 0.9);
    }

    #[test]
    fn render_spec_parses_with_hierarchical_nodes_block() {
        use crate::data::{KindDef, KindId, VfsPath};
        let ron_text = r#"(
            id: "currawong:tank",
            render: (
                shape: "building",
                mesh: "tanks/chassis.glb",
                albedo: "tanks/chassis_albedo.png",
                metallic: 0.6,
                roughness: 0.4,
                bounds_min: (-1.5, -1.5, 0.0),
                bounds_max: (1.5, 1.5, 1.4),
                nodes: [
                    (
                        id: 0,
                        name: "chassis",
                        parent: None,
                        kind: "mesh",
                        mesh: Some((
                            mesh: "tanks/chassis.glb",
                        )),
                    ),
                    (
                        id: 1,
                        name: "turret",
                        parent: Some(0),
                        transform: (
                            translation: (0.0, 0.0, 0.5),
                            rotation: (0.0, 0.0, 0.0, 1.0),
                            scale: (1.0, 1.0, 1.0),
                        ),
                        kind: "mesh",
                        mesh: Some((
                            mesh: "tanks/turret.glb",
                            metallic: 0.8,
                            roughness: 0.3,
                        )),
                    ),
                ],
            ),
        )"#;
        let value: ron::Value = ron::from_str(ron_text).expect("parse");
        let def = KindDef {
            id: KindId::new("currawong:tank").expect("valid id"),
            source: VfsPath::new("kinds/tank.ron").expect("valid path"),
            value,
        };
        let spec = RenderSpec::from_def(&def).expect("render block parses");
        let nodes = &spec.nodes;
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].id, 0);
        assert_eq!(nodes[0].parent, None);
        assert_eq!(nodes[0].kind, node_kind::MESH);
        let mesh = nodes[1].mesh.as_ref().expect("mesh payload");
        assert_eq!(mesh.mesh, "tanks/turret.glb");
        assert_eq!(mesh.metallic, 0.8);
        assert_eq!(mesh.roughness, 0.3);
        assert_eq!(nodes[1].parent, Some(0));
        assert_eq!(nodes[1].transform.translation, (0.0, 0.0, 0.5));
    }

    #[test]
    fn render_spec_parses_empty_node_kind_and_defaults() {
        use crate::data::{KindDef, KindId, VfsPath};
        let ron_text = r#"(
            id: "currawong:rig",
            render: (
                shape: "building",
                mesh: "rig.glb",
                albedo: "rig.png",
                metallic: 0.0,
                roughness: 0.9,
                bounds_min: (-1.0, -1.0, 0.0),
                bounds_max: (1.0, 1.0, 1.0),
                nodes: [
                    (
                        id: 0,
                        name: "attach",
                        kind: "empty",
                    ),
                ],
            ),
        )"#;
        let value: ron::Value = ron::from_str(ron_text).expect("parse");
        let def = KindDef {
            id: KindId::new("currawong:rig").expect("valid id"),
            source: VfsPath::new("kinds/rig.ron").expect("valid path"),
            value,
        };
        let spec = RenderSpec::from_def(&def).expect("render block parses");
        let nodes = &spec.nodes;
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].kind, node_kind::EMPTY);
        assert!(nodes[0].mesh.is_none());
        // Default parent + default transform fill in for the omitted fields.
        assert_eq!(nodes[0].parent, None);
        assert_eq!(nodes[0].transform, TransformSpec::default());
    }

    #[test]
    fn transform_spec_round_trips_through_mat4() {
        let original = TransformSpec {
            translation: (1.0, 2.0, 3.0),
            // 30° around Y, decomposed: (0, sin(15°), 0, cos(15°))
            rotation: (0.0, 0.258819, 0.0, 0.9659258),
            scale: (0.5, 0.5, 0.5),
        };
        let mat = original.to_mat4();
        let round_tripped = TransformSpec::from_mat4(mat);
        // Approximate equality — to_scale_rotation_translation can pick a
        // different but equivalent quaternion branch, so just check the
        // composed matrices match.
        assert!(
            (mat - round_tripped.to_mat4()).abs_diff_eq(Mat4::ZERO, 1e-5),
            "TransformSpec round-trip should preserve the composed matrix",
        );
    }

    #[test]
    fn render_spec_from_def_rejects_def_without_render_block() {
        use crate::data::{KindDef, KindId, VfsPath};
        let ron_text = r#"(id: "currawong:recipe_plank", inputs: ["log"], output: "plank")"#;
        let value: ron::Value = ron::from_str(ron_text).expect("parse");
        let def = KindDef {
            id: KindId::new("currawong:recipe_plank").expect("valid id"),
            source: VfsPath::new("kinds/recipe.ron").expect("valid path"),
            value,
        };
        assert!(
            RenderSpec::from_def(&def).is_err(),
            "rules-only kinds must surface the missing-render-block as Err"
        );
    }
}
