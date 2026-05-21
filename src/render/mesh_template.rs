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
//! slice + fallback adjustment, and binds each primitive's material from
//! `template.materials[i]`.
//!
//! ## Per-primitive materials
//!
//! Templates store one [`Arc<dyn MeshMaterial>`](super::MeshMaterial) per
//! primitive in [`MeshTemplate::materials`], so a multi-material glb routes
//! each primitive through the right pipeline + bind group with no per-frame
//! dispatch. For inline templates the vec is filled at build time. For
//! streamed templates it stays empty until the glb decodes, then
//! [`MeshTemplate::resolve_materials`] resolves each primitive's
//! [`material_name`](super::MeshPrimitive::material_name) against the
//! application's [`MaterialRegistry`](super::MaterialRegistry); misses fall
//! back to `fallback_material`.
//!
//! The two PBR-flavoured constructors —
//! [`PbrMaterial::streamed_template`] and [`PbrMaterial::inline_template`] —
//! are the conventional way to build one. Future material families can add
//! their own constructors with the same shape.

use std::sync::Arc;

use bytemuck::cast_slice;
use glam::{Mat4, Vec3, Vec4};
use serde::Deserialize;
use wgpu::util::DeviceExt;

use crate::data::{KindDef, KindId, VfsPath};

use super::asset_server::{AssetServer, MeshSource, ResolvedMesh};
use super::handle::Handle;
use super::material::MeshMaterial;
use super::material_registry::MaterialRegistry;
use super::mesh::{Mesh, MeshPrimitive};
use super::mesh_primitives::PrimitiveMesh;
use super::pbr::{PbrMaterial, PbrMaterialParams};
use super::renderer::Renderer;
use super::texture::{SamplerKind, SamplerRegistry, Texture, TextureColorSpace};
use super::visibility::Aabb;

/// Bundled GPU resources for one drawable part: mesh buffers (streamed or
/// inline), the visual AABB used for fallback sizing and culling, and per-
/// primitive materials. One per `PartKey` in the render-object pipeline.
///
/// `materials[i]` is the resolved [`MeshMaterial`] for the i-th primitive in
/// the underlying mesh — pre-resolved once (at build time for inline
/// templates, or on the frame the streamed glb decodes via
/// [`Self::resolve_materials`]) so the per-frame draw is a uniform
/// `materials[i].bind(pass, 2)` walk with no concrete-type dispatch.
///
/// `fallback_material` covers two cases:
/// - the streamed glb is still loading and `materials` is empty;
/// - a resolved primitive's `material_name` didn't match any entry in the
///   registry (typical for a glb whose author didn't namespace their slot
///   names, or for the asset server's magenta unit-cube fallback whose
///   primitive has `material_name = None`).
pub struct MeshTemplate {
    pub mesh: MeshBacking,
    pub visual_bounds: Aabb,
    /// Per-primitive resolved materials. Empty for streamed templates
    /// until the glb decodes (then filled by
    /// [`Self::resolve_materials`]); pre-filled at build time for inline
    /// templates.
    pub materials: Vec<Arc<dyn MeshMaterial>>,
    /// Material the draw loop falls through to when `materials` is empty
    /// (streaming) or `materials[i]` was a registry miss. Always present.
    pub fallback_material: Arc<dyn MeshMaterial>,
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

impl MeshTemplate {
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

    /// Resolve per-primitive material slot names against `registry` and
    /// populate [`Self::materials`]. No-op when `materials.len()` already
    /// matches the primitive count — call every frame; the work only
    /// happens on the frame a streamed glb's
    /// [`MeshSource`](super::MeshSource) flips to `Real`.
    ///
    /// Primitives whose `material_name` is `None` or doesn't resolve in
    /// the registry get [`fallback_material`](Self::fallback_material) —
    /// same Arc the empty-materials path uses, so the bind side is
    /// uniform.
    pub fn resolve_materials(
        &mut self,
        asset_server: &AssetServer,
        registry: &MaterialRegistry<Arc<dyn MeshMaterial>>,
    ) {
        let resolved = self.resolve(asset_server);
        // Only resolve once the *real* glb has decoded. While the asset
        // server is still serving the magenta unit-cube fallback, leaving
        // `materials` empty routes draws through `fallback_material`
        // which is the intended shape.
        if resolved.source != MeshSource::Real {
            return;
        }
        if self.materials.len() == resolved.primitives.len() {
            return;
        }
        self.materials = resolved
            .primitives
            .iter()
            .map(|prim| {
                prim.material_name
                    .as_deref()
                    .and_then(|n| registry.get_by_name(n))
                    .cloned()
                    .unwrap_or_else(|| Arc::clone(&self.fallback_material))
            })
            .collect();
    }

    /// Material for primitive index `i`. Falls through to
    /// [`fallback_material`](Self::fallback_material) when `materials`
    /// hasn't been resolved yet (streamed glb still loading) or `i` is
    /// out of bounds.
    pub fn material_for(&self, i: usize) -> &Arc<dyn MeshMaterial> {
        self.materials.get(i).unwrap_or(&self.fallback_material)
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
#[derive(Debug, Clone, Deserialize)]
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
    /// The returned template's
    /// [`fallback_material`](MeshTemplate::fallback_material) is a fresh
    /// `PbrMaterialInstance` bound to `spec.albedo` — used both for the
    /// streaming-fallback unit cube and for any primitive whose
    /// `material_name` doesn't resolve in the application's
    /// [`MaterialRegistry`](super::MaterialRegistry). Per-primitive
    /// materials are populated lazily by
    /// [`MeshTemplate::resolve_materials`] on the frame the real glb
    /// decodes.
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
    ) -> MeshTemplate {
        let mesh_path = VfsPath::new(spec.mesh.clone())
            .unwrap_or_else(|e| panic!("kind {kind_id}: invalid render.mesh path: {e}"));
        let albedo_path = VfsPath::new(spec.albedo.clone())
            .unwrap_or_else(|e| panic!("kind {kind_id}: invalid render.albedo path: {e}"));
        let mesh_handle = asset_server.mesh(mesh_path);
        let albedo_handle = asset_server.texture(albedo_path, TextureColorSpace::Srgb);
        let fallback = self.create_instance(
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
            materials: Vec::new(),
            fallback_material: Arc::new(fallback),
        }
    }

    /// Build an inline [`MeshTemplate`] from a [`PrimitiveMesh`] + flat albedo
    /// factor. Shared helper for procedural ancillary parts (markers, carried
    /// items, gizmos) that don't go through the asset pipeline — they still
    /// plug into the same PBR material surface streamed bodies use, via a 1×1
    /// white texture wrapped in a ready [`Handle`].
    ///
    /// The constructed [`PbrMaterialInstance`](super::PbrMaterialInstance)
    /// is bound for both the single-primitive `materials` slot and the
    /// `fallback_material` — they point at the same `Arc` so the inline
    /// path has no fallback bookkeeping at the draw site.
    pub fn inline_template(
        &self,
        renderer: &Renderer,
        samplers: &SamplerRegistry,
        asset_server: &AssetServer,
        params: InlineTemplate<'_>,
    ) -> MeshTemplate {
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
        let instance = self.create_instance(
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
        let material: Arc<dyn MeshMaterial> = Arc::new(instance);
        MeshTemplate {
            mesh: MeshBacking::Inline {
                primitives: vec![primitive],
            },
            visual_bounds: params.bounds,
            materials: vec![Arc::clone(&material)],
            fallback_material: material,
        }
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
