//! Helpers that turn a frame's worth of [`InstanceBuckets`] into draws.
//!
//! Views built on the kind→[`MeshTemplate`] convention end up writing the same
//! two loops:
//!
//! 1. **PBR / atlas dispatch.** Walk filled buckets, look up each primitive's
//!    `material_name` in a [`MaterialRegistry<PbrAtlasMaterialInstance>`], and
//!    switch between [`PbrMaterial`] and [`PbrAtlasMaterial`] on the fly. The
//!    active pipeline is tracked across primitives so `set_pipeline` only fires
//!    on transitions.
//! 2. **Depth-only.** The same bucket walk under [`ShadowMeshPipeline`] for
//!    cascade fills — no atlas dispatch, no per-primitive material switch.
//!
//! Both are mechanical glue with no per-example variation; bundle them here so
//! a third consumer doesn't copy them again. The caller still binds bind
//! groups 0 (camera) + 1 (scene environment) — these helpers only touch the
//! pipeline, bind group 2 (material instance), and the vertex / index buffer
//! slots.

use std::collections::HashMap;
use std::hash::Hash;

use super::asset_server::AssetServer;
use super::instance::InstanceBuckets;
use super::material::MeshInstanceAttribs;
use super::material_registry::MaterialRegistry;
use super::mesh_template::MeshTemplate;
use super::pbr::{PbrMaterial, PbrMaterialInstance};
use super::pbr_atlas::{PbrAtlasMaterial, PbrAtlasMaterialInstance};
use super::renderer::Renderer;
use super::shadow::ShadowMeshPipeline;

/// Which pipeline is currently bound across the per-primitive draw loop in
/// [`MeshDraw::pbr_with_atlas`]. `None` forces a bind on the first draw of
/// the frame; the two real variants flip on transition.
#[derive(PartialEq, Eq)]
enum ActivePipeline {
    None,
    Pbr,
    Atlas,
}

/// Borrowed bundle of the three things `MeshDraw::pbr_with_atlas` dispatches
/// over: the streamed-PBR pipeline, the stylized atlas pipeline, and the
/// registry of atlas instances keyed by glb material slot name. Built fresh at
/// each call site — the three references travel together every time so a
/// struct is cheaper than four extra positional arguments.
pub struct PbrAtlasMaterials<'a> {
    pub pbr: &'a PbrMaterial,
    pub atlas: &'a PbrAtlasMaterial,
    pub atlas_instances: &'a MaterialRegistry<PbrAtlasMaterialInstance>,
}

/// Stateless helper for the two canonical bucket draw loops.
pub struct MeshDraw;

impl MeshDraw {
    /// Issue one indexed-instanced draw per primitive across every filled
    /// bucket, dispatching each primitive between the PBR pipeline and the
    /// stylized atlas pipeline by its `material_name`.
    ///
    /// The lookup is best-effort: a primitive whose `material_name` resolves
    /// through `materials.atlas_instances` draws under the atlas pipeline with
    /// that instance's bind group; everything else (including primitives with
    /// no `material_name`) falls back to the template's own
    /// [`PbrMaterialInstance`]. Same fallback shape `MaterialRegistry`
    /// already exposes — registering nothing is equivalent to "all PBR."
    ///
    /// The caller binds bind groups 0 (camera) + 1 (scene env) before
    /// calling; this helper only touches the pipeline, bind group 2, and
    /// vertex/index buffer slots.
    pub fn pbr_with_atlas<K>(
        pass: &mut wgpu::RenderPass<'_>,
        renderer: &Renderer,
        asset_server: &AssetServer,
        materials: PbrAtlasMaterials<'_>,
        mesh_templates: &HashMap<K, MeshTemplate<PbrMaterialInstance>>,
        buckets: &InstanceBuckets<K, MeshInstanceAttribs>,
    ) where
        K: Clone + Eq + Hash,
    {
        let mut active = ActivePipeline::None;
        for (part_key, instance_buffer, count) in buckets.iter_filled() {
            let Some(template) = mesh_templates.get(part_key) else {
                continue;
            };
            let resolved = template.resolve(asset_server);
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            for prim in resolved.primitives {
                let atlas_instance = prim
                    .material_name
                    .as_deref()
                    .and_then(|name| materials.atlas_instances.get_by_name(name));
                match atlas_instance {
                    Some(instance) => {
                        if active != ActivePipeline::Atlas {
                            pass.set_pipeline(materials.atlas.pipeline());
                            active = ActivePipeline::Atlas;
                        }
                        pass.set_bind_group(2, instance.bind_group(), &[]);
                    }
                    None => {
                        if active != ActivePipeline::Pbr {
                            pass.set_pipeline(materials.pbr.pipeline());
                            active = ActivePipeline::Pbr;
                        }
                        pass.set_bind_group(2, template.material.bind_group(), &[]);
                    }
                }
                pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
                pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, 0..count);
                renderer.record_draw(count);
            }
        }
    }

    /// Issue depth-only draws for every filled bucket under
    /// [`ShadowMeshPipeline`]. Used inside [`View::shadow_pass`](super::View::shadow_pass)
    /// to fill cascade depth attachments — the depth pipeline ignores normals,
    /// UVs, tints, and per-primitive material slots, so no material lookup is
    /// needed and any [`MeshTemplate`] backing works.
    ///
    /// Generic over the template's material type so a depth pass can run over
    /// templates from any material family — the depth pipeline only reads
    /// position + per-instance model matrix.
    pub fn depth_only<K, M>(
        pass: &mut wgpu::RenderPass<'_>,
        renderer: &Renderer,
        asset_server: &AssetServer,
        shadow: &ShadowMeshPipeline,
        mesh_templates: &HashMap<K, MeshTemplate<M>>,
        buckets: &InstanceBuckets<K, MeshInstanceAttribs>,
    ) where
        K: Clone + Eq + Hash,
    {
        pass.set_pipeline(shadow.pipeline());
        for (part_key, instance_buffer, count) in buckets.iter_filled() {
            let Some(template) = mesh_templates.get(part_key) else {
                continue;
            };
            let resolved = template.resolve(asset_server);
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            for prim in resolved.primitives {
                pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
                pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, 0..count);
                renderer.record_draw(count);
            }
        }
    }
}
