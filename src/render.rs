//! View side of the engine.
//!
//! Reads a [`Simulation`](crate::Simulation) each frame and renders it
//! through wgpu/winit. Compiled only with the default `render` feature; the
//! sim layer never sees anything in this module.
//!
//! ## Module layout
//!
//! - [`camera`] — perspective camera helper.
//! - [`camera_rig`] — input-driven controllers that drive a [`Camera`]
//!   (currently [`OrbitRig`] for strategy-game-style cameras).
//! - [`cell_highlight`] — single-cell outline overlay paired with
//!   [`TerrainPicker`] for hover/selection feedback.
//! - [`environment`] — view-side environment (sun, ambient, sky) + the
//!   engine-managed scene bind group.
//! - [`frame_stats`] — per-frame CPU counters (draws, instances, proxies)
//!   surfaced on [`EngineCtx`] for debug overlays.
//! - [`frame_timings`] — per-frame CPU + GPU timing breakdown surfaced on
//!   [`EngineCtx`] for debug overlays.
//! - [`gpu_profiler`] — wgpu timestamp-query ring driving
//!   [`FrameTimings::gpu`].
//! - [`renderer`] — window + GPU device/queue/surface + scene resources.
//! - [`scene_resources`] — engine-managed per-scene GPU state: depth
//!   attachment, scene-environment binding (and future shadow maps, IBL probes,
//!   MSAA resolve targets, …).
//! - [`screenshot`] — engine-driven F12 capture; copies the swapchain image
//!   into a staging buffer and writes a PNG.
//! - [`view`] — the [`View`] trait + [`EngineCtx`].
//! - [`runner`] — event loop integration ([`run`], [`run_with_clock`]).
//! - [`instance`] — per-key instance bucketing for batched instanced rendering.
//! - [`emitter`] — declarative emitter reconciliation + particle integration.
//! - [`render_object`] — render-object templates + registry (mesh + emitter parts).
//! - [`render_object_traversal`] — engine-driven per-frame walk: sim → declare → cull → fan-out.
//! - [`line_material`] — unlit `LineList`-topology material for debug
//!   gizmos (bounding boxes, axis overlays). Parallel in shape to
//!   [`material::UnlitColoredMaterial`]. Fixed 1 px width; cheaper than
//!   [`fat_line_material`] for dense overlays where stylization isn't needed.
//! - [`fat_line_material`] — quad-expanded thick lines with screen-space
//!   pixel width. The right tool when line width matters visually
//!   (bounding boxes at viewing distance, gameplay overlays). Slightly
//!   more expensive per segment than [`line_material`].
//! - [`material`] — material template/instance/per-instance-attribs primitives.
//! - [`material_registry`] — name-keyed [`MaterialRegistry`] resolving glb material
//!   slot names to [`PbrMaterialInstance`]s (or any other instance type).
//! - [`handle`] — streaming-asset reference type (`Handle<T>` + states).
//! - [`asset_server`] — view-side asset gateway; spawns background loads,
//!   serves the magenta fallback, carries the debug "force loading" toggle.
//! - [`mesh`] — streamable static-mesh asset (`Mesh`) + glTF 2.0 loader.
//! - [`mesh_primitives`] — CPU-side mesh generators (cube, plane, sphere,
//!   cylinder, cone) in the canonical [`vertex`] layout.
//! - [`mesh_template`] — bundled per-part GPU resources (mesh buffers +
//!   visual bounds + material instance) + [`RenderSpec`] RON projection.
//! - [`pbr`] — metallic-roughness PBR material; reads scene env + camera.
//! - [`pbr_atlas`] — stylized PBR material that reads albedo + MRE from two
//!   atlases; resolves through [`MaterialRegistry`] by glb material name.
//! - [`picking`] — screen-space cursor → world ray + tile-grid hover picker.
//! - [`picking_buffer`] — GPU hit-ID buffer plumbing: per-frame indirection
//!   table + readback ring for sloped-terrain and mesh-object picking.
//! - [`terrain`] — view-side meshing of tile-grid terrain into chunk meshes.
//! - [`terrain_material`] — opaque + transparent terrain material pipelines.
//! - [`terrain_renderer`] — per-chunk GPU buffer cache + draw routine.
//! - [`texture`] — `Texture` asset (RGBA8 + CPU mip generation) + canonical samplers.
//! - [`vertex`] — closed set of canonical per-vertex layouts.
//! - [`visibility`] — AABB + view-frustum culling primitives.
//! - [`render_proxy`] — live render-object instance reconciler with cull hysteresis.
//! - [`yakui_assets`] — VFS → yakui [`ManagedTextureId`](yakui::ManagedTextureId)
//!   cache for game UI (behind the `yakui` feature).
//!
//! Submodules are private; their public types are re-exported here so callers
//! see a flat `currawong::*` surface.

mod asset_server;
mod camera;
mod camera_rig;
mod cell_highlight;
#[cfg(feature = "egui")]
mod debug_ui;
mod emitter;
mod environment;
mod fat_line_material;
mod frame_stats;
mod frame_timings;
#[cfg(feature = "yakui")]
mod game_ui;
// Force-disabled in `Renderer::new` because pass-level + encoder-level
// timestamp writes trip Metal API Validation
// (`METAL_DEVICE_WRAPPER_TYPE=1`) into aborting every submit. Code parked
// here so the wiring can be re-enabled once the Metal validation issue
// is understood; in the meantime `dead_code` is expected.
#[allow(dead_code)]
mod gpu_profiler;
mod handle;
mod instance;
mod line_material;
mod material;
mod material_registry;
mod mesh;
mod mesh_draw;
mod mesh_primitives;
mod mesh_template;
mod pbr;
mod pbr_atlas;
mod picking;
mod picking_buffer;
mod render_object;
mod render_object_traversal;
mod render_proxy;
mod renderer;
mod runner;
mod scene_resources;
mod screenshot;
mod shadow;
mod terrain;
mod terrain_material;
mod terrain_renderer;
mod texture;
mod vertex;
mod view;
mod visibility;
#[cfg(feature = "yakui")]
mod yakui_assets;

pub use asset_server::{AssetServer, MeshSource, ResolvedMesh, ResolvedTexture, TextureSource};
pub use camera::{Camera, CameraBinding, CameraUniformData};
pub use camera_rig::{OrbitRig, OrbitRigConfig};
pub use cell_highlight::CellHighlight;
pub use emitter::{EmitterReconciler, EmitterTemplate, Particle, ParticleLifecycle};
pub use environment::{SceneEnvironmentBinding, SunCascades, ViewEnvironment};
pub use fat_line_material::{
    FatLineMaterial, FatLineMaterialInstance, FatLineMaterialParams, FatLineVertex,
    unit_cube_fat_line_geometry,
};
pub use frame_stats::FrameStats;
pub use frame_timings::{FrameTimings, GpuSegments};
pub use handle::{Handle, HandleError, HandleState};
pub use instance::{InstanceBuckets, mat4_instance_attributes, u32_id_instance_attribute};
pub use line_material::{LineMaterial, LineMaterialInstance, unit_cube_line_geometry};
pub use material::{
    MaterialInstanceRegistry, MeshInstanceAttribs, MeshMaterial, UnlitColoredInstance,
    UnlitColoredMaterial,
};
pub use material_registry::{MaterialId, MaterialIdError, MaterialRegistry};
pub use mesh::{
    DecodedMesh, DecodedPrimitive, Mesh, MeshLoadError, MeshPrimitive, decode_gltf_mesh,
};
pub use mesh_draw::{MeshDraw, PbrAtlasMaterials};
pub use mesh_primitives::PrimitiveMesh;
pub use mesh_template::{
    InlineTemplate, MeshBacking, MeshNodeSpec, MeshTemplate, NodeSpec, RenderSpec, TransformSpec,
    build_hierarchical_render_template, build_streamed_pbr_mesh_template, node_kind,
};
pub use pbr::{PbrMaterial, PbrMaterialInstance, PbrMaterialParams};
pub use pbr_atlas::{PbrAtlasMaterial, PbrAtlasMaterialInstance, PbrAtlasMaterialParams};
pub use picking::{Hover, Ray, TerrainPicker};
pub use picking_buffer::{FrameIdTable, HitTarget};
pub use render_object::{
    EmitterPart, MeshPart, NodeId, NodeKind, RenderRegistry, RenderTemplate, TemplateNode,
};
pub use render_object_traversal::RenderObjectTraversal;
pub use render_proxy::{NodeState, RenderProxies, RenderProxy};
pub use renderer::Renderer;
pub use runner::{run, run_with_clock};
pub use shadow::ShadowMeshPipeline;
pub use terrain::{
    ChunkMeshes, FlatTopsMesher, MeshData, SlopeMesher, TerrainMesher, TerrainVertex,
};
pub use terrain_material::{TerrainMaterial, TerrainMaterialInstance};
pub use terrain_renderer::TerrainRenderer;
pub use texture::{SamplerKind, SamplerRegistry, Texture, TextureColorSpace, TextureLoadError};
pub use vertex::PosNormalUv;
pub use view::{EngineCtx, View, ViewConfig};
pub use visibility::{Aabb, Frustum};
#[cfg(feature = "yakui")]
pub use yakui_assets::{YakuiAssetError, YakuiAssets};
