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
//! - [`renderer`] — window + GPU device/queue/surface + scene resources.
//! - [`scene_resources`] — engine-managed per-scene GPU state: depth
//!   attachment, scene-environment binding (and future shadow maps, IBL probes,
//!   MSAA resolve targets, …).
//! - [`view`] — the [`View`] trait + [`EngineCtx`].
//! - [`runner`] — event loop integration ([`run`], [`run_with_clock`]).
//! - [`instance`] — per-key instance bucketing for batched instanced rendering.
//! - [`emitter`] — declarative emitter reconciliation + particle integration.
//! - [`render_object`] — render-object templates + registry (slot schema + parts).
//! - [`render_object_pass`] — engine-driven per-frame walk: sim → declare → cull → fan-out.
//! - [`material`] — material template/instance/per-instance-attribs primitives.
//! - [`mesh_primitives`] — CPU-side mesh generators (cube, plane, sphere,
//!   cylinder, cone) in the canonical [`vertex`] layout.
//! - [`pbr`] — metallic-roughness PBR material; reads scene env + camera.
//! - [`picking`] — screen-space cursor → world ray + tile-grid hover picker.
//! - [`picking_buffer`] — GPU hit-ID buffer plumbing: per-frame indirection
//!   table + readback ring for sloped-terrain and mesh-object picking.
//! - [`terrain`] — view-side meshing of tile-grid terrain into chunk meshes.
//! - [`terrain_material`] — opaque + transparent terrain material pipelines.
//! - [`terrain_renderer`] — per-chunk GPU buffer cache + draw routine.
//! - [`texture`] — `Texture` asset (RGBA8 + CPU mip generation) + canonical samplers.
//! - [`vertex`] — closed set of canonical per-vertex layouts.
//! - [`visibility`] — AABB + view-frustum culling primitives.
//! - [`render_instances`] — live render-object instance reconciler with cull hysteresis.
//!
//! Submodules are private; their public types are re-exported here so callers
//! see a flat `currawong::*` surface.

mod camera;
mod camera_rig;
mod cell_highlight;
#[cfg(feature = "egui")]
mod debug_ui;
mod emitter;
mod environment;
#[cfg(feature = "yakui")]
mod game_ui;
mod instance;
mod live_render_objects;
mod material;
mod mesh_primitives;
mod pbr;
mod picking;
mod picking_buffer;
mod render_object;
mod render_object_pass;
mod renderer;
mod runner;
mod scene_resources;
mod terrain;
mod terrain_material;
mod terrain_renderer;
mod texture;
mod vertex;
mod view;
mod visibility;

pub use camera::{Camera, CameraBinding, CameraUniformData};
pub use camera_rig::{OrbitRig, OrbitRigConfig};
pub use cell_highlight::CellHighlight;
pub use emitter::{EmitterReconciler, EmitterTemplate, Particle, ParticleLifecycle};
pub use environment::{SceneEnvironmentBinding, ViewEnvironment};
pub use instance::{InstanceBuckets, mat4_instance_attributes, u32_id_instance_attribute};
pub use live_render_objects::{LiveRenderObject, LiveRenderObjects, RenderPartState};
pub use material::{
    MaterialInstanceRegistry, MeshInstanceAttribs, MeshMaterial, UnlitColoredInstance,
    UnlitColoredMaterial,
};
pub use mesh_primitives::PrimitiveMesh;
pub use pbr::{PbrMaterial, PbrMaterialInstance, PbrMaterialParams};
pub use picking::{Hover, Ray, TerrainPicker};
pub use picking_buffer::{FrameIdTable, HitTarget};
pub use render_object::{
    EmitterPart, MeshPart, RenderRegistry, RenderTemplate, SlotDescriptor, SlotKind, SlotRouting,
    SlotValue, SlotValues,
};
pub use render_object_pass::{RenderObjectPass, validate_slot_values};
pub use renderer::Renderer;
pub use runner::{run, run_with_clock};
pub use terrain::{
    ChunkMeshes, FlatTopsMesher, MeshData, SlopeMesher, TerrainMesher, TerrainVertex,
};
pub use terrain_material::{TerrainMaterial, TerrainMaterialInstance};
pub use terrain_renderer::TerrainRenderer;
pub use texture::{SamplerKind, SamplerRegistry, Texture, TextureColorSpace, TextureLoadError};
pub use vertex::PosNormalUv;
pub use view::{EngineCtx, View, ViewConfig};
pub use visibility::{Aabb, Frustum};
