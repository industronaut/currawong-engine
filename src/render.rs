//! View side of the engine.
//!
//! Reads a [`Simulation`](crate::Simulation) each frame and renders it
//! through wgpu/winit. Compiled only with the default `render` feature; the
//! sim layer never sees anything in this module.
//!
//! ## Module layout
//!
//! - [`camera`] — perspective camera helper.
//! - [`environment`] — view-side environment (sun, ambient, sky) + the
//!   engine-managed scene bind group.
//! - [`renderer`] — window + GPU device/queue/surface + optional depth.
//! - [`view`] — the [`View`] trait + [`EngineCtx`].
//! - [`runner`] — event loop integration ([`run`], [`run_with_clock`]).
//! - [`instance`] — per-key instance bucketing for batched instanced rendering.
//! - [`emitter`] — declarative emitter reconciliation + particle integration.
//! - [`render_object`] — render-object templates + registry (slot schema + parts).
//! - [`render_object_pass`] — engine-driven per-frame walk: sim → declare → cull → fan-out.
//! - [`material`] — material template/instance/per-instance-attribs primitives.
//! - [`pbr`] — metallic-roughness PBR material; reads scene env + camera.
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
#[cfg(feature = "egui")]
mod debug_ui;
mod emitter;
mod environment;
mod instance;
mod material;
mod pbr;
mod render_instances;
mod render_object;
mod render_object_pass;
mod renderer;
mod runner;
mod terrain;
mod terrain_material;
mod terrain_renderer;
mod texture;
mod vertex;
mod view;
mod visibility;

pub use camera::{Camera, CameraBinding, CameraUniformData};
pub use emitter::{EmitterReconciler, EmitterTemplate, Particle, ParticleLifecycle};
pub use environment::{SceneEnvironmentBinding, ViewEnvironment};
pub use instance::{InstanceBuckets, mat4_instance_attributes};
pub use material::{
    MaterialInstanceRegistry, UnlitColoredAttribs, UnlitColoredInstance, UnlitColoredMaterial,
};
pub use pbr::{PbrInstanceAttribs, PbrMaterial, PbrMaterialInstance, PbrMaterialParams};
pub use render_instances::{RenderInstance, RenderInstances};
pub use render_object::{
    EmitterPart, MeshPart, RenderRegistry, RenderTemplate, SlotDescriptor, SlotKind, SlotRouting,
    SlotValue, SlotValues,
};
pub use render_object_pass::{RenderObjectPass, validate_slot_values};
pub use renderer::Renderer;
pub use runner::{run, run_with_clock};
pub use terrain::{ChunkMeshes, FlatTopsMesher, MeshData, TerrainMesher, TerrainVertex};
pub use terrain_material::{TerrainMaterial, TerrainMaterialInstance};
pub use terrain_renderer::TerrainRenderer;
pub use texture::{SamplerKind, SamplerRegistry, Texture};
pub use vertex::PosNormalUv;
pub use view::{EngineCtx, View};
pub use visibility::{Aabb, Frustum};
