//! View side of the engine.
//!
//! Reads a [`Simulation`](crate::Simulation) each frame and renders it
//! through wgpu/winit. Compiled only with the default `render` feature; the
//! sim layer never sees anything in this module.
//!
//! ## Module layout
//!
//! - [`camera`] — perspective camera helper.
//! - [`renderer`] — window + GPU device/queue/surface + optional depth.
//! - [`view`] — the [`View`] trait + [`EngineCtx`].
//! - [`runner`] — event loop integration ([`run`], [`run_with_clock`]).
//! - [`instance`] — per-key instance bucketing for batched instanced rendering.
//!
//! Submodules are private; their public types are re-exported here so callers
//! see a flat `currawong::*` surface.

mod camera;
mod instance;
mod renderer;
mod runner;
mod view;

pub use camera::Camera;
pub use instance::InstanceBuckets;
pub use renderer::Renderer;
pub use runner::{run, run_with_clock};
pub use view::{EngineCtx, View};
