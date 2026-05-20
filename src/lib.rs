//! Currawong — a game engine in Rust.
//!
//! ## Architecture
//!
//! The engine separates the simulation from the view:
//!
//! - [`Simulation`] owns the world model and ticks regardless of whether
//!   anything is being rendered. It does not depend on `wgpu` or `winit`.
//! - [`View`] is a separate concern that reads simulation state to draw
//!   frames and route input.
//! - [`data`] is the third top-level module: the virtual filesystem and
//!   definitions pipeline. Sim depends on [`data::Definitions`]; the view
//!   pulls assets through the same VFS. Always compiled, no
//!   `wgpu`/`winit` dependencies.
//!
//! Headless tick — running the simulation without a window — is a
//! first-class mode. The render-side machinery ([`Renderer`], [`View`],
//! [`run`]) is enabled by the default `render` feature; building with
//! `--no-default-features` excludes `wgpu`/`winit` entirely and yields a
//! sim-only crate.

pub use glam;

mod sim;
pub use sim::*;

pub mod data;

pub mod grid;
pub use grid::{Grid, HexGrid, SquareGrid};

#[cfg(feature = "render")]
pub use pollster;
#[cfg(feature = "render")]
pub use wgpu;
#[cfg(feature = "render")]
pub use winit;

#[cfg(feature = "egui")]
pub use egui;

#[cfg(feature = "yakui")]
pub use yakui;

#[cfg(feature = "render")]
mod render;
#[cfg(feature = "render")]
pub use render::*;
