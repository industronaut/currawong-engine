//! The `View` trait: a renderer reads simulation state through one of these.

use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;

use crate::sim::{SimClock, Simulation, ZoneId};

use super::environment::ViewEnvironment;
use super::renderer::Renderer;

/// Mutable engine state passed to view callbacks. Use `event_loop` to
/// request exit and `clock` to read or change sim speed and tick rate.
pub struct EngineCtx<'a> {
    pub event_loop: &'a ActiveEventLoop,
    pub clock: &'a mut SimClock,
}

/// Static window + render-target configuration for a [`View`]. Returned
/// from [`View::init`] alongside the view itself and read once by the
/// engine at startup.
///
/// Construct with [`Default`] and adjust the fields you care about:
///
/// ```ignore
/// ViewConfig {
///     title: "my game",
///     depth_format: Some(wgpu::TextureFormat::Depth32Float),
///     ..Default::default()
/// }
/// ```
///
/// New static knobs (MSAA samples, present mode, …) land here. New
/// per-frame hooks land on [`View`].
pub struct ViewConfig {
    /// Window title.
    pub title: &'static str,
    /// Colour the engine clears the swapchain to at the start of each frame.
    pub clear_colour: wgpu::Color,
    /// Set to `Some(format)` to have the engine allocate a depth texture
    /// and pre-attach it to the frame's render pass. Pipelines must declare
    /// the same format in their `DepthStencilState`. `None` is right for
    /// 2D / UI views that draw in clip space.
    pub depth_format: Option<wgpu::TextureFormat>,
}

impl Default for ViewConfig {
    fn default() -> Self {
        Self {
            title: "currawong",
            clear_colour: wgpu::Color {
                r: 0.05,
                g: 0.07,
                b: 0.10,
                a: 1.0,
            },
            depth_format: None,
        }
    }
}

/// A view onto a [`Simulation`].
///
/// `render` receives `&Sim` (read-only), so the rendering path is structurally
/// prevented from mutating the simulation. `input` receives `&mut Sim` so
/// user-driven actions (clicks, key presses) can drive sim changes.
///
/// `init` runs once after the GPU is ready (build pipelines, load assets)
/// and returns the view plus a [`ViewConfig`] carrying the static window
/// and render-target settings.
pub trait View: 'static {
    /// The kind of simulation this view reads from.
    type Sim: Simulation;

    /// Build the view and return its static config. Called once after the
    /// GPU is ready; the config is read by the engine to set the window
    /// title, clear colour, and depth attachment.
    fn init(renderer: &Renderer) -> (Self, ViewConfig)
    where
        Self: Sized;

    /// Render a frame.
    ///
    /// `alpha` is the interpolation factor in `[0, 1]` between the most recent
    /// completed tick and the next pending tick — useful for smooth animation
    /// when tick rate is below refresh rate. Views that don't interpolate can
    /// ignore it.
    fn render(
        &mut self,
        sim: &Self::Sim,
        alpha: f32,
        renderer: &Renderer,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        let _ = (sim, alpha, renderer, pass);
    }

    fn input(&mut self, sim: &mut Self::Sim, ctx: &mut EngineCtx, event: &WindowEvent) {
        let _ = (sim, ctx, event);
    }

    /// Build the per-frame debug UI. Called once per frame after `render`,
    /// with the engine's `egui::Context`. Mirrors `input`'s mutability:
    /// widgets can read sim state and drive sim/engine changes (pause,
    /// speed change, exit) via `ctx`.
    ///
    /// Default no-op; opt in by overriding. Behind the `egui` feature.
    #[cfg(feature = "egui")]
    fn ui(&mut self, sim: &mut Self::Sim, ctx: &mut EngineCtx, egui_ctx: &egui::Context) {
        let _ = (sim, ctx, egui_ctx);
    }

    /// Build the per-frame game UI with yakui. Called once per frame after
    /// `render`, between `Yakui::start` and `Yakui::finish` so widget calls
    /// (`yakui::widgets::*`, `yakui::label`, `yakui::button`, …) attach to the
    /// engine's `Yakui` state via yakui's thread-local context.
    ///
    /// Mirrors `input`'s mutability: widgets can read sim state and drive
    /// sim/engine changes via `ctx`.
    ///
    /// Default no-op; opt in by overriding. Behind the `yakui` feature.
    /// Independent of [`ui`](Self::ui) — both can be implemented when both
    /// `egui` and `yakui` features are enabled.
    #[cfg(feature = "yakui")]
    fn game_ui(&mut self, sim: &mut Self::Sim, ctx: &mut EngineCtx) {
        let _ = (sim, ctx);
    }

    /// Which zone the camera is currently looking at, if any. The engine
    /// uses this to drive [`extract_environment`](Self::extract_environment)
    /// each frame; later it'll also gate visibility culling and terrain
    /// streaming. Default `None` is right for UI/2D views that have no
    /// notion of an active zone.
    ///
    /// `sim` is provided so the implementation can derive the active zone
    /// from world state if it isn't a constant — e.g. "the zone holding
    /// my player object." Conventionally the View holds a
    /// [`Camera`](crate::Camera) and forwards `self.camera.zone`.
    fn active_zone(&self, sim: &Self::Sim) -> Option<ZoneId> {
        let _ = sim;
        None
    }

    /// Build the per-frame [`ViewEnvironment`] for the currently-active zone.
    /// Called by the engine each frame *before* `render`, when
    /// [`active_zone`](Self::active_zone) returns `Some`. The result is
    /// packed into the engine-managed scene bind group
    /// ([`Renderer::scene_bind_group`](crate::Renderer::scene_bind_group))
    /// so any pipeline that declares
    /// [`Renderer::scene_layout`](crate::Renderer::scene_layout) reads it
    /// automatically.
    ///
    /// Default returns [`ViewEnvironment::neutral`] — full ambient, no sun.
    /// Override to drive lighting from `SimEnvironment` (or anything else).
    fn extract_environment(&self, sim: &Self::Sim, zone: ZoneId) -> ViewEnvironment {
        let _ = (sim, zone);
        ViewEnvironment::neutral()
    }
}
