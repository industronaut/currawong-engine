//! The `View` trait: a renderer reads simulation state through one of these.

use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;

use crate::sim::{SimClock, Simulation};

use super::renderer::Renderer;

/// Mutable engine state passed to view callbacks. Use `event_loop` to
/// request exit and `clock` to read or change sim speed and tick rate.
pub struct EngineCtx<'a> {
    pub event_loop: &'a ActiveEventLoop,
    pub clock: &'a mut SimClock,
}

/// A view onto a [`Simulation`].
///
/// `render` receives `&Sim` (read-only), so the rendering path is structurally
/// prevented from mutating the simulation. `input` receives `&mut Sim` so
/// user-driven actions (clicks, key presses) can drive sim changes.
///
/// `init` runs once after the GPU is ready (build pipelines, load assets).
pub trait View: 'static {
    /// The kind of simulation this view reads from.
    type Sim: Simulation;

    fn init(renderer: &Renderer) -> Self;

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

    fn title() -> &'static str {
        "currawong"
    }

    fn clear_colour() -> wgpu::Color {
        wgpu::Color {
            r: 0.05,
            g: 0.07,
            b: 0.10,
            a: 1.0,
        }
    }

    /// Return `Some(format)` to have the engine allocate a depth texture and
    /// pre-attach it to the frame's render pass. Pipelines must declare the
    /// same format in their `DepthStencilState`. Default `None` is right for
    /// 2D / UI views that draw in clip space.
    fn depth_format() -> Option<wgpu::TextureFormat> {
        None
    }
}
