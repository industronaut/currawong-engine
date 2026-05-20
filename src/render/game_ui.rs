//! Optional yakui integration for game UI. Behind the `yakui` feature.
//!
//! Wraps the three yakui pieces — `yakui::Yakui` (UI state), `yakui_winit::YakuiWinit`
//! (winit→yakui input translation), and `yakui_wgpu::YakuiWgpu` (tessellated
//! geometry + texture upload + draw) — so the engine can drive a `View::game_ui`
//! callback once per frame and render the result on top of the main scene.
//!
//! Independent of and parallel to [`super::debug_ui`]: yakui is for shipped
//! game UI (HUDs, menus, panels); egui is for engine/debug overlays. Both can
//! be enabled together; in that case yakui paints first (game UI sits between
//! the world and the debug overlay) and egui consumes events first
//! (debug overlay is visually on top).

use winit::event::WindowEvent;

use super::renderer::Renderer;

/// Engine-owned yakui plumbing. Constructed alongside the user's `View` when
/// the `yakui` feature is on; not exposed to user code directly.
pub(super) struct GameUi {
    state: yakui::Yakui,
    winit_state: yakui_winit::YakuiWinit,
    renderer: yakui_wgpu::YakuiWgpu,
    surface_format: wgpu::TextureFormat,
}

impl GameUi {
    pub(super) fn new(renderer: &Renderer) -> Self {
        let state = yakui::Yakui::new();
        let winit_state = yakui_winit::YakuiWinit::new(renderer.window.as_ref());
        let yakui_renderer = yakui_wgpu::YakuiWgpu::new(&renderer.device, &renderer.queue);
        Self {
            state,
            winit_state,
            renderer: yakui_renderer,
            surface_format: renderer.surface_format(),
        }
    }

    /// Feed a winit event to yakui. `true` means yakui handled it — the
    /// engine should not dispatch it to `View::input`.
    pub(super) fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        self.winit_state.handle_window_event(&mut self.state, event)
    }

    /// Run the UI closure, then record yakui's draw calls into `encoder`,
    /// compositing on top of whatever the main pass left in `view_tex`.
    ///
    /// The closure receives `&mut yakui::Yakui` so callers can reach
    /// state-level APIs that the thread-local widget surface doesn't cover —
    /// chiefly [`Yakui::add_texture`](yakui::Yakui::add_texture) for
    /// registering yakui-managed images (consumed via
    /// [`crate::YakuiAssets`]).
    pub(super) fn run_and_render(
        &mut self,
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        view_tex: &wgpu::TextureView,
        run_ui: impl FnOnce(&mut yakui::Yakui),
    ) {
        self.state.start();
        run_ui(&mut self.state);
        self.state.finish();

        self.renderer.paint_with_encoder(
            &mut self.state,
            &renderer.device,
            &renderer.queue,
            encoder,
            yakui_wgpu::SurfaceInfo {
                format: self.surface_format,
                sample_count: 1,
                color_attachment: view_tex,
                resolve_target: None,
            },
        );
    }
}
