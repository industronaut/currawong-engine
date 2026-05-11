//! Optional egui integration for debug overlays. Behind the `egui` feature.
//!
//! Wraps the three egui pieces — `egui::Context` (UI state), `egui_winit::State`
//! (winit→egui input translation), and `egui_wgpu::Renderer` (tessellated
//! geometry + texture upload + draw) — so the engine can drive a `View::ui`
//! callback once per frame and render the result on top of the main scene.

use winit::event::WindowEvent;
use winit::window::Window;

use super::renderer::Renderer;

/// Engine-owned egui plumbing. Constructed alongside the user's `View` when
/// the `egui` feature is on; not exposed to user code directly.
pub(super) struct DebugUi {
    ctx: egui::Context,
    winit_state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl DebugUi {
    pub(super) fn new(renderer: &Renderer) -> Self {
        let ctx = egui::Context::default();
        let winit_state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            renderer.window.as_ref(),
            Some(renderer.window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &renderer.device,
            renderer.surface_format(),
            egui_wgpu::RendererOptions::default(),
        );
        Self {
            ctx,
            winit_state,
            renderer: egui_renderer,
        }
    }

    /// Feed a winit event to egui. `consumed` means egui handled it — the
    /// engine should not dispatch it to `View::input`.
    pub(super) fn on_window_event(
        &mut self,
        window: &Window,
        event: &WindowEvent,
    ) -> egui_winit::EventResponse {
        self.winit_state.on_window_event(window, event)
    }

    /// Run the UI closure, upload texture deltas and tessellated geometry, and
    /// draw the overlay into `view_tex` on top of whatever the main pass left
    /// behind. Returns any staging command buffers egui produced — the caller
    /// must submit them before the frame's main encoder.
    pub(super) fn run_and_render(
        &mut self,
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        view_tex: &wgpu::TextureView,
        run_ui: impl FnOnce(&egui::Context),
    ) -> Vec<wgpu::CommandBuffer> {
        let window = renderer.window.as_ref();
        let raw_input = self.winit_state.take_egui_input(window);
        self.ctx.begin_pass(raw_input);
        run_ui(&self.ctx);
        let full_output = self.ctx.end_pass();
        self.winit_state
            .handle_platform_output(window, full_output.platform_output);

        let pixels_per_point = self.ctx.pixels_per_point();
        let paint_jobs = self.ctx.tessellate(full_output.shapes, pixels_per_point);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(&renderer.device, &renderer.queue, *id, image_delta);
        }

        let (width, height) = renderer.surface_size();
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point,
        };
        let staging = self.renderer.update_buffers(
            &renderer.device,
            &renderer.queue,
            encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        {
            // egui-wgpu::Renderer::render requires RenderPass<'static>.
            // forget_lifetime hands ownership of the borrow over to runtime
            // checks; the pass is dropped at the end of this block before the
            // encoder is used again.
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("currawong egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: view_tex,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                })
                .forget_lifetime();
            self.renderer
                .render(&mut pass, &paint_jobs, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }

        staging
    }
}
