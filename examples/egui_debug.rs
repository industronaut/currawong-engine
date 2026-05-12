//! Minimal egui overlay: FPS counter (wall-clock, ticks even when sim is
//! paused) plus a small sim-control panel.
//!
//! Run with: `cargo run --example egui_debug --features egui`

use std::collections::VecDeque;
use std::time::Instant;

use currawong::{EngineCtx, Renderer, View, ViewConfig, egui, winit};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

struct Debug {
    last_frame: Instant,
    frame_samples: VecDeque<f32>,
}

impl View for Debug {
    type Sim = ();

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — egui_debug",
        ..ViewConfig::DEFAULT
    };

    fn init(_: &Renderer) -> Self {
        Self {
            last_frame: Instant::now(),
            frame_samples: VecDeque::with_capacity(120),
        }
    }

    fn input(&mut self, _: &mut (), ctx: &mut EngineCtx, event: &WindowEvent) {
        if let WindowEvent::KeyboardInput { event, .. } = event
            && event.state == ElementState::Pressed
            && let PhysicalKey::Code(KeyCode::Escape) = event.physical_key
        {
            ctx.event_loop.exit();
        }
    }

    fn ui(&mut self, _: &mut (), ctx: &mut EngineCtx, egui_ctx: &egui::Context) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        if self.frame_samples.len() == 120 {
            self.frame_samples.pop_front();
        }
        self.frame_samples.push_back(dt);
        let avg_dt = self.frame_samples.iter().sum::<f32>() / self.frame_samples.len() as f32;
        let fps = if avg_dt > 0.0 { 1.0 / avg_dt } else { 0.0 };

        egui::Window::new("debug")
            .default_pos([12.0, 12.0])
            .resizable(false)
            .show(egui_ctx, |ui| {
                ui.label(format!("fps: {fps:5.1}  ({:.2} ms)", avg_dt * 1000.0));
                ui.separator();
                ui.label(format!("sim ticks: {}", ctx.clock.total_ticks()));
                ui.label(format!(
                    "sim time: {:6.2} s",
                    ctx.clock.sim_time().as_secs_f32()
                ));
                ui.separator();
                let mut speed = ctx.clock.speed();
                ui.horizontal(|ui| {
                    let label = if ctx.clock.is_paused() {
                        "play"
                    } else {
                        "pause"
                    };
                    if ui.button(label).clicked() {
                        let new = if ctx.clock.is_paused() { 1.0 } else { 0.0 };
                        ctx.clock.set_speed(new);
                        speed = new;
                    }
                    ui.add(egui::Slider::new(&mut speed, 0.0..=4.0).text("speed"));
                });
                if (speed - ctx.clock.speed()).abs() > f32::EPSILON {
                    ctx.clock.set_speed(speed);
                }
                if ui.button("quit").clicked() {
                    ctx.event_loop.exit();
                }
            });
    }
}

fn main() {
    currawong::run::<Debug>(());
}
