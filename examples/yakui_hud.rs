//! Minimal yakui game-UI overlay: a panel with sim controls (pause/speed/quit)
//! and a live FPS readout, painted on a clear background. Demonstrates that
//! yakui widgets can drive sim and engine state via `EngineCtx`, and that the
//! per-event integration consumes clicks before they reach `View::input`.
//!
//! Run with: `cargo run --example yakui_hud --features yakui`

use std::collections::VecDeque;
use std::time::Instant;

use currawong::{EngineCtx, Renderer, View, ViewConfig, winit, yakui};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

struct Hud {
    last_frame: Instant,
    frame_samples: VecDeque<f32>,
}

impl View for Hud {
    type Sim = ();

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — yakui_hud",
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

    fn game_ui(&mut self, _: &mut (), ctx: &mut EngineCtx) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        if self.frame_samples.len() == 120 {
            self.frame_samples.pop_front();
        }
        self.frame_samples.push_back(dt);
        let avg_dt = self.frame_samples.iter().sum::<f32>() / self.frame_samples.len() as f32;
        let fps = if avg_dt > 0.0 { 1.0 / avg_dt } else { 0.0 };

        yakui::pad(yakui::widgets::Pad::all(16.0), || {
            yakui::align(yakui::Alignment::TOP_LEFT, || {
                yakui::colored_box_container(yakui::Color::rgba(20, 24, 32, 220), || {
                    yakui::pad(yakui::widgets::Pad::all(12.0), || {
                        yakui::column(|| {
                            yakui::label(format!("fps: {fps:5.1}  ({:.2} ms)", avg_dt * 1000.0));
                            yakui::label(format!("sim ticks: {}", ctx.clock.total_ticks()));
                            yakui::label(format!(
                                "sim time: {:6.2} s",
                                ctx.clock.sim_time().as_secs_f32()
                            ));

                            yakui::row(|| {
                                let label = if ctx.clock.is_paused() {
                                    "play"
                                } else {
                                    "pause"
                                };
                                if yakui::button(label).clicked {
                                    let new = if ctx.clock.is_paused() { 1.0 } else { 0.0 };
                                    ctx.clock.set_speed(new);
                                }
                                if yakui::button("quit").clicked {
                                    ctx.event_loop.exit();
                                }
                            });

                            let speed = ctx.clock.speed();
                            yakui::label(format!("speed: {speed:.2}"));
                            if let Some(new_speed) = yakui::slider(speed as f64, 0.0, 4.0).value {
                                ctx.clock.set_speed(new_speed as f32);
                            }
                        });
                    });
                });
            });
        });
    }
}

fn main() {
    currawong::run::<Hud>(());
}
