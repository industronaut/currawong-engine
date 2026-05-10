//! Demonstrates input handling and sim-speed control.
//!
//! - Esc: quit
//! - 0: pause
//! - 1: 1x (real-time)
//! - 2: 2x
//! - 3: 3x
//! - `-`: halve speed (down to 0.25x)
//! - `=`: double speed
//! - any other key/mouse press is logged to stdout

use currawong::{EngineCtx, Renderer, View, winit};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

struct Input;

impl View for Input {
    type Sim = ();

    fn init(_: &Renderer) -> Self {
        Input
    }

    fn input(&mut self, _: &mut (), ctx: &mut EngineCtx, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let PhysicalKey::Code(code) = event.physical_key else { return };
                match code {
                    KeyCode::Escape => ctx.event_loop.exit(),
                    KeyCode::Digit0 => set_speed(ctx, 0.0),
                    KeyCode::Digit1 => set_speed(ctx, 1.0),
                    KeyCode::Digit2 => set_speed(ctx, 2.0),
                    KeyCode::Digit3 => set_speed(ctx, 3.0),
                    KeyCode::Minus => set_speed(ctx, (ctx.clock.speed() * 0.5).max(0.25)),
                    KeyCode::Equal => set_speed(ctx, ctx.clock.speed() * 2.0),
                    _ => println!("key pressed: {code:?}"),
                }
            }
            WindowEvent::MouseInput { button, state, .. } if *state == ElementState::Pressed => {
                println!("mouse pressed: {button:?}");
            }
            _ => {}
        }
    }

    fn title() -> &'static str {
        "currawong — input"
    }
}

fn set_speed(ctx: &mut EngineCtx, speed: f32) {
    ctx.clock.set_speed(speed);
    println!(
        "speed: {speed}x  (ticks={}, sim_time={:.2}s)",
        ctx.clock.total_ticks(),
        ctx.clock.sim_time().as_secs_f32(),
    );
}

fn main() {
    currawong::run::<Input>(());
}
