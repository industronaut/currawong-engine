//! Demonstrates input handling.
//!
//! Logs key and mouse-button presses to stdout. Press Escape to quit.

use currawong::{Renderer, View, winit};
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};

struct Input;

impl View for Input {
    type Sim = ();

    fn init(_: &Renderer) -> Self {
        Input
    }

    fn input(&mut self, _: &mut (), event_loop: &ActiveEventLoop, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                    event_loop.exit();
                    return;
                }
                println!("key pressed: {:?}", event.physical_key);
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

fn main() {
    currawong::run::<Input>(());
}
