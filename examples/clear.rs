//! Minimal example: open a window and clear it each frame.

use currawong::{Renderer, View, ViewConfig};

struct Clear;

impl View for Clear {
    type Sim = ();

    fn init(_: &Renderer) -> (Self, ViewConfig) {
        (
            Clear,
            ViewConfig {
                title: "currawong — clear",
                ..Default::default()
            },
        )
    }
}

fn main() {
    currawong::run::<Clear>(());
}
