//! Minimal example: open a window and clear it each frame.

use currawong::{Renderer, View};

struct Clear;

impl View for Clear {
    type Sim = ();

    fn init(_: &Renderer) -> Self {
        Clear
    }

    fn title() -> &'static str {
        "currawong — clear"
    }
}

fn main() {
    currawong::run::<Clear>(());
}
