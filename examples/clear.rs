//! Minimal example: open a window and clear it each frame.

use currawong::{Renderer, View, ViewConfig};

struct Clear;

impl View for Clear {
    type Sim = ();

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — clear",
        ..ViewConfig::DEFAULT
    };

    fn init(_: &Renderer) -> Self {
        Clear
    }
}

fn main() {
    currawong::run::<Clear>(());
}
