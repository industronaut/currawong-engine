//! Minimal example: open a window and clear it each frame.

use currawong::{App, Renderer};

struct Clear;

impl App for Clear {
    fn init(_: &Renderer) -> Self {
        Clear
    }

    fn title() -> &'static str {
        "currawong — clear"
    }
}

fn main() {
    currawong::run::<Clear>();
}
