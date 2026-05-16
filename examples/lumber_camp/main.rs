//! Lumber camp PoC — first end-to-end game on currawong (#58).
//!
//! This crate root is intentionally tiny: the example is split feature-vertically
//! into [`sim`] (sim-side state) and [`render`] (view-side rendering). New
//! modules land here as the game grows (jobs, pathfind, picking, hud, …) —
//! see issue #58 for the planned shape.
//!
//! Today the example renders a 16×16 flat square-grid zone with three pawns
//! (capsules), five trees (cones), and a stockpile (cube). Hovered objects
//! light up gold; left-clicking a tree toggles a designation marker on it.
//! Nothing moves yet — pawns are inert until the `Move` component slice
//! lands.
//!
//! Controls:
//! - Mouse over — hover highlight on the object under the cursor.
//! - Left-click on a tree — toggle "designated for chopping".
//! - Right-click drag — rotate the camera.
//! - W / A / S / D — pan the focal point.
//! - Scroll wheel — zoom.
//! - Esc — quit.

mod render;
mod sim;

fn main() {
    currawong::run::<render::LumberCampView>(sim::Game::new());
}
