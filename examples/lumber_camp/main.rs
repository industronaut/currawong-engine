//! Lumber camp PoC — first end-to-end game on currawong (#58).
//!
//! Migrated onto the asset pipeline as the integration proof for #67:
//! - `assets/kinds/*.ron` carries tree species + lumberjack + lumber-camp
//!   stats and points at each kind's mesh / texture paths.
//! - Visual content (glTF meshes + PNG textures) streams via
//!   [`AssetServer`](currawong::AssetServer); first frames show the magenta
//!   fallback while the background threads decode, then snap to the real
//!   asset on the transition frame.
//!
//! ## Layout
//!
//! - [`sim`] owns gameplay state. `Game::new(defs)` reads the kind defs at
//!   startup and caches resolved [`KindId`](currawong::data::KindId)s +
//!   numeric stats on `Game::stats`.
//! - [`render`] owns view state. `View::init` builds a fresh [`Vfs`]
//!   alongside the sim's (cheap — same on-disk content, independent
//!   caches), registers one [`RenderTemplate`](currawong::RenderTemplate)
//!   per kind in the defs, and dispatches the per-instance update on a
//!   precomputed `KindId → RenderShape` table.
//!
//! Controls:
//! - Mouse over — hover highlight on the object under the cursor.
//! - Left-click on a tree — toggle "designated for chopping".
//! - Right-click drag — rotate the camera.
//! - W / A / S / D — pan the focal point.
//! - Scroll wheel — zoom.
//! - F (hold) — force the magenta fallback (`AssetServer::set_force_loading`).
//! - Esc — quit.

mod render;
mod sim;

use std::path::Path;
use std::sync::Arc;

use currawong::data::{Definitions, FsSource, Vfs, VfsPath};
use currawong::pollster;

/// Mount the repo's `assets/` directory as a fresh [`Vfs`]. Called twice
/// at startup — once by main to populate [`Definitions`], once by the
/// view to back the [`AssetServer`](currawong::AssetServer). Two
/// independent [`Vfs`] instances over the same on-disk content is the
/// right shape for an example: each side stays autonomous, and there's
/// no engine-level requirement that they share one. Mods would mount
/// additional layers on top of this base.
pub(crate) fn lumber_camp_vfs() -> Vfs {
    let assets_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    let mut vfs = Vfs::new();
    vfs.mount(FsSource::new(assets_root));
    vfs
}

fn main() {
    // Phase 1: load defs synchronously-from-the-sim's-perspective. The
    // VFS read happens on a background thread (already-resident in the
    // `FsSource` case but the API is async-shaped for the eventual WASM
    // port); `pollster::block_on` parks main until every `.ron` under
    // `kinds/` is parsed.
    let vfs = Arc::new(lumber_camp_vfs());
    let kinds_prefix = VfsPath::new("kinds").expect("valid VFS path");
    let defs = pollster::block_on(Definitions::load(&vfs, &kinds_prefix))
        .expect("loading kind definitions");

    // Phase 2: hand the resolved defs to the sim. From here on the sim
    // owns its own state; defs are read once and never touched again.
    let game = sim::Game::new(defs);

    // Phase 3: hand off to the engine. `View::init` will mount its own
    // VFS for visual streaming — see `lumber_camp_vfs` above for the
    // shared-content / separate-instance shape.
    currawong::run::<render::LumberCampView>(game);
}
