//! Sanity check: every shipped kind RON file parses, and the kinds we've
//! declared interactable resolve to the expected [`Interaction`] variant.
//!
//! Lives outside `src/` so it loads the on-disk `assets/kinds/*.ron` through
//! the public `FsSource` + `Definitions::load` pipeline — the same path the
//! lumber_camp example uses at startup. Catches typos in either direction:
//! a malformed RON file fails the load; a renamed serde tag fails the
//! per-kind assertions below.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

use currawong::data::{Definitions, FsSource, KindId, Vfs, VfsPath};
use currawong::{Footprint, Interaction};

/// Single-threaded executor for driving `Definitions::load` to completion.
/// `pollster` is render-feature-gated, so duplicate the tiny stand-in here
/// rather than reach for the crate-private `data::test_helpers::block_on`.
fn block_on<F: Future>(mut fut: F) -> F::Output {
    struct ThreadWaker(Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut cx = Context::from_waker(&waker);
    // Safety: `fut` is owned here and never moved again.
    let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => thread::park(),
        }
    }
}

fn load_assets() -> Definitions {
    // CARGO_MANIFEST_DIR is the workspace root at test time. Assets live one
    // dir down at `assets/`; the definitions loader looks under `kinds/`.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    let mut vfs = Vfs::new();
    vfs.mount(FsSource::new(root));
    let prefix = VfsPath::new("kinds").unwrap();
    block_on(Definitions::load(&vfs, &prefix)).expect("assets/kinds/ should parse")
}

fn kind(s: &str) -> KindId {
    KindId::new(s).unwrap()
}

#[test]
fn shipped_kind_files_all_load() {
    let defs = load_assets();
    // Spot-check: at least the four lumber_camp-example kinds are present.
    // Mods adding new kinds later will still satisfy this.
    for id in [
        "currawong:oak_tree",
        "currawong:pine_tree",
        "currawong:lumber_camp",
        "currawong:lumberjack",
    ] {
        assert!(defs.get(&kind(id)).is_some(), "missing kind: {id}");
    }
}

#[test]
fn oak_tree_declares_surround_radius_1() {
    let defs = load_assets();
    let def = defs.get(&kind("currawong:oak_tree")).unwrap();
    let interaction = Interaction::from_def(def).unwrap();
    assert_eq!(interaction, Interaction::Surround { radius_tiles: 1 });
}

#[test]
fn pine_tree_declares_surround_radius_1() {
    let defs = load_assets();
    let def = defs.get(&kind("currawong:pine_tree")).unwrap();
    let interaction = Interaction::from_def(def).unwrap();
    assert_eq!(interaction, Interaction::Surround { radius_tiles: 1 });
}

#[test]
fn lumber_camp_declares_single_facing_offset() {
    let defs = load_assets();
    let def = defs.get(&kind("currawong:lumber_camp")).unwrap();
    let interaction = Interaction::from_def(def).unwrap();
    assert_eq!(
        interaction,
        Interaction::Facing {
            offsets: vec![(2, 1, 0)],
        }
    );
}

#[test]
fn lumberjack_has_no_interaction() {
    // Pawns aren't interacted with — the implicit-default-None path.
    let defs = load_assets();
    let def = defs.get(&kind("currawong:lumberjack")).unwrap();
    let interaction = Interaction::from_def(def).unwrap();
    assert_eq!(interaction, Interaction::None);
}

#[test]
fn lumber_camp_declares_two_by_two_footprint() {
    let defs = load_assets();
    let def = defs.get(&kind("currawong:lumber_camp")).unwrap();
    let footprint = Footprint::from_def(def).unwrap();
    assert_eq!(footprint, Footprint(vec![(0, 0), (1, 0), (0, 1), (1, 1)]));
}

#[test]
fn trees_declare_single_tile_footprint() {
    let defs = load_assets();
    for id in ["currawong:oak_tree", "currawong:pine_tree"] {
        let def = defs.get(&kind(id)).unwrap();
        let footprint = Footprint::from_def(def).unwrap();
        assert_eq!(footprint, Footprint(vec![(0, 0)]), "{id} footprint");
    }
}

#[test]
fn lumberjack_has_no_footprint() {
    // Pawns aren't placed on the grid — the implicit-default empty-footprint
    // path (no `footprint:` field in the def).
    let defs = load_assets();
    let def = defs.get(&kind("currawong:lumberjack")).unwrap();
    let footprint = Footprint::from_def(def).unwrap();
    assert!(footprint.is_empty());
}
