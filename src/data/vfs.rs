//! [`Vfs`] — the layered overlay that stitches multiple [`AssetSource`]s
//! into one read-only namespace.
//!
//! Ordered top-to-bottom by precedence. Mod-friendly by construction: a
//! base layer (game assets) plus N override layers (mods) — the top layer
//! wins on every read, and a path that's only in the base layer falls
//! through cleanly. Listings union across all layers and dedupe by path.

use std::collections::HashSet;

use super::path::VfsPath;
use super::source::{AssetError, AssetSource};

/// Ordered stack of [`AssetSource`] layers. Layer 0 is the top (highest
/// precedence); reads walk forward and return the first hit.
#[derive(Default)]
pub struct Vfs {
    layers: Vec<Box<dyn AssetSource>>,
}

impl Vfs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `source` to the top of the stack — it overrides everything
    /// already mounted. Typical startup order: mount the base archive
    /// first, then each mod in load order; the last mounted layer ends up
    /// on top and wins ties.
    pub fn mount<S: AssetSource + 'static>(&mut self, source: S) {
        self.layers.insert(0, Box::new(source));
    }

    /// Number of mounted layers. Useful for "did the mod load?" assertions.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Read `path`, walking top to bottom. Returns
    /// [`AssetError::NotFound`] iff no layer has it; layer-local I/O errors
    /// short-circuit and are returned immediately rather than being
    /// swallowed (a corrupted top layer should surface, not be papered
    /// over by a lower one).
    pub async fn read(&self, path: &VfsPath) -> Result<Vec<u8>, AssetError> {
        for source in &self.layers {
            match source.read(path).await {
                Ok(bytes) => return Ok(bytes),
                Err(AssetError::NotFound) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(AssetError::NotFound)
    }

    /// Union of every layer's listing, deduplicated by path, sorted
    /// lexicographically. The sort is load-bearing: per-layer listings
    /// arrive in OS- and filesystem-defined order (`fs::read_dir` makes no
    /// guarantees), so without it [`Definitions::load`](super::Definitions)
    /// would register kinds in a different sequence on different machines
    /// and the resulting `IndexMap` iteration would no longer be
    /// reproducible.
    pub fn list(&self) -> Result<Vec<VfsPath>, AssetError> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for source in &self.layers {
            for path in source.list()? {
                if seen.insert(path.clone()) {
                    out.push(path);
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// Drain queued change notifications from every layer that supports
    /// watching. Returns deduped, sorted paths — same convention as
    /// [`Self::list`]. The [`AssetServer`](crate::AssetServer) pump calls
    /// this once per frame and uses the result to evict cached handles.
    ///
    /// A path appearing here is a *possibly-stale* signal: bytes may have
    /// changed, the file may have been created or removed. The asset
    /// server's response is uniform — evict the cache entry and let the
    /// next request spawn a fresh load.
    pub fn drain_changes(&self) -> Vec<VfsPath> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for source in &self.layers {
            for path in source.drain_changes() {
                if seen.insert(path.clone()) {
                    out.push(path);
                }
            }
        }
        out.sort();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::memory_source::MemorySource;
    use crate::data::test_helpers::block_on;

    fn p(s: &str) -> VfsPath {
        VfsPath::new(s).unwrap()
    }

    #[test]
    fn empty_vfs_returns_not_found() {
        let vfs = Vfs::new();
        match block_on(vfs.read(&p("foo"))) {
            Err(AssetError::NotFound) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn top_layer_overrides_base() {
        let mut vfs = Vfs::new();
        vfs.mount(MemorySource::new().with("config.ron", b"base".to_vec()));
        vfs.mount(MemorySource::new().with("config.ron", b"override".to_vec()));

        let bytes = block_on(vfs.read(&p("config.ron"))).unwrap();
        assert_eq!(bytes, b"override");
    }

    #[test]
    fn falls_through_when_top_silent() {
        let mut vfs = Vfs::new();
        vfs.mount(MemorySource::new().with("only_in_base.ron", b"base"));
        // Top layer doesn't carry this path.
        vfs.mount(MemorySource::new().with("other.ron", b"override"));

        let bytes = block_on(vfs.read(&p("only_in_base.ron"))).unwrap();
        assert_eq!(bytes, b"base");
    }

    #[test]
    fn listing_unions_and_dedupes() {
        let mut vfs = Vfs::new();
        vfs.mount(
            MemorySource::new()
                .with("shared.ron", b"")
                .with("only_in_base.ron", b""),
        );
        vfs.mount(
            MemorySource::new()
                .with("shared.ron", b"")
                .with("only_in_override.ron", b""),
        );

        let listing: Vec<String> = vfs
            .list()
            .unwrap()
            .into_iter()
            .map(|p| p.as_str().to_owned())
            .collect();
        assert_eq!(
            listing,
            vec!["only_in_base.ron", "only_in_override.ron", "shared.ron",]
        );
    }

    #[test]
    fn drain_changes_empty_without_watching_layers() {
        // No watcher mounted (MemorySource doesn't override drain_changes);
        // the aggregate is empty and the asset-server pump becomes a no-op.
        let mut vfs = Vfs::new();
        vfs.mount(MemorySource::new().with("foo", b""));
        assert!(vfs.drain_changes().is_empty());
    }

    #[test]
    fn drain_changes_unions_dedupes_and_sorts() {
        use crate::data::source::{AssetError, AssetFuture, AssetSource};
        use std::sync::Mutex;
        // A tiny fake source that pretends to be watching: drain_changes
        // returns whatever paths the test parked in its queue.
        struct Parked(Mutex<Vec<VfsPath>>);
        impl AssetSource for Parked {
            fn read<'a>(&'a self, _: &'a VfsPath) -> AssetFuture<'a, Vec<u8>> {
                Box::pin(std::future::ready(Err(AssetError::NotFound)))
            }
            fn list(&self) -> Result<Vec<VfsPath>, AssetError> {
                Ok(Vec::new())
            }
            fn drain_changes(&self) -> Vec<VfsPath> {
                std::mem::take(&mut *self.0.lock().unwrap())
            }
        }

        let mut vfs = Vfs::new();
        // Two layers each parking some changes — "shared" lands twice.
        vfs.mount(Parked(Mutex::new(vec![p("a"), p("shared")])));
        vfs.mount(Parked(Mutex::new(vec![p("shared"), p("b")])));
        let changes: Vec<String> = vfs
            .drain_changes()
            .into_iter()
            .map(|p| p.as_str().to_owned())
            .collect();
        // Sorted, deduped — same shape as list().
        assert_eq!(changes, vec!["a", "b", "shared"]);
        // Second drain is empty — the fake sources returned their queues
        // by `mem::take`, and a real watcher's channel works the same way.
        assert!(vfs.drain_changes().is_empty());
    }

    #[test]
    fn three_layer_precedence() {
        // Mount order base → mid → top: top wins where it has the path,
        // mid where top is silent, base where both upper layers are.
        let mut vfs = Vfs::new();
        vfs.mount(
            MemorySource::new()
                .with("a", b"base-a")
                .with("b", b"base-b")
                .with("c", b"base-c"),
        );
        vfs.mount(MemorySource::new().with("b", b"mid-b").with("c", b"mid-c"));
        vfs.mount(MemorySource::new().with("c", b"top-c"));

        assert_eq!(block_on(vfs.read(&p("a"))).unwrap(), b"base-a");
        assert_eq!(block_on(vfs.read(&p("b"))).unwrap(), b"mid-b");
        assert_eq!(block_on(vfs.read(&p("c"))).unwrap(), b"top-c");
    }
}
