//! [`FsSource`] — an [`AssetSource`] rooted at a host filesystem directory.
//!
//! Native-only loose-files mode: dev iteration without a build step. The
//! `read` is implemented synchronously and wrapped in `std::future::ready`
//! — file I/O blocks on the calling thread for PR1. PR2 will introduce
//! thread-pool offloading; the trait already returns a future so callers
//! don't change.
//!
//! WASM has no analogue — `wasm32` builds mount a [`MemorySource`] with the
//! base archive embedded via `include_bytes!`. That's not in this PR; the
//! note is here as a reminder when WASM lands.
//!
//! ## Hot reload
//!
//! Opt-in via [`FsSource::start_watching`]. Spawns a `notify` recursive
//! watcher rooted at [`Self::root`]; on every filesystem event the source
//! translates the host path back into a [`VfsPath`] (relative to root,
//! forward slashes, sandbox-validated) and queues it on an mpsc channel.
//! [`AssetSource::drain_changes`] hands the queued paths back to the VFS,
//! which forwards them to the [`AssetServer`](crate::AssetServer) for cache
//! eviction.
//!
//! The watcher lives in interior mutability behind a `Mutex<Option<…>>` so
//! `start_watching` can take `&mut self` while the trait surface
//! (`drain_changes` / `read` / `list`) stays `&self`-friendly — `Vfs` only
//! holds boxed trait objects and can't hand out mutable borrows.

use std::future::ready;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver};
use std::{fs, io};

use notify::{RecommendedWatcher, RecursiveMode, Watcher};

use super::path::VfsPath;
use super::source::{AssetError, AssetFuture, AssetSource};

/// An [`AssetSource`] backed by a directory on the host filesystem. All
/// VFS paths are resolved relative to [`root`](Self::root).
pub struct FsSource {
    root: PathBuf,
    /// Hot-reload state. `None` until [`Self::start_watching`] is called.
    /// `Mutex` because the trait surface is `&self` and the receiver isn't
    /// `Sync` on its own.
    watcher: Mutex<Option<WatcherState>>,
}

/// Per-source watcher state: the live `notify` handle (kept around for its
/// `Drop` to tear down the OS-level watch) plus the channel the closure
/// writes into.
struct WatcherState {
    /// The closure inside this watcher captures the sender; dropping the
    /// watcher also drops the closure, closing the channel cleanly.
    _watcher: RecommendedWatcher,
    rx: Receiver<VfsPath>,
}

impl FsSource {
    /// Root this source at `root`. The directory does not have to exist at
    /// construction time — listing and reads return [`AssetError::NotFound`]
    /// (via the underlying `io::Error`) if it doesn't.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            watcher: Mutex::new(None),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Begin watching [`Self::root`] for changes. From this point on,
    /// [`AssetSource::drain_changes`] returns any VFS paths whose underlying
    /// bytes have changed since the previous drain.
    ///
    /// One watcher per source; calling twice replaces the previous one.
    /// `root` must exist at the time of this call — `notify` errors out
    /// otherwise. Hot reload is a dev-time convenience; the caller is
    /// expected to be fine with the watch failing (e.g. when running headless
    /// without an asset tree) and surface a warning rather than aborting.
    pub fn start_watching(&mut self) -> Result<(), notify::Error> {
        let (tx, rx) = mpsc::channel::<VfsPath>();
        // Canonicalise root for prefix-stripping inside the closure.
        // FSEvents on macOS reports paths under `/private/var/...` even
        // when the caller passes a `/var/...` symlink, and Windows can
        // surface UNC-prefixed forms; canonicalisation normalises both so
        // `strip_prefix` matches. Fall back to the raw root if canonicalise
        // fails (e.g. root doesn't exist yet) — `notify::watch` below will
        // surface the real error.
        let canonical_root = fs::canonicalize(&self.root).unwrap_or_else(|_| self.root.clone());
        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                let Ok(event) = res else {
                    return;
                };
                for path in event.paths {
                    if let Some(vfs) = vfs_path_for(&canonical_root, &path) {
                        // Send failures (receiver dropped) are silent —
                        // the source has been torn down, nothing to do.
                        let _ = tx.send(vfs);
                    }
                }
            })?;
        watcher.watch(&self.root, RecursiveMode::Recursive)?;
        *self.watcher.lock().expect("watcher mutex poisoned") = Some(WatcherState {
            _watcher: watcher,
            rx,
        });
        Ok(())
    }

    /// Translate a [`VfsPath`] into a host path under [`root`](Self::root).
    /// Safe because `VfsPath` is sandbox-validated — no `..`, no absolute
    /// paths, no backslashes — so the join can't escape `root`.
    fn host_path(&self, path: &VfsPath) -> PathBuf {
        self.root.join(path.as_str())
    }
}

impl AssetSource for FsSource {
    fn read<'a>(&'a self, path: &'a VfsPath) -> AssetFuture<'a, Vec<u8>> {
        let result = fs::read(self.host_path(path)).map_err(AssetError::from);
        Box::pin(ready(result))
    }

    fn list(&self) -> Result<Vec<VfsPath>, AssetError> {
        let mut out = Vec::new();
        match walk(&self.root, &self.root, &mut out) {
            Ok(()) => Ok(out),
            // A missing root is a valid "no assets here" state for an
            // unmounted mods directory — return an empty listing rather
            // than a hard error.
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(AssetError::Io(e)),
        }
    }

    fn drain_changes(&self) -> Vec<VfsPath> {
        let guard = self.watcher.lock().expect("watcher mutex poisoned");
        let Some(state) = guard.as_ref() else {
            return Vec::new();
        };
        // try_recv until empty — the channel is the only buffer; consumers
        // get a single drain per call and any events queued in the meantime
        // come through on the next call.
        let mut out = Vec::new();
        while let Ok(path) = state.rx.try_recv() {
            out.push(path);
        }
        out
    }
}

/// Translate a notify event path back into a [`VfsPath`] relative to
/// `root`. Returns `None` for paths outside the root, paths with non-UTF-8
/// segments, or paths that fail the VFS grammar (e.g. the root itself,
/// which would normalise to the empty string).
fn vfs_path_for(root: &Path, host_path: &Path) -> Option<VfsPath> {
    let rel = host_path.strip_prefix(root).ok()?;
    let rel_str = rel.to_str()?;
    let normalised = rel_str.replace(std::path::MAIN_SEPARATOR, "/");
    VfsPath::new(normalised).ok()
}

/// Recursively walk `dir`, pushing each regular file's VFS path (relative
/// to `root`) into `out`. Skips entries whose names contain characters the
/// VFS grammar rejects (backslashes on Unix paths, colons in macOS
/// filenames) — silently, so a mod with a stray file can't crash the load.
fn walk(root: &Path, dir: &Path, out: &mut Vec<VfsPath>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let entry_path = entry.path();
        if file_type.is_dir() {
            walk(root, &entry_path, out)?;
        } else if file_type.is_file() {
            let Ok(rel) = entry_path.strip_prefix(root) else {
                continue;
            };
            // OsStr → &str: filenames with non-UTF8 bytes are skipped.
            let Some(rel_str) = rel.to_str() else {
                continue;
            };
            // Use forward slashes regardless of host. Windows hosts have
            // `\` as their MAIN_SEPARATOR; replace before validating.
            let normalised = rel_str.replace(std::path::MAIN_SEPARATOR, "/");
            if let Ok(vfs) = VfsPath::new(normalised) {
                out.push(vfs);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::io::Write;

    /// Spin up a unique temp directory under the OS temp root, populate it,
    /// and return the path. The directory is cleaned up at drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut p = env::temp_dir();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            p.push(format!("currawong-fs-source-{}-{}", tag, nanos));
            fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }

        fn write(&self, rel: &str, bytes: &[u8]) {
            let full = self.0.join(rel);
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            let mut f = fs::File::create(full).unwrap();
            f.write_all(bytes).unwrap();
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    use crate::data::test_helpers::block_on;

    #[test]
    fn reads_a_file_from_disk() {
        let tmp = TempDir::new("read");
        tmp.write("a/b.txt", b"hello");
        let src = FsSource::new(&tmp.0);
        let path = VfsPath::new("a/b.txt").unwrap();
        let bytes = block_on(src.read(&path)).unwrap();
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn missing_file_is_not_found() {
        let tmp = TempDir::new("missing");
        let src = FsSource::new(&tmp.0);
        let path = VfsPath::new("nope").unwrap();
        match block_on(src.read(&path)) {
            Err(AssetError::NotFound) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn missing_root_lists_empty() {
        let src = FsSource::new("/path/that/definitely/does/not/exist/currawong-test");
        let listing = src.list().unwrap();
        assert!(listing.is_empty());
    }

    #[test]
    fn vfs_path_for_translates_relative_paths() {
        let root = PathBuf::from("/tmp/root");
        assert_eq!(
            vfs_path_for(&root, &PathBuf::from("/tmp/root/a/b.ron")).map(|p| p.as_str().to_owned()),
            Some("a/b.ron".to_owned()),
        );
        // Outside the root → None (notify can fire on parent dir events).
        assert!(vfs_path_for(&root, &PathBuf::from("/tmp/other/c.ron")).is_none());
        // The root itself normalises to empty → VfsPath rejects → None.
        assert!(vfs_path_for(&root, &root).is_none());
    }

    #[test]
    fn drain_changes_empty_until_watcher_started() {
        // Before start_watching, the source has no channel and the trait
        // method returns the default empty vec.
        let tmp = TempDir::new("drain-empty");
        let src = FsSource::new(&tmp.0);
        assert!(src.drain_changes().is_empty());
    }

    /// Smoke test the live notify watcher. Timing-dependent: `notify`
    /// coalesces events at the OS layer (FSEvents on macOS can take
    /// hundreds of ms to flush), so the sleep windows have to be
    /// generous. `#[ignore]` so the default `cargo test` stays
    /// deterministic; run with `cargo test -- --ignored` to exercise.
    #[test]
    #[ignore = "filesystem-watcher timing — run manually with --ignored"]
    fn watcher_surfaces_writes_as_drain_changes() {
        let tmp = TempDir::new("watcher-smoke");
        // Plant the file *before* start_watching so notify is registering a
        // pre-existing tree, then mutate it — that exercises the modify
        // path, which is the common dev-iteration shape (editing an
        // existing kind file rather than creating one).
        tmp.write("seed.ron", b"initial");
        let mut src = FsSource::new(&tmp.0);
        src.start_watching().expect("watcher started");
        // FSEvents needs a generous beat to register the watch on macOS.
        std::thread::sleep(std::time::Duration::from_millis(1000));
        tmp.write("seed.ron", b"updated");
        std::thread::sleep(std::time::Duration::from_millis(1500));
        let changes: Vec<String> = src
            .drain_changes()
            .into_iter()
            .map(|p| p.as_str().to_owned())
            .collect();
        assert!(
            changes.iter().any(|p| p == "seed.ron"),
            "expected `seed.ron` in {changes:?}",
        );
    }

    #[test]
    fn lists_files_recursively() {
        let tmp = TempDir::new("list");
        tmp.write("a.ron", b"");
        tmp.write("kinds/oak.ron", b"");
        tmp.write("kinds/sub/birch.ron", b"");

        let src = FsSource::new(&tmp.0);
        let mut listing: Vec<String> = src
            .list()
            .unwrap()
            .into_iter()
            .map(|p| p.as_str().to_owned())
            .collect();
        listing.sort();
        assert_eq!(
            listing,
            vec!["a.ron", "kinds/oak.ron", "kinds/sub/birch.ron",]
        );
    }
}
