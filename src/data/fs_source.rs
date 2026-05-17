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

use std::future::ready;
use std::path::{Path, PathBuf};
use std::{fs, io};

use super::path::VfsPath;
use super::source::{AssetError, AssetFuture, AssetSource};

/// An [`AssetSource`] backed by a directory on the host filesystem. All
/// VFS paths are resolved relative to [`root`](Self::root).
pub struct FsSource {
    root: PathBuf,
}

impl FsSource {
    /// Root this source at `root`. The directory does not have to exist at
    /// construction time — listing and reads return [`AssetError::NotFound`]
    /// (via the underlying `io::Error`) if it doesn't.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
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
