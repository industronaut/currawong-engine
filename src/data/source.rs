//! [`AssetSource`] — the trait every VFS layer implements.
//!
//! `read` is async because the eventual web port has no synchronous
//! filesystem and a "block until disk catches up" call would be a hard error
//! there. PR1 implementations resolve immediately (`std::future::ready`) —
//! real concurrency lands in PR2 when streaming handles arrive — but the
//! signature is shaped now so the executor swap is the only change.
//!
//! `list` is synchronous: every PR1 source can produce its listing without
//! I/O (`MemorySource` keeps a `HashMap`, `FsSource` walks `std::fs`
//! eagerly). When that stops being true (large archives with on-demand
//! indexes, networked sources) we can promote `list` to async too; doing it
//! now would force every caller into an `await` they don't need.

use std::fmt;
use std::future::Future;
use std::io;
use std::pin::Pin;

use super::path::VfsPath;

/// Boxed future returned by [`AssetSource::read`]. Boxed because
/// `dyn AssetSource` has to be object-safe; a bare `async fn` in the trait
/// would lose object safety. Avoids pulling in `async-trait` — the boxing
/// only appears at the trait declaration, never at call sites.
pub type AssetFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, AssetError>> + Send + 'a>>;

/// Errors a source can return from [`AssetSource::read`] (and that bubble
/// up through [`Vfs::read`](super::Vfs::read)).
#[derive(Debug)]
pub enum AssetError {
    /// The source has no entry at this path. For [`Vfs`](super::Vfs), this
    /// is a *layer-local* miss: the VFS keeps walking to lower layers.
    NotFound,
    /// Underlying I/O failure (disk error, permission denied, etc.). Wraps
    /// the [`io::Error`] verbatim so callers can match on its `kind` if
    /// they need to distinguish.
    Io(io::Error),
}

impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => f.write_str("asset not found"),
            Self::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for AssetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::NotFound => None,
        }
    }
}

impl From<io::Error> for AssetError {
    fn from(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::NotFound {
            AssetError::NotFound
        } else {
            AssetError::Io(e)
        }
    }
}

/// A read-only source of assets, addressable by [`VfsPath`].
///
/// Implementations are mounted into a [`Vfs`](super::Vfs) as ordered
/// layers; the VFS walks them top-to-bottom on each read. Implementations
/// must be `Send + Sync` so the VFS itself is `Send + Sync` and can be
/// shared across the renderer's threads.
pub trait AssetSource: Send + Sync {
    /// Read the bytes at `path`. Returns [`AssetError::NotFound`] if this
    /// source has no entry there — the VFS treats that as "fall through to
    /// the next layer" rather than a hard error.
    fn read<'a>(&'a self, path: &'a VfsPath) -> AssetFuture<'a, Vec<u8>>;

    /// Every path this source can serve, in unspecified order. The VFS
    /// unions and dedupes across layers; individual sources are not
    /// required to sort or dedupe their own output.
    fn list(&self) -> Result<Vec<VfsPath>, AssetError>;

    /// Drain queued change notifications since the last call. Each returned
    /// path is one the consumer should treat as potentially-stale: bytes
    /// changed, file was created, renamed, or removed. The VFS unions the
    /// per-layer drains; the [`AssetServer`](crate::AssetServer) then
    /// evicts its cache entries and the next request spawns a fresh load.
    ///
    /// Default returns empty — sources that don't watch (in-memory archives,
    /// the eventual WASM `include_bytes!` layer) never produce
    /// notifications. [`FsSource`](super::FsSource) implements it via the
    /// `notify` crate when [`FsSource::start_watching`](super::FsSource::start_watching)
    /// is called.
    ///
    /// Takes `&self` rather than `&mut self` because [`Vfs`](super::Vfs)
    /// holds sources as `Box<dyn AssetSource>` and can't hand out mutable
    /// borrows; implementations carry their own interior mutability.
    fn drain_changes(&self) -> Vec<VfsPath> {
        Vec::new()
    }
}
