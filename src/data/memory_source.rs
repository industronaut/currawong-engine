//! [`MemorySource`] — an [`AssetSource`] backed by an in-memory map.
//!
//! Two roles, same shape:
//! - Tests stand up known content without touching the filesystem.
//! - The eventual WASM build will mount the base archive via
//!   `include_bytes!`, which is a `&'static [u8]` blob plus an index — the
//!   same `HashMap<VfsPath, Vec<u8>>` shape.

use std::collections::HashMap;
use std::future::ready;

use super::path::VfsPath;
use super::source::{AssetError, AssetFuture, AssetSource};

/// In-memory [`AssetSource`]. Build with [`MemorySource::new`] and populate
/// via [`insert`](Self::insert).
#[derive(Default)]
pub struct MemorySource {
    files: HashMap<VfsPath, Vec<u8>>,
}

impl MemorySource {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or overwrite) the bytes at `path`. Returns the previous bytes
    /// if there were any.
    pub fn insert(&mut self, path: VfsPath, bytes: impl Into<Vec<u8>>) -> Option<Vec<u8>> {
        self.files.insert(path, bytes.into())
    }

    /// Convenience for tests: insert a path given as a `&str`, panicking
    /// on an invalid path. Not for production code — `insert` is the
    /// real API.
    #[cfg(test)]
    pub(crate) fn with(mut self, path: &str, bytes: impl Into<Vec<u8>>) -> Self {
        self.insert(VfsPath::new(path).expect("valid test path"), bytes);
        self
    }
}

impl AssetSource for MemorySource {
    fn read<'a>(&'a self, path: &'a VfsPath) -> AssetFuture<'a, Vec<u8>> {
        let result = self.files.get(path).cloned().ok_or(AssetError::NotFound);
        Box::pin(ready(result))
    }

    fn list(&self) -> Result<Vec<VfsPath>, AssetError> {
        Ok(self.files.keys().cloned().collect())
    }
}
