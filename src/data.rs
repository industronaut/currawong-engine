//! Data layer: virtual filesystem + definitions pipeline.
//!
//! Third top-level module alongside [`sim`](crate::sim) and `render`. Always
//! compiled — `cargo build --no-default-features` includes it. Pure data; no
//! `wgpu`/`winit` dependencies, and the I/O surface is shaped so a WASM port
//! is a straight swap of the bottom layer rather than a rewrite.
//!
//! ## Layers
//!
//! - [`VfsPath`] — newtype wrapping a normalised forward-slash path string.
//!   Constructor rejects `..`, drive letters, backslashes, and absolute
//!   paths. The grammar is enforced at the type level so mod-loaded layers
//!   can't escape the sandbox by handing the engine a bare string.
//! - [`AssetSource`] — async read + sync list. Implemented by [`FsSource`]
//!   (native disk) and [`MemorySource`] (in-memory; used for tests and the
//!   future WASM `include_bytes!` archive).
//! - [`Vfs`] — ordered stack of `AssetSource` layers. Top wins on read;
//!   listings union and dedupe across layers. Mods are additional layers
//!   mounted at startup.
//! - [`Definitions`] — RON-backed registry of namespaced kinds, populated
//!   from the VFS at startup before the sim is constructed.

mod definitions;
mod fs_source;
mod memory_source;
mod path;
mod source;
mod vfs;

#[cfg(test)]
mod test_helpers;

pub use definitions::{DefError, Definitions, KindDef, KindId, KindIdError};
pub use fs_source::FsSource;
pub use memory_source::MemorySource;
pub use path::{VfsPath, VfsPathError};
pub use source::{AssetError, AssetFuture, AssetSource};
pub use vfs::Vfs;
