//! [`YakuiAssets`] — the yakui-managed-texture equivalent of
//! [`AssetServer`](super::AssetServer).
//!
//! ## Why a separate cache
//!
//! Yakui owns its own texture arena. Widgets like
//! [`nineslice`](https://docs.rs/yakui-widgets/latest/yakui_widgets/widgets/struct.NineSlice.html)
//! take a [`ManagedTextureId`] — an id into yakui's internal pool — not a
//! `wgpu::TextureView` we hand over. Registering goes through
//! [`Yakui::add_texture`], which keeps the RGBA8 bytes resident and uploads
//! them through yakui-wgpu's own path. So the VFS → GPU seam for game UI
//! sidesteps [`AssetServer`] entirely; this cache fills that role instead.
//!
//! ## Sync, not async
//!
//! [`AssetServer`](super::AssetServer) streams textures with a magenta
//! fallback because the render path always needs *something* to bind every
//! frame. UI chrome doesn't have a meaningful loading state: you can't draw
//! a window frame without its corners. The first call for a path blocks on
//! the VFS read + decode + upload; subsequent calls hit the cache. Tiny
//! per-startup cost in exchange for never having to write fallback widgets.
//!
//! ## Memory note
//!
//! Yakui's managed-texture arena keeps the RGBA8 bytes alive indefinitely
//! (no eviction). For typical UI chrome (window frames, button atlases) this
//! is fine; for hundreds of large textures it would matter and an eviction
//! pass would land alongside.

use std::collections::HashMap;
use std::sync::Arc;

use glam::UVec2;
use yakui::paint::{Texture as YakuiTexture, TextureFormat as YakuiTextureFormat};
use yakui::{ManagedTextureId, Yakui};

use crate::data::{AssetError, Vfs, VfsPath};

use super::texture::{TextureColorSpace, TextureLoadError, decode_rgba8_from_bytes};

/// View-side cache from [`VfsPath`] → [`ManagedTextureId`] for yakui game UI.
///
/// Construct in [`View::init`](crate::View::init) alongside
/// [`AssetServer`](super::AssetServer), typically sharing the same
/// [`Arc<Vfs>`] so mods and base assets resolve identically across the
/// rendering and UI paths. Pass the same `&mut yakui::Yakui` the engine
/// hands to [`View::game_ui`](crate::View::game_ui) into
/// [`texture`](Self::texture).
pub struct YakuiAssets {
    vfs: Arc<Vfs>,
    cache: HashMap<CacheKey, ManagedTextureId>,
}

/// Cache key. Colour-space mirrors [`AssetServer`](super::AssetServer)'s
/// keying: the same PNG sampled as sRGB vs linear is semantically a
/// different texture.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct CacheKey {
    path: VfsPath,
    color_space: TextureColorSpace,
}

impl YakuiAssets {
    /// Build a fresh cache over `vfs`. Empty until the first
    /// [`texture`](Self::texture) call.
    pub fn new(vfs: Arc<Vfs>) -> Self {
        Self {
            vfs,
            cache: HashMap::new(),
        }
    }

    /// Get-or-load a yakui-managed texture at `path`, interpreted as sRGB
    /// colour data (the right choice for UI chrome — window frames, button
    /// art, icons). First call for a `(path, sRGB)` pair blocks: VFS read,
    /// PNG decode, [`Yakui::add_texture`] upload. Subsequent calls return
    /// the cached id immediately.
    ///
    /// For data-style images (signed-distance fields, mask atlases) use
    /// [`texture_with_color_space`](Self::texture_with_color_space) with
    /// [`TextureColorSpace::Linear`].
    pub fn texture(
        &mut self,
        yakui: &mut Yakui,
        path: &VfsPath,
    ) -> Result<ManagedTextureId, YakuiAssetError> {
        self.texture_with_color_space(yakui, path, TextureColorSpace::Srgb)
    }

    /// Get-or-load variant that lets the caller pick the colour-space
    /// interpretation. Linear is only correct for data-style images; for
    /// anything visible to the player, prefer [`texture`](Self::texture).
    pub fn texture_with_color_space(
        &mut self,
        yakui: &mut Yakui,
        path: &VfsPath,
        color_space: TextureColorSpace,
    ) -> Result<ManagedTextureId, YakuiAssetError> {
        let key = CacheKey {
            path: path.clone(),
            color_space,
        };
        if let Some(&id) = self.cache.get(&key) {
            return Ok(id);
        }

        let bytes = pollster::block_on(self.vfs.read(path)).map_err(YakuiAssetError::Vfs)?;
        let (width, height, rgba) =
            decode_rgba8_from_bytes(&bytes).map_err(YakuiAssetError::Decode)?;

        let format = match color_space {
            TextureColorSpace::Srgb => YakuiTextureFormat::Rgba8Srgb,
            // Yakui's `Texture` doesn't carry a linear RGBA8 variant —
            // its formats are tuned to UI compositing. The pragmatic fit
            // for "linear data interpreted as a UI image" is still
            // `Rgba8Srgb` (yakui's blend math expects sRGB pixels);
            // grant the caller's intent and accept the slight gamma slip.
            // Revisit if a real linear-data UI use case appears.
            TextureColorSpace::Linear => YakuiTextureFormat::Rgba8Srgb,
        };
        let texture = YakuiTexture::new(format, UVec2::new(width, height), rgba);
        let id = yakui.add_texture(texture);
        self.cache.insert(key, id);
        Ok(id)
    }
}

/// Failure modes for [`YakuiAssets::texture`] — VFS read or image decode.
#[derive(Debug)]
pub enum YakuiAssetError {
    Vfs(AssetError),
    Decode(TextureLoadError),
}

impl std::fmt::Display for YakuiAssetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vfs(e) => write!(f, "yakui asset VFS error: {e}"),
            Self::Decode(e) => write!(f, "yakui asset decode error: {e}"),
        }
    }
}

impl std::error::Error for YakuiAssetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vfs(e) => Some(e),
            Self::Decode(e) => Some(e),
        }
    }
}
