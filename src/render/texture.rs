//! Texture asset and the canonical sampler set.
//!
//! ## Roles
//!
//! - [`Texture`] owns a `wgpu::Texture` + `wgpu::TextureView`, plus enough
//!   metadata (size, format, mip count) to introspect after upload. Built
//!   from raw RGBA8 byte slices via [`Texture::from_rgba8`] — generates
//!   mipmaps on CPU using box-filter downsampling.
//! - [`SamplerKind`] is a closed enum of canonical sampler configurations.
//!   Adding a new one is a deliberate code change.
//! - [`SamplerRegistry`] holds the live `wgpu::Sampler` for each
//!   [`SamplerKind`]. Build once at `View::init` and share across
//!   materials.
//!
//! ## Constructor surface
//!
//! Each upload path comes in two flavours. The `_with_device` variants take
//! `&wgpu::Device` + `&wgpu::Queue` directly so background loaders running
//! on a worker thread can upload without touching `Renderer` (which is
//! single-owner and lives on the main thread). The `Renderer`-flavoured
//! constructors are the convenience wrappers user code reaches for.
//!
//! ## What's not here yet
//!
//! - glTF / asset-manager plumbing. [`Texture::from_path`] reads PNG/JPEG
//!   from disk synchronously; streaming flows through
//!   [`AssetServer`](super::AssetServer) and `from_png_bytes_with_device`.
//! - Linear-space mipmap downsampling for sRGB. We average in raw 8-bit
//!   sRGB space, which is slightly wrong but invisible at the quality
//!   level we render at. Worth revisiting before shipping.

use std::collections::HashMap;
use std::path::Path;

use super::renderer::Renderer;

/// A 2D texture with a precomputed mip chain.
///
/// Built from raw RGBA8 bytes — see [`Self::from_rgba8`]. Mipmaps are
/// generated on the CPU at upload time using box-filter downsampling.
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: (u32, u32),
    pub format: wgpu::TextureFormat,
    pub mip_levels: u32,
}

impl Texture {
    /// Upload a 2D texture from RGBA8 byte data via the caller's
    /// `Renderer`. Convenience wrapper over
    /// [`Self::from_rgba8_with_device`].
    pub fn from_rgba8(
        renderer: &Renderer,
        label: &str,
        width: u32,
        height: u32,
        rgba: &[u8],
        srgb: bool,
    ) -> Self {
        Self::from_rgba8_with_device(
            &renderer.device,
            &renderer.queue,
            label,
            width,
            height,
            rgba,
            srgb,
        )
    }

    /// Upload a 2D texture from RGBA8 byte data (`width * height * 4`
    /// bytes, row-major, no padding) and generate a full mip chain.
    ///
    /// `srgb = true` selects `Rgba8UnormSrgb`, suitable for albedo / colour
    /// data the shader expects in linear space (the GPU decodes sRGB →
    /// linear on sample). Use `srgb = false` for normal maps,
    /// metallic-roughness packs, and other "data" textures.
    ///
    /// Takes `&wgpu::Device` + `&wgpu::Queue` directly rather than a
    /// `Renderer` because [`AssetServer`](super::AssetServer) background
    /// loaders run on a worker thread and clone the device/queue handles
    /// into their task — `Renderer` is single-owner and lives on the main
    /// thread.
    pub fn from_rgba8_with_device(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        width: u32,
        height: u32,
        rgba: &[u8],
        srgb: bool,
    ) -> Self {
        assert!(width > 0 && height > 0, "{label}: zero-sized texture");
        assert_eq!(
            rgba.len() as u32,
            width * height * 4,
            "{label}: rgba slice has wrong length for {width}x{height}"
        );

        let format = if srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let mip_levels = mip_level_count(width, height);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload level 0 from the caller's bytes, then walk the mip chain
        // downsampling in CPU-side RGBA8.
        let mut level_w = width;
        let mut level_h = height;
        let mut level_data: Vec<u8> = rgba.to_vec();
        for level in 0..mip_levels {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &level_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(level_w * 4),
                    rows_per_image: Some(level_h),
                },
                wgpu::Extent3d {
                    width: level_w,
                    height: level_h,
                    depth_or_array_layers: 1,
                },
            );
            if level + 1 < mip_levels {
                let next_w = (level_w / 2).max(1);
                let next_h = (level_h / 2).max(1);
                level_data = downsample_rgba8(&level_data, level_w, level_h, next_w, next_h);
                level_w = next_w;
                level_h = next_h;
            }
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            ..Default::default()
        });

        Self {
            texture,
            view,
            size: (width, height),
            format,
            mip_levels,
        }
    }

    /// Load a 2D texture from a PNG or JPEG file on disk, decode it to
    /// RGBA8, and upload via [`Self::from_rgba8`] (which generates the mip
    /// chain).
    ///
    /// `color_space` is a required parameter, not inferred. Albedo and
    /// other colour maps want [`TextureColorSpace::Srgb`] — the GPU then
    /// decodes sRGB → linear on every sample, which is what lighting
    /// expects. Normal maps, metallic/roughness packs, height maps, and
    /// other "data" textures want [`TextureColorSpace::Linear`] so their
    /// bytes pass through untouched. Inferring from filename is fragile;
    /// callers commit to the correct interpretation here.
    ///
    /// Synchronous file read; large textures will block the caller.
    /// Suitable for example-scale assets. Streaming through
    /// [`AssetServer`](super::AssetServer) is the right path for shipping
    /// content; this constructor is the eager-load escape hatch.
    pub fn from_path(
        renderer: &Renderer,
        path: impl AsRef<Path>,
        color_space: TextureColorSpace,
    ) -> Result<Self, TextureLoadError> {
        let path = path.as_ref();
        let (width, height, rgba) = decode_rgba8_from_path(path)?;
        let label = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned());
        Ok(Self::from_rgba8(
            renderer,
            &label,
            width,
            height,
            &rgba,
            color_space == TextureColorSpace::Srgb,
        ))
    }

    /// Decode an in-memory PNG/JPEG byte buffer and upload it. Used by
    /// [`AssetServer`](super::AssetServer) loader tasks after a successful
    /// [`Vfs::read`](crate::data::Vfs::read).
    ///
    /// Same `color_space` semantics as [`Self::from_path`] — the caller
    /// commits to sRGB-vs-linear interpretation here, not inferring from
    /// extension or magic bytes.
    pub fn from_png_bytes_with_device(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        bytes: &[u8],
        color_space: TextureColorSpace,
    ) -> Result<Self, TextureLoadError> {
        let (width, height, rgba) = decode_rgba8_from_bytes(bytes)?;
        Ok(Self::from_rgba8_with_device(
            device,
            queue,
            label,
            width,
            height,
            &rgba,
            color_space == TextureColorSpace::Srgb,
        ))
    }
}

/// How the GPU should interpret a texture's pixel data on sample.
///
/// Required by [`Texture::from_path`] — no default, no inference from
/// filename. The wrong choice silently darkens or brightens output.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TextureColorSpace {
    /// Albedo / colour maps. Selects `Rgba8UnormSrgb`; samples decode
    /// sRGB → linear automatically.
    Srgb,
    /// Normal maps, metallic/roughness, height — anything whose bytes
    /// are data, not colour. Selects `Rgba8Unorm`; samples pass through
    /// untouched.
    Linear,
}

/// Failure mode for [`Texture::from_path`]. Wraps the two underlying
/// error sources — filesystem I/O and image decoding — without leaking
/// either type into the public API.
#[derive(Debug)]
pub enum TextureLoadError {
    Io(std::io::Error),
    Decode(image::ImageError),
}

impl std::fmt::Display for TextureLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "texture I/O error: {e}"),
            Self::Decode(e) => write!(f, "texture decode error: {e}"),
        }
    }
}

impl std::error::Error for TextureLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Decode(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for TextureLoadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<image::ImageError> for TextureLoadError {
    fn from(value: image::ImageError) -> Self {
        Self::Decode(value)
    }
}

/// Read a file and decode it to RGBA8. Split out so tests can exercise the
/// decode path without a GPU.
fn decode_rgba8_from_path(path: &Path) -> Result<(u32, u32, Vec<u8>), TextureLoadError> {
    let bytes = std::fs::read(path)?;
    decode_rgba8_from_bytes(&bytes)
}

/// Decode an in-memory image buffer to RGBA8. Whatever the source pixel
/// format is, we force-convert to 8-bit RGBA so [`Texture::from_rgba8`]'s
/// invariants hold.
fn decode_rgba8_from_bytes(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), TextureLoadError> {
    let img = image::load_from_memory(bytes)?;
    let rgba = img.into_rgba8();
    let (w, h) = rgba.dimensions();
    Ok((w, h, rgba.into_raw()))
}

/// Mip levels for a texture of size `width × height`, all the way down to
/// `1×1`. Matches `wgpu` convention.
fn mip_level_count(width: u32, height: u32) -> u32 {
    32 - width.max(height).leading_zeros()
}

/// Box-filter downsample RGBA8 from `(src_w, src_h)` to `(dst_w, dst_h)`.
/// Caller guarantees `dst_w <= src_w` and `dst_h <= src_h` and that each
/// dst dimension is either `src/2` or `1`.
fn downsample_rgba8(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];
    let sx = src_w / dst_w;
    let sy = src_h / dst_h;
    for y in 0..dst_h {
        for x in 0..dst_w {
            // Average over a sx*sy block of source texels.
            let mut acc = [0u32; 4];
            for dy in 0..sy {
                for dx in 0..sx {
                    let sx_idx = x * sx + dx;
                    let sy_idx = y * sy + dy;
                    let i = ((sy_idx * src_w + sx_idx) * 4) as usize;
                    acc[0] += src[i] as u32;
                    acc[1] += src[i + 1] as u32;
                    acc[2] += src[i + 2] as u32;
                    acc[3] += src[i + 3] as u32;
                }
            }
            let n = sx * sy;
            let i = ((y * dst_w + x) * 4) as usize;
            dst[i] = (acc[0] / n) as u8;
            dst[i + 1] = (acc[1] / n) as u8;
            dst[i + 2] = (acc[2] / n) as u8;
            dst[i + 3] = (acc[3] / n) as u8;
        }
    }
    dst
}

/// Closed set of canonical sampler configurations. Adding a new variant is
/// a deliberate code change.
///
/// Naming is `<filter>-<address-mode>`. Mipmaps are always enabled (linear
/// between mip levels for the linear filters, nearest for the nearest one).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SamplerKind {
    /// Trilinear filtering, repeating UVs. The default for tiling surface
    /// textures (terrain, wood, brick).
    LinearRepeat,
    /// Trilinear filtering, clamped UVs. Use for sprites, UI textures, and
    /// any single-image texture where edge bleed is unwanted.
    LinearClamp,
    /// Nearest filtering, clamped UVs. Use for pixel-art / blocky looks.
    NearestClamp,
}

/// Live `wgpu::Sampler` for each [`SamplerKind`]. Build once at
/// `View::init`; share across all materials.
pub struct SamplerRegistry {
    samplers: HashMap<SamplerKind, wgpu::Sampler>,
}

impl SamplerRegistry {
    pub fn new(device: &wgpu::Device) -> Self {
        let mut samplers = HashMap::new();
        samplers.insert(
            SamplerKind::LinearRepeat,
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("currawong sampler LinearRepeat"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
        );
        samplers.insert(
            SamplerKind::LinearClamp,
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("currawong sampler LinearClamp"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
        );
        samplers.insert(
            SamplerKind::NearestClamp,
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("currawong sampler NearestClamp"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            }),
        );
        Self { samplers }
    }

    /// The live sampler for the given kind. Panics if a (newly-added)
    /// variant isn't populated — programming error, not runtime condition.
    pub fn get(&self, kind: SamplerKind) -> &wgpu::Sampler {
        self.samplers
            .get(&kind)
            .unwrap_or_else(|| panic!("SamplerRegistry missing entry for {kind:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_count_powers_of_two() {
        assert_eq!(mip_level_count(1, 1), 1);
        assert_eq!(mip_level_count(2, 2), 2);
        assert_eq!(mip_level_count(4, 4), 3);
        assert_eq!(mip_level_count(256, 256), 9);
        assert_eq!(mip_level_count(1024, 1024), 11);
    }

    #[test]
    fn mip_count_non_square() {
        assert_eq!(mip_level_count(8, 2), 4); // ceil(log2(max(8, 2))) + 1
        assert_eq!(mip_level_count(7, 3), 3); // ceil(log2(7)) + 1
    }

    #[test]
    fn downsample_halves() {
        // 4x4 solid red → 2x2 solid red.
        let src = [255u8, 0, 0, 255].repeat(16);
        let dst = downsample_rgba8(&src, 4, 4, 2, 2);
        assert_eq!(dst.len(), 16);
        for px in dst.chunks(4) {
            assert_eq!(px, &[255, 0, 0, 255]);
        }
    }

    #[test]
    fn decode_png_yields_expected_dimensions_and_mip_count() {
        // Encode a 16x8 RGB PNG in-memory via `image`, decode it back
        // through our loader, and verify the dimensions + the mip count
        // the upload path would compute. Pure CPU — no GPU touched.
        let (src_w, src_h) = (16u32, 8u32);
        let mut src = image::RgbImage::new(src_w, src_h);
        for (x, y, px) in src.enumerate_pixels_mut() {
            *px = image::Rgb([(x * 16) as u8, (y * 32) as u8, 200]);
        }
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(src)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .expect("encode png");

        let (w, h, rgba) = decode_rgba8_from_bytes(&png_bytes).expect("decode png");
        assert_eq!((w, h), (src_w, src_h));
        assert_eq!(rgba.len() as u32, w * h * 4);
        // 16x8 → max dim 16 → 5 mip levels (16, 8, 4, 2, 1).
        assert_eq!(mip_level_count(w, h), 5);
    }

    #[test]
    fn decode_garbage_bytes_returns_decode_error() {
        let err =
            decode_rgba8_from_bytes(b"not a real image").expect_err("garbage must not decode");
        assert!(matches!(err, TextureLoadError::Decode(_)));
    }

    #[test]
    fn from_path_missing_file_returns_io_error() {
        let err = decode_rgba8_from_path(Path::new(
            "/this/path/should/never/exist/currawong-test-fixture.png",
        ))
        .expect_err("missing path must not succeed");
        assert!(matches!(err, TextureLoadError::Io(_)));
    }

    #[test]
    fn downsample_averages() {
        // 2x2 checker (red/black) → 1x1 grey-red.
        // src: R K
        //      K R    in row-major bytes.
        let r = [255u8, 0, 0, 255];
        let k = [0u8, 0, 0, 255];
        let mut src = Vec::new();
        src.extend_from_slice(&r);
        src.extend_from_slice(&k);
        src.extend_from_slice(&k);
        src.extend_from_slice(&r);
        let dst = downsample_rgba8(&src, 2, 2, 1, 1);
        // Average of (255,0,0,255), (0,0,0,255), (0,0,0,255), (255,0,0,255):
        // R = 510/4 = 127, A = 1020/4 = 255.
        assert_eq!(dst, [127, 0, 0, 255]);
    }
}
