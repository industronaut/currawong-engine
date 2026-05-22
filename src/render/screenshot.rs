//! Engine-driven F12 screenshot capture (#20).
//!
//! Two-step shape, mirroring the picking-buffer readback in
//! [`super::picking_buffer`]: while encoding the frame, [`ScreenshotRequest::record`]
//! allocates a `MAP_READ` staging buffer and records a `copy_texture_to_buffer`
//! from the swapchain image into it. After the frame's encoder has been
//! submitted, [`ScreenshotRequest::save_blocking`] calls
//! `device.poll(Wait)` + `map_async` to read the bytes, strips the
//! 256-byte row-stride padding wgpu requires, swaps channels for BGRA
//! formats, and writes a PNG.
//!
//! sRGB note: the swapchain format is one of the `*8UnormSrgb` variants
//! (picked in [`super::Renderer::new`]). The raw storage bytes are the
//! sRGB-encoded values displayed to the screen — writing them straight to
//! an RGBA8 PNG round-trips correctly because PNG viewers treat unsigned
//! 8-bit channels as sRGB by default.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Row-stride alignment wgpu requires for `copy_texture_to_buffer`.
const ROW_STRIDE_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// One in-flight screenshot capture. Lives from `record` (inside frame
/// encoding) to `save_blocking` (after frame submit) and is then dropped.
pub(super) struct ScreenshotRequest {
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    format: wgpu::TextureFormat,
}

impl ScreenshotRequest {
    /// Allocate a padded readback buffer and record a `copy_texture_to_buffer`
    /// of the entire `surface_texture` into it. The caller submits the
    /// `encoder` and then calls [`Self::save_blocking`].
    pub(super) fn record(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        surface_texture: &wgpu::Texture,
        format: wgpu::TextureFormat,
    ) -> Self {
        let size = surface_texture.size();
        let width = size.width;
        let height = size.height;
        let padded_bytes_per_row = align_up(width * BYTES_PER_PIXEL, ROW_STRIDE_ALIGNMENT);
        let buffer_size = padded_bytes_per_row as u64 * height as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("currawong screenshot readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: surface_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        Self {
            buffer,
            width,
            height,
            padded_bytes_per_row,
            format,
        }
    }

    /// Wait for the GPU copy to finish, strip the row padding, normalise the
    /// channel order to RGBA, and write `<dir>/<timestamp>.png`. Returns the
    /// absolute path on success.
    ///
    /// Must be called after the encoder that recorded the copy has been
    /// submitted; the `device.poll(Wait)` here drives the `map_async`
    /// callback to completion.
    pub(super) fn save_blocking(self, device: &wgpu::Device, dir: &Path) -> Result<PathBuf, Error> {
        let Self {
            buffer,
            width,
            height,
            padded_bytes_per_row,
            format,
        } = self;

        let (tx, rx) = mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Poll(format!("{e:?}")))?;
        rx.recv()
            .map_err(|_| Error::MapDropped)?
            .map_err(|e| Error::MapAsync(format!("{e:?}")))?;

        let unpadded_bytes_per_row = width * BYTES_PER_PIXEL;
        let mut pixels = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
        {
            let view = buffer.slice(..).get_mapped_range();
            for row in 0..height {
                let start = (row * padded_bytes_per_row) as usize;
                let end = start + unpadded_bytes_per_row as usize;
                pixels.extend_from_slice(&view[start..end]);
            }
        }
        buffer.unmap();

        if is_bgra(format) {
            for px in pixels.chunks_exact_mut(BYTES_PER_PIXEL as usize) {
                px.swap(0, 2);
            }
        }

        std::fs::create_dir_all(dir).map_err(Error::Io)?;
        let filename = format!("{}.png", timestamp_filename(SystemTime::now()));
        let path = dir.join(filename);
        let image =
            image::RgbaImage::from_raw(width, height, pixels).ok_or(Error::PixelLengthMismatch)?;
        image
            .save_with_format(&path, image::ImageFormat::Png)
            .map_err(Error::Encode)?;

        let absolute = std::fs::canonicalize(&path).unwrap_or(path);
        Ok(absolute)
    }
}

/// All four bytes-per-pixel surface formats wgpu can hand us are
/// `*8UnormSrgb`, which is what [`super::Renderer::new`] picks.
const BYTES_PER_PIXEL: u32 = 4;

fn is_bgra(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    )
}

fn align_up(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

/// Format a `SystemTime` as `YYYYMMDD-HHMMSS-mmm` in UTC. Manual conversion
/// avoids pulling in a date crate for one filename; the algorithm for
/// days-since-epoch → civil date is Howard Hinnant's.
fn timestamp_filename(now: SystemTime) -> String {
    let dur = now.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
    let secs = dur.as_secs();
    let ms = dur.subsec_millis();

    let days = (secs / 86_400) as i64;
    let secs_in_day = secs % 86_400;
    let hour = (secs_in_day / 3600) as u32;
    let minute = ((secs_in_day % 3600) / 60) as u32;
    let second = (secs_in_day % 60) as u32;

    let (year, month, day) = civil_from_days(days);
    format!("{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}-{ms:03}")
}

/// Convert days-since-1970-01-01 to a `(year, month, day)` civil date in
/// the proleptic Gregorian calendar. From Howard Hinnant's `date.h`
/// "civil_from_days" — verified at the boundary 1970-01-01 → (1970,1,1)
/// in the test below.
fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32)
}

#[derive(Debug)]
pub(super) enum Error {
    Io(std::io::Error),
    Poll(String),
    MapDropped,
    MapAsync(String),
    PixelLengthMismatch,
    Encode(image::ImageError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Poll(s) => write!(f, "device poll failed: {s}"),
            Self::MapDropped => write!(f, "map_async callback was dropped"),
            Self::MapAsync(s) => write!(f, "map_async failed: {s}"),
            Self::PixelLengthMismatch => write!(f, "pixel buffer length did not match dimensions"),
            Self::Encode(e) => write!(f, "png encode failed: {e}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_rounds_to_alignment() {
        assert_eq!(align_up(3200, 256), 3328);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
    }

    #[test]
    fn civil_from_days_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2000-01-01 is 30 years and 7 leap days after 1970-01-01.
        assert_eq!(civil_from_days(10_957), (2000, 1, 1));
        // 2024-02-29 — a leap day.
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
    }

    #[test]
    fn timestamp_filename_at_epoch() {
        let name = timestamp_filename(UNIX_EPOCH);
        assert_eq!(name, "19700101-000000-000");
    }

    #[test]
    fn timestamp_filename_known_moment() {
        // 2024-02-29 12:34:56.789 UTC = 1_709_210_096 + 0.789s since epoch.
        let when = UNIX_EPOCH + Duration::new(1_709_210_096, 789_000_000);
        assert_eq!(timestamp_filename(when), "20240229-123456-789");
    }
}
