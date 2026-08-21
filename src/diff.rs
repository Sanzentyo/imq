//! Diff-image and heatmap generation for full-reference comparisons.
//!
//! The functions here are Sans-I/O at the core: they return RGBA8 bytes that a
//! caller can save, stream, or embed. When the `image-codecs` feature is enabled,
//! [`RgbaImageData::save_png`] provides a small PNG-writing convenience wrapper.

use crate::frame::{FrameView, PixelFormat, Validated};
use crate::{Error, Result};
#[cfg(feature = "image-codecs")]
use std::path::Path;

/// RGBA8 image data returned by diff/heatmap helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RgbaImageData {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Tight RGBA8 pixels in row-major order.
    pub data: Vec<u8>,
}

impl RgbaImageData {
    /// Creates a tight RGBA8 image buffer.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let expected = usize::try_from(u64::from(width) * u64::from(height) * 4)
            .map_err(|_| Error::invalid_frame("RGBA image byte count overflows usize"))?;
        if data.len() != expected {
            return Err(Error::invalid_frame(format!(
                "RGBA image has {} bytes, expected {expected}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Saves this RGBA image as a PNG through the `image` crate.
    #[cfg(feature = "image-codecs")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image-codecs")))]
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<()> {
        let image = image::RgbaImage::from_raw(self.width, self.height, self.data.clone())
            .ok_or_else(|| {
                Error::invalid_frame("RGBA image byte count does not match dimensions")
            })?;
        image.save(path)?;
        Ok(())
    }
}

/// Visual encoding used by [`diff_image`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DiffImageMode {
    /// Encode absolute per-channel RGB differences directly.
    AbsoluteRgb,
    /// Encode a scalar difference magnitude as a blue/cyan/yellow/red heatmap.
    Heatmap,
    /// Encode signed luma difference: red means distorted is brighter, blue means darker.
    SignedLuma,
}

/// Options for diff and heatmap generation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiffImageOptions {
    /// Visual encoding mode.
    pub mode: DiffImageMode,
    /// Multiplier applied before converting normalized differences to 8-bit colors.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub scale: f64,
    /// Whether alpha differences should affect the output alpha channel when both frames contain alpha.
    pub include_alpha: bool,
}

impl Default for DiffImageOptions {
    fn default() -> Self {
        Self {
            mode: DiffImageMode::Heatmap,
            scale: 4.0,
            include_alpha: false,
        }
    }
}

/// Builds an RGBA8 diff image from two validated frames.
pub fn diff_image(
    reference: &FrameView<'_, Validated>,
    distorted: &FrameView<'_, Validated>,
    options: DiffImageOptions,
) -> Result<RgbaImageData> {
    if reference.dimensions() != distorted.dimensions() {
        return Err(Error::incompatible(format!(
            "dimensions differ: {:?} vs {:?}",
            reference.dimensions(),
            distorted.dimensions()
        )));
    }
    let dims = reference.dimensions();
    let (width, height) = dims.as_usize()?;
    let mut out = vec![
        0u8;
        dims.pixels()?.checked_mul(4).ok_or_else(|| {
            Error::invalid_frame("diff output byte count overflows usize")
        })?
    ];

    for y in 0..height {
        for x in 0..width {
            let a = read_display_rgba(reference, x, y)?;
            let b = read_display_rgba(distorted, x, y)?;
            let dst = (y * width + x) * 4;
            let encoded = encode_diff_pixel(a, b, options);
            out[dst..dst + 4].copy_from_slice(&encoded);
        }
    }

    RgbaImageData::new(dims.width, dims.height, out)
}

fn encode_diff_pixel(
    reference: [f64; 4],
    distorted: [f64; 4],
    options: DiffImageOptions,
) -> [u8; 4] {
    match options.mode {
        DiffImageMode::AbsoluteRgb => {
            let r = to_u8((reference[0] - distorted[0]).abs() * options.scale);
            let g = to_u8((reference[1] - distorted[1]).abs() * options.scale);
            let b = to_u8((reference[2] - distorted[2]).abs() * options.scale);
            let a = if options.include_alpha {
                to_u8((reference[3] - distorted[3]).abs() * options.scale)
            } else {
                255
            };
            [r, g, b, a]
        }
        DiffImageMode::Heatmap => {
            let dr = reference[0] - distorted[0];
            let dg = reference[1] - distorted[1];
            let db = reference[2] - distorted[2];
            let magnitude = ((dr * dr + dg * dg + db * db) / 3.0).sqrt() * options.scale;
            heatmap_color(magnitude)
        }
        DiffImageMode::SignedLuma => {
            let ref_luma = 0.2126 * reference[0] + 0.7152 * reference[1] + 0.0722 * reference[2];
            let dist_luma = 0.2126 * distorted[0] + 0.7152 * distorted[1] + 0.0722 * distorted[2];
            let signed = (dist_luma - ref_luma) * options.scale;
            if signed >= 0.0 {
                [to_u8(signed), 0, 0, 255]
            } else {
                [0, 0, to_u8(-signed), 255]
            }
        }
    }
}

fn heatmap_color(value: f64) -> [u8; 4] {
    let t = value.clamp(0.0, 1.0);
    if t < 0.25 {
        let k = t / 0.25;
        [0, to_u8(k), 255, 255]
    } else if t < 0.5 {
        let k = (t - 0.25) / 0.25;
        [0, 255, to_u8(1.0 - k), 255]
    } else if t < 0.75 {
        let k = (t - 0.5) / 0.25;
        [to_u8(k), 255, 0, 255]
    } else {
        let k = (t - 0.75) / 0.25;
        [255, to_u8(1.0 - k), 0, 255]
    }
}

fn read_display_rgba(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<[f64; 4]> {
    let format = frame.pixel_format();
    let plane = frame.plane(0)?;
    let bytes_per_pixel = format.bytes_per_pixel().ok_or_else(|| {
        Error::unsupported("diff_image currently requires packed RGB/RGBA/luma inputs")
    })?;
    let width = usize::try_from(frame.dimensions().width)
        .map_err(|_| Error::invalid_frame("width overflows usize"))?;
    let row_bytes = width
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| Error::invalid_frame("row byte count overflows usize"))?;
    let row = plane.row(y, row_bytes)?;
    let offset = x
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| Error::invalid_frame("pixel offset overflows usize"))?;
    let max = format.max_code_value();

    let rgba = match format {
        PixelFormat::Luma8 => {
            let l = f64::from(row[offset]) / max;
            [l, l, l, 1.0]
        }
        PixelFormat::Rgb8 => [
            f64::from(row[offset]) / max,
            f64::from(row[offset + 1]) / max,
            f64::from(row[offset + 2]) / max,
            1.0,
        ],
        PixelFormat::Rgba8 => [
            f64::from(row[offset]) / max,
            f64::from(row[offset + 1]) / max,
            f64::from(row[offset + 2]) / max,
            f64::from(row[offset + 3]) / max,
        ],
        PixelFormat::Bgr8 => [
            f64::from(row[offset + 2]) / max,
            f64::from(row[offset + 1]) / max,
            f64::from(row[offset]) / max,
            1.0,
        ],
        PixelFormat::Bgra8 => [
            f64::from(row[offset + 2]) / max,
            f64::from(row[offset + 1]) / max,
            f64::from(row[offset]) / max,
            f64::from(row[offset + 3]) / max,
        ],
        PixelFormat::Luma16Le => {
            let l = f64::from(read_u16_le(row, offset)?) / max;
            [l, l, l, 1.0]
        }
        PixelFormat::Rgb16Le => [
            f64::from(read_u16_le(row, offset)?) / max,
            f64::from(read_u16_le(row, offset + 2)?) / max,
            f64::from(read_u16_le(row, offset + 4)?) / max,
            1.0,
        ],
        PixelFormat::Rgba16Le => [
            f64::from(read_u16_le(row, offset)?) / max,
            f64::from(read_u16_le(row, offset + 2)?) / max,
            f64::from(read_u16_le(row, offset + 4)?) / max,
            f64::from(read_u16_le(row, offset + 6)?) / max,
        ],
        PixelFormat::RgbF32 => [
            f64::from(read_f32_ne(row, offset)?),
            f64::from(read_f32_ne(row, offset + 4)?),
            f64::from(read_f32_ne(row, offset + 8)?),
            1.0,
        ],
        PixelFormat::RgbaF32 => [
            f64::from(read_f32_ne(row, offset)?),
            f64::from(read_f32_ne(row, offset + 4)?),
            f64::from(read_f32_ne(row, offset + 8)?),
            f64::from(read_f32_ne(row, offset + 12)?),
        ],
        _ => {
            return Err(Error::unsupported(
                "diff_image currently does not convert planar YUV inputs; compare luma metrics or convert to RGB first",
            ));
        }
    };
    Ok(rgba.map(|value| value.clamp(0.0, 1.0)))
}

fn read_u16_le(row: &[u8], offset: usize) -> Result<u16> {
    let bytes = row
        .get(offset..offset + 2)
        .ok_or_else(|| Error::invalid_frame("u16 sample out of bounds"))?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_f32_ne(row: &[u8], offset: usize) -> Result<f32> {
    let bytes = row
        .get(offset..offset + 4)
        .ok_or_else(|| Error::invalid_frame("f32 sample out of bounds"))?;
    Ok(f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn to_u8(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FrameOwned, PixelFormat};

    #[test]
    fn absolute_diff_reports_channel_delta() {
        let reference =
            FrameOwned::packed_tight(vec![0, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let distorted =
            FrameOwned::packed_tight(vec![255, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let diff = diff_image(
            &reference.as_view(),
            &distorted.as_view(),
            DiffImageOptions {
                mode: DiffImageMode::AbsoluteRgb,
                scale: 1.0,
                include_alpha: false,
            },
        )
        .unwrap();
        assert_eq!(diff.data, vec![255, 0, 0, 255]);
    }
}
