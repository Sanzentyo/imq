//! Adapters for the `image` crate.

use crate::Result;
use crate::frame::{FrameOwned, PixelFormat};
use image::{DynamicImage, GenericImageView, ImageReader};
use std::io::Cursor;
use std::path::Path;

/// Converts an `image::DynamicImage` to owned tightly packed RGBA8.
///
/// This intentionally normalizes to RGBA8 because it is the most convenient
/// common interchange format for image files, ffmpeg rawvideo, and wgpu kernels.
pub fn from_dynamic_image_rgba8(img: &DynamicImage) -> Result<FrameOwned> {
    let (w, h) = img.dimensions();
    let rgba = img.to_rgba8();
    FrameOwned::packed_tight(rgba.into_raw(), w, h, PixelFormat::Rgba8)
}

/// Converts an `image::RgbaImage` to owned tightly packed RGBA8 without another copy.
pub fn from_rgba_image(img: image::RgbaImage) -> Result<FrameOwned> {
    let (w, h) = img.dimensions();
    FrameOwned::packed_tight(img.into_raw(), w, h, PixelFormat::Rgba8)
}

/// Converts an `image::RgbImage` to owned tightly packed RGB8 without another copy.
pub fn from_rgb_image(img: image::RgbImage) -> Result<FrameOwned> {
    let (w, h) = img.dimensions();
    FrameOwned::packed_tight(img.into_raw(), w, h, PixelFormat::Rgb8)
}

/// Decodes an image file via the `image` crate and returns RGBA8.
pub fn load_image_path(path: impl AsRef<Path>) -> Result<FrameOwned> {
    let img = ImageReader::open(path)?.with_guessed_format()?.decode()?;
    from_dynamic_image_rgba8(&img)
}

/// Decodes image bytes via the `image` crate and returns RGBA8.
pub fn decode_image_bytes(bytes: &[u8]) -> Result<FrameOwned> {
    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()?;
    from_dynamic_image_rgba8(&img)
}

/// Lists formats enabled in the `image` dependency's default feature set.
pub fn enabled_format_hint() -> &'static [&'static str] {
    &[
        "AVIF", "BMP", "DDS", "EXR", "Farbfeld", "GIF", "HDR", "ICO", "JPEG", "PNG", "PNM", "QOI",
        "TGA", "TIFF", "WebP",
    ]
}
