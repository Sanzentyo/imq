//! Terminal-friendly still-image and video preview generation.

use crate::{Error, Result};
use image::{DynamicImage, GenericImageView, ImageReader, imageops::FilterType};
use std::io::Cursor;
use std::path::{Path, PathBuf};

mod ffmpeg;
mod terminal;

pub use ffmpeg::{HwAccel, HwAccelReport, detect_hw_accels};
pub use terminal::{DisplayMode, TerminalCapabilities, terminal_capabilities};

/// Hardware decode policy for ffmpeg-backed video previews.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeMode {
    /// Try detected hardware decode backends, then fall back to CPU.
    Auto,
    /// Require a detected hardware decode backend.
    Hardware,
    /// Use CPU decode only.
    Cpu,
}

/// How previews should fit into the requested dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitMode {
    /// Preserve aspect ratio and fit entirely inside the requested dimensions.
    Contain,
    /// Preserve aspect ratio, fill the requested dimensions, and center-crop overflow.
    Cover,
    /// Ignore aspect ratio and stretch to the requested dimensions.
    Stretch,
}

/// Small RGB preview image.
#[derive(Debug, Clone)]
pub struct PreviewImage {
    /// Width in pixels/cells.
    pub width: u32,
    /// Height in pixels/cells.
    pub height: u32,
    /// Natural source width before preview resizing.
    pub source_width: u32,
    /// Natural source height before preview resizing.
    pub source_height: u32,
    /// Row-major RGB pixels.
    pub pixels: Vec<[u8; 3]>,
    /// Decoder path used to produce the preview.
    pub source: String,
}

impl PreviewImage {
    /// Returns a pixel, clamping coordinates into bounds.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        self.pixels[(y * self.width + x) as usize]
    }
}

/// Preview generation options.
#[derive(Debug, Clone)]
pub struct PreviewOptions {
    /// Maximum decoded preview width.
    pub width: u32,
    /// Maximum decoded preview height.
    pub height: u32,
    /// ffmpeg executable for video thumbnails.
    pub ffmpeg: PathBuf,
    /// Video decode mode.
    pub decode: DecodeMode,
    /// Fit policy for resizing.
    pub fit: FitMode,
}

impl Default for PreviewOptions {
    fn default() -> Self {
        Self {
            width: 80,
            height: 40,
            ffmpeg: PathBuf::from("ffmpeg"),
            decode: DecodeMode::Auto,
            fit: FitMode::Contain,
        }
    }
}

/// Generates a preview for an image or video path.
pub fn preview_path(path: &Path, options: &PreviewOptions) -> Result<PreviewImage> {
    if is_video_path(path) {
        preview_video(path, options)
    } else {
        preview_image(path, options)
    }
}

/// Generates a preview for a still image.
pub fn preview_image(path: &Path, options: &PreviewOptions) -> Result<PreviewImage> {
    let image = ImageReader::open(path)?.decode()?;
    Ok(dynamic_to_preview(
        image,
        options.width,
        options.height,
        options.fit,
        "image".to_string(),
    ))
}

/// Generates a preview for a video by extracting a frame with ffmpeg.
pub fn preview_video(path: &Path, options: &PreviewOptions) -> Result<PreviewImage> {
    let frame = ffmpeg::thumbnail_png(path, options)?;
    let image = ImageReader::new(Cursor::new(frame.bytes))
        .with_guessed_format()?
        .decode()?;
    Ok(dynamic_to_preview(
        image,
        options.width,
        options.height,
        options.fit,
        frame.source,
    ))
}

/// Renders a preview as terminal text/escape sequences.
pub fn render_preview(image: &PreviewImage, mode: DisplayMode) -> String {
    match mode {
        DisplayMode::Kitty => terminal::render_kitty(image),
        DisplayMode::Sixel => terminal::render_sixel(image),
        DisplayMode::Ansi => terminal::render_ansi_blocks(image),
        DisplayMode::None => String::new(),
        DisplayMode::Auto => {
            let capabilities = terminal_capabilities();
            if capabilities.kitty {
                terminal::render_kitty(image)
            } else if capabilities.sixel {
                terminal::render_sixel(image)
            } else {
                terminal::render_ansi_blocks(image)
            }
        }
    }
}

/// Returns true for file extensions treated as videos.
pub fn is_video_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "avi" | "m4v" | "mkv" | "mov" | "mp4" | "mpeg" | "mpg" | "webm" | "wmv"
            )
        })
        .unwrap_or(false)
}

fn dynamic_to_preview(
    image: DynamicImage,
    max_width: u32,
    max_height: u32,
    fit: FitMode,
    source: String,
) -> PreviewImage {
    let (width, height) = image.dimensions();
    let max_width = max_width.clamp(1, width.max(1));
    let max_height = max_height.clamp(1, height.max(1));
    let resized = match fit {
        FitMode::Contain => {
            let scale =
                (max_width as f64 / f64::from(width)).min(max_height as f64 / f64::from(height));
            let target_width = (f64::from(width) * scale).round().max(1.0) as u32;
            let target_height = (f64::from(height) * scale).round().max(1.0) as u32;
            image.resize_exact(target_width, target_height, FilterType::Triangle)
        }
        FitMode::Cover => {
            let scale =
                (max_width as f64 / f64::from(width)).max(max_height as f64 / f64::from(height));
            let resized_width = (f64::from(width) * scale).round().max(1.0) as u32;
            let resized_height = (f64::from(height) * scale).round().max(1.0) as u32;
            let resized = image.resize_exact(resized_width, resized_height, FilterType::Triangle);
            let crop_x = resized_width.saturating_sub(max_width) / 2;
            let crop_y = resized_height.saturating_sub(max_height) / 2;
            resized.crop_imm(crop_x, crop_y, max_width, max_height)
        }
        FitMode::Stretch => image.resize_exact(max_width, max_height, FilterType::Triangle),
    };
    let resized = resized.to_rgb8();
    let target_width = resized.width();
    let target_height = resized.height();
    let pixels = resized.pixels().map(|p| [p[0], p[1], p[2]]).collect();
    PreviewImage {
        width: target_width,
        height: target_height,
        source_width: width,
        source_height: height,
        pixels,
        source,
    }
}

fn process_failed(program: &Path, status: String, stderr: Vec<u8>) -> Error {
    Error::ProcessFailed {
        program: program.display().to_string(),
        status,
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn preview_tracks_source_dimensions() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(8, 6, Rgb([255, 0, 0])));
        let preview = dynamic_to_preview(image, 4, 4, FitMode::Contain, "test".to_string());
        assert_eq!(preview.source_width, 8);
        assert_eq!(preview.source_height, 6);
        assert_eq!((preview.width, preview.height), (4, 3));
    }

    #[test]
    fn preview_does_not_upscale_past_source() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(8, 6, Rgb([255, 0, 0])));
        let preview = dynamic_to_preview(
            image,
            u32::MAX,
            u32::MAX,
            FitMode::Contain,
            "test".to_string(),
        );
        assert_eq!((preview.width, preview.height), (8, 6));
        assert_eq!((preview.source_width, preview.source_height), (8, 6));
    }
}
