//! Terminal-friendly still-image and video preview generation.

use crate::{Error, Result};
use fast_image_resize::{ResizeOptions, Resizer};
use image::{DynamicImage, GenericImageView, ImageReader, imageops::FilterType};
use std::io::Cursor;
use std::path::{Path, PathBuf};

mod ffmpeg;
mod terminal;

pub use ffmpeg::{HwAccel, HwAccelReport, detect_hw_accels};
pub use terminal::{DisplayMode, TerminalCapabilities, auto_display_mode, terminal_capabilities};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    if !path.exists() {
        return Err(Error::input_not_found(path.display().to_string()));
    }
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
        DisplayMode::Iterm2 => terminal::render_iterm2(image),
        DisplayMode::Ansi => terminal::render_ansi_blocks(image),
        DisplayMode::None => String::new(),
        DisplayMode::Auto => render_preview(image, terminal::auto_display_mode()),
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
    let max_width = if max_width == u32::MAX {
        width.max(1)
    } else {
        max_width.max(1)
    };
    let max_height = if max_height == u32::MAX {
        height.max(1)
    } else {
        max_height.max(1)
    };
    let resized = match fit {
        FitMode::Contain => {
            let scale =
                (max_width as f64 / f64::from(width)).min(max_height as f64 / f64::from(height));
            let target_width = (f64::from(width) * scale).round().max(1.0) as u32;
            let target_height = (f64::from(height) * scale).round().max(1.0) as u32;
            fast_resize_rgb(&image, target_width, target_height)
        }
        FitMode::Cover => {
            let scale =
                (max_width as f64 / f64::from(width)).max(max_height as f64 / f64::from(height));
            let resized_width = (f64::from(width) * scale).round().max(1.0) as u32;
            let resized_height = (f64::from(height) * scale).round().max(1.0) as u32;
            let resized = fast_resize_rgb(&image, resized_width, resized_height);
            let crop_x = resized_width.saturating_sub(max_width) / 2;
            let crop_y = resized_height.saturating_sub(max_height) / 2;
            resized.crop_imm(crop_x, crop_y, max_width, max_height)
        }
        FitMode::Stretch => fast_resize_rgb(&image, max_width, max_height),
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

fn fast_resize_rgb(image: &DynamicImage, target_width: u32, target_height: u32) -> DynamicImage {
    let source_width = image.width();
    let source_height = image.height();
    if target_width >= source_width && target_height >= source_height {
        return image.resize_exact(target_width, target_height, FilterType::Nearest);
    }

    let shrink_width = target_width.min(source_width);
    let shrink_height = target_height.min(source_height);
    let shrunk = if shrink_width < source_width || shrink_height < source_height {
        fir_resize_rgb(image, shrink_width, shrink_height)
    } else {
        image.clone()
    };
    if target_width > shrink_width || target_height > shrink_height {
        return shrunk.resize_exact(target_width, target_height, FilterType::Nearest);
    }
    shrunk
}

fn fir_resize_rgb(image: &DynamicImage, target_width: u32, target_height: u32) -> DynamicImage {
    let src = DynamicImage::ImageRgb8(image.to_rgb8());
    let mut dst = DynamicImage::new_rgb8(target_width, target_height);
    let mut resizer = Resizer::new();
    let options = ResizeOptions::new();
    match resizer.resize(&src, &mut dst, &options) {
        Ok(()) => dst,
        Err(_) => image.resize_exact(target_width, target_height, FilterType::Triangle),
    }
}

fn process_failed(program: &Path, status: String, stderr: Vec<u8>) -> Error {
    Error::ProcessFailed {
        program: program.display().to_string(),
        status,
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
    }
}

fn process_start_failed(program: &Path, option: &str, source: std::io::Error) -> Error {
    Error::external_tool_start_failed(program.display().to_string(), option, source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use image::{Rgb, RgbImage};
    use std::fs;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("imq-preview-test-{}-{name}", std::process::id()))
    }

    #[test]
    fn preview_tracks_source_dimensions() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(8, 6, Rgb([255, 0, 0])));
        let preview = dynamic_to_preview(image, 4, 4, FitMode::Contain, "test".to_string());
        assert_eq!(preview.source_width, 8);
        assert_eq!(preview.source_height, 6);
        assert_eq!((preview.width, preview.height), (4, 3));
    }

    #[test]
    fn preview_upscales_with_nearest_neighbor() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(8, 6, Rgb([255, 0, 0])));
        let preview = dynamic_to_preview(image, 16, 16, FitMode::Contain, "test".to_string());
        assert_eq!((preview.width, preview.height), (16, 12));
        assert_eq!((preview.source_width, preview.source_height), (8, 6));
        assert!(preview.pixels.iter().all(|pixel| *pixel == [255, 0, 0]));
    }

    #[test]
    fn preview_reports_missing_input_before_ffmpeg() {
        let path = temp_path("missing.mp4");
        let err = preview_path(&path, &PreviewOptions::default()).unwrap_err();

        assert!(matches!(err, Error::InputNotFound { .. }));
        assert!(err.to_string().contains("input file not found"));
    }

    #[test]
    fn video_preview_reports_missing_ffmpeg_separately() {
        let path = temp_path("empty.mp4");
        fs::write(&path, []).unwrap();
        let options = PreviewOptions {
            ffmpeg: temp_path("missing-ffmpeg"),
            ..Default::default()
        };

        let err = preview_path(&path, &options).unwrap_err();
        let _ = fs::remove_file(path);

        assert!(matches!(err, Error::ExternalToolNotFound { .. }));
        assert!(err.to_string().contains("external tool"));
        assert!(err.to_string().contains("--ffmpeg"));
    }
}
