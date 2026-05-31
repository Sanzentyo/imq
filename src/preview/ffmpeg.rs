//! ffmpeg thumbnail extraction with runtime hardware-decode probing.

use super::{DecodeMode, PreviewOptions, process_failed, process_start_failed};
use crate::Result;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

static HW_ACCELS: OnceLock<Vec<HwAccel>> = OnceLock::new();

/// ffmpeg hardware acceleration backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwAccel {
    /// macOS VideoToolbox.
    VideoToolbox,
    /// NVIDIA CUDA.
    Cuda,
    /// Linux VAAPI.
    Vaapi,
    /// Intel Quick Sync Video.
    Qsv,
    /// VDPAU.
    Vdpau,
    /// Windows DXVA2.
    Dxva2,
    /// Windows D3D11VA.
    D3d11va,
    /// DRM hardware acceleration.
    Drm,
    /// OpenCL hardware acceleration.
    Opencl,
    /// Vulkan hardware acceleration.
    Vulkan,
}

impl HwAccel {
    /// All ffmpeg backend names `imq` knows how to try.
    pub const fn all() -> &'static [Self] {
        &[
            Self::VideoToolbox,
            Self::Cuda,
            Self::Vaapi,
            Self::Qsv,
            Self::Vdpau,
            Self::Dxva2,
            Self::D3d11va,
            Self::Drm,
            Self::Opencl,
            Self::Vulkan,
        ]
    }

    /// ffmpeg `-hwaccel` value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VideoToolbox => "videotoolbox",
            Self::Cuda => "cuda",
            Self::Vaapi => "vaapi",
            Self::Qsv => "qsv",
            Self::Vdpau => "vdpau",
            Self::Dxva2 => "dxva2",
            Self::D3d11va => "d3d11va",
            Self::Drm => "drm",
            Self::Opencl => "opencl",
            Self::Vulkan => "vulkan",
        }
    }
}

/// Hardware acceleration report for the configured ffmpeg binary.
#[derive(Debug, Clone)]
pub struct HwAccelReport {
    /// Backends reported by `ffmpeg -hwaccels`.
    pub detected: Vec<HwAccel>,
    /// Whether this build forces CPU-only decode.
    pub cpu_only_feature: bool,
}

/// Raw thumbnail frame encoded as PNG.
pub struct ThumbnailFrame {
    /// PNG bytes.
    pub bytes: Vec<u8>,
    /// Decoder description.
    pub source: String,
}

/// Detects ffmpeg hardware acceleration backends once per process.
pub fn detect_hw_accels(ffmpeg: &Path) -> HwAccelReport {
    let cpu_only_feature = cfg!(feature = "cpu-only");
    if cpu_only_feature {
        return HwAccelReport {
            detected: Vec::new(),
            cpu_only_feature,
        };
    }
    let detected = HW_ACCELS
        .get_or_init(|| detect_hw_accels_uncached(ffmpeg))
        .clone();
    HwAccelReport {
        detected,
        cpu_only_feature,
    }
}

/// Extracts a PNG thumbnail from a video.
pub fn thumbnail_png(path: &Path, options: &PreviewOptions) -> Result<ThumbnailFrame> {
    let decode = if cfg!(feature = "cpu-only") {
        DecodeMode::Cpu
    } else {
        options.decode
    };

    match decode {
        DecodeMode::Cpu => run_thumbnail(options, path, None).map(|bytes| ThumbnailFrame {
            bytes,
            source: "ffmpeg cpu".to_string(),
        }),
        DecodeMode::Auto | DecodeMode::Hardware => {
            let report = detect_hw_accels(&options.ffmpeg);
            for accel in report.detected {
                if let Ok(bytes) = run_thumbnail(options, path, Some(accel)) {
                    return Ok(ThumbnailFrame {
                        bytes,
                        source: format!("ffmpeg {}", accel.as_str()),
                    });
                }
            }
            if decode == DecodeMode::Hardware {
                run_thumbnail(options, path, Some(HwAccel::all()[0])).map(|bytes| ThumbnailFrame {
                    bytes,
                    source: "ffmpeg hardware".to_string(),
                })
            } else {
                run_thumbnail(options, path, None).map(|bytes| ThumbnailFrame {
                    bytes,
                    source: "ffmpeg cpu fallback".to_string(),
                })
            }
        }
    }
}

fn detect_hw_accels_uncached(ffmpeg: &Path) -> Vec<HwAccel> {
    let Ok(output) = Command::new(ffmpeg)
        .args(["-hide_banner", "-hwaccels"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&output.stdout).to_ascii_lowercase();
    HwAccel::all()
        .iter()
        .copied()
        .filter(|accel| stdout.lines().any(|line| line.trim() == accel.as_str()))
        .collect()
}

fn run_thumbnail(options: &PreviewOptions, path: &Path, accel: Option<HwAccel>) -> Result<Vec<u8>> {
    let mut command = Command::new(&options.ffmpeg);
    command.args(["-hide_banner", "-loglevel", "error"]);
    if let Some(accel) = accel {
        command.args(["-hwaccel", accel.as_str()]);
    }
    command.args(["-ss", "0"]).arg("-i").arg(path);
    command.args(["-frames:v", "1"]);
    if options.width != u32::MAX && options.height != u32::MAX {
        let filter = format!(
            "scale={}:{}:force_original_aspect_ratio=decrease",
            options.width.max(1),
            options.height.max(1)
        );
        command.args(["-vf", filter.as_str()]);
    }
    command.args(["-f", "image2pipe", "-vcodec", "png", "pipe:1"]);
    let output = command
        .output()
        .map_err(|error| process_start_failed(&options.ffmpeg, "--ffmpeg", error))?;
    if output.status.success() && !output.stdout.is_empty() {
        Ok(output.stdout)
    } else {
        Err(process_failed(
            &options.ffmpeg,
            output.status.to_string(),
            output.stderr,
        ))
    }
}
