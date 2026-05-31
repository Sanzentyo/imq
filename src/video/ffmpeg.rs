//! External ffmpeg adapter.
//!
//! This module is intentionally not part of the Sans-I/O core. It shells out to
//! `ffprobe` for metadata and `ffmpeg` for raw RGBA frames on stdout, then feeds
//! those frames into the same zero-copy/owned core types as still images.

use crate::frame::{Dimensions, FrameOwned, PixelFormat};
use crate::metrics::{MetricOutput, MetricSet};
use crate::report::{FrameReport, VideoReport};
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

/// Paths and flags used for ffmpeg/ffprobe.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FfmpegOptions {
    /// `ffmpeg` executable path.
    pub ffmpeg: PathBuf,
    /// `ffprobe` executable path.
    pub ffprobe: PathBuf,
    /// Video stream index.
    pub stream_index: usize,
    /// Optional target dimensions; ffmpeg will scale both videos if set.
    pub scale: Option<Dimensions>,
    /// Additional args inserted before `-i`.
    pub input_args: Vec<String>,
}

impl Default for FfmpegOptions {
    fn default() -> Self {
        Self {
            ffmpeg: PathBuf::from("ffmpeg"),
            ffprobe: PathBuf::from("ffprobe"),
            stream_index: 0,
            scale: None,
            input_args: Vec::new(),
        }
    }
}

/// Basic probed video metadata.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VideoInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Average frame rate, if ffprobe reports it.
    pub avg_frame_rate: Option<f64>,
    /// Number of frames, if container metadata reports it.
    pub nb_frames: Option<u64>,
    /// Codec name, if known.
    pub codec_name: Option<String>,
    /// Duration in seconds, if known.
    pub duration_seconds: Option<f64>,
}

impl VideoInfo {
    /// Dimensions as `Dimensions`.
    pub fn dimensions(&self) -> Result<Dimensions> {
        Dimensions::new(self.width, self.height)
    }
}

/// Video comparison options.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VideoCompareOptions {
    /// Compare every Nth decoded frame. `1` compares every frame.
    pub every: u64,
    /// Maximum number of frame pairs to compare.
    pub max_frames: Option<u64>,
}

impl Default for VideoCompareOptions {
    fn default() -> Self {
        Self {
            every: 1,
            max_frames: None,
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct ProbeRoot {
    streams: Vec<ProbeStream>,
}

#[derive(Debug, serde::Deserialize)]
struct ProbeStream {
    width: Option<u32>,
    height: Option<u32>,
    avg_frame_rate: Option<String>,
    nb_frames: Option<String>,
    codec_name: Option<String>,
    duration: Option<String>,
}

/// Probes a video using ffprobe.
pub fn probe_video(path: impl AsRef<Path>, options: &FfmpegOptions) -> Result<VideoInfo> {
    let path = path.as_ref();
    ensure_input_exists(path)?;
    let output = Command::new(&options.ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            &format!("v:{}", options.stream_index),
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,codec_name,duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(|error| process_start_failed(&options.ffprobe, "--ffprobe", error))?;

    if !output.status.success() {
        return Err(Error::ProcessFailed {
            program: options.ffprobe.display().to_string(),
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    let root: ProbeRoot = serde_json::from_slice(&output.stdout)?;
    let stream = root
        .streams
        .into_iter()
        .next()
        .ok_or_else(|| Error::invalid_frame("ffprobe did not return a video stream"))?;

    Ok(VideoInfo {
        width: stream
            .width
            .ok_or_else(|| Error::invalid_frame("ffprobe stream has no width"))?,
        height: stream
            .height
            .ok_or_else(|| Error::invalid_frame("ffprobe stream has no height"))?,
        avg_frame_rate: stream.avg_frame_rate.as_deref().and_then(parse_rational),
        nb_frames: stream.nb_frames.as_deref().and_then(|s| s.parse().ok()),
        codec_name: stream.codec_name,
        duration_seconds: stream.duration.as_deref().and_then(|s| s.parse().ok()),
    })
}

fn parse_rational(s: &str) -> Option<f64> {
    if let Some((n, d)) = s.split_once('/') {
        let n: f64 = n.parse().ok()?;
        let d: f64 = d.parse().ok()?;
        if d == 0.0 { None } else { Some(n / d) }
    } else {
        s.parse().ok()
    }
}

/// Iterator over decoded RGBA8 frames from ffmpeg stdout.
pub struct FfmpegFrameIter {
    child: Child,
    stdout: ChildStdout,
    dims: Dimensions,
    frame_bytes: usize,
    next_index: u64,
    done: bool,
}

impl FfmpegFrameIter {
    /// Spawns ffmpeg and starts reading raw RGBA frames.
    pub fn spawn(path: impl AsRef<Path>, options: &FfmpegOptions) -> Result<Self> {
        let path = path.as_ref();
        let probed = probe_video(path, options)?;
        let dims = options.scale.unwrap_or(probed.dimensions()?);
        let frame_bytes = dims
            .pixels()?
            .checked_mul(4)
            .ok_or_else(|| Error::invalid_frame("RGBA frame byte size overflows usize"))?;

        let mut command = Command::new(&options.ffmpeg);
        command.args(["-hide_banner", "-loglevel", "error"]);
        command.args(&options.input_args);
        command.arg("-i").arg(path);
        command.args(["-map", &format!("0:v:{}", options.stream_index)]);
        if let Some(scale) = options.scale {
            command.args(["-vf", &format!("scale={}:{}", scale.width, scale.height)]);
        }
        command.args(["-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"]);
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        let mut child = command
            .spawn()
            .map_err(|error| process_start_failed(&options.ffmpeg, "--ffmpeg", error))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::invalid_frame("failed to capture ffmpeg stdout"))?;
        Ok(Self {
            child,
            stdout,
            dims,
            frame_bytes,
            next_index: 0,
            done: false,
        })
    }

    /// Dimensions of yielded frames.
    pub fn dimensions(&self) -> Dimensions {
        self.dims
    }

    /// Number of bytes in one RGBA frame.
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }
}

impl Iterator for FfmpegFrameIter {
    type Item = Result<(u64, FrameOwned)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let mut buf = vec![0u8; self.frame_bytes];
        match self.stdout.read_exact(&mut buf) {
            Ok(()) => {
                let idx = self.next_index;
                self.next_index += 1;
                Some(
                    FrameOwned::packed_tight(
                        buf,
                        self.dims.width,
                        self.dims.height,
                        PixelFormat::Rgba8,
                    )
                    .map(|f| (idx, f)),
                )
            }
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => {
                self.done = true;
                match self.child.wait() {
                    Ok(status) if status.success() => None,
                    Ok(status) => Some(Err(Error::ProcessFailed {
                        program: "ffmpeg".to_string(),
                        status: status.to_string(),
                        stderr: "ffmpeg ended before a complete raw frame was available"
                            .to_string(),
                    })),
                    Err(e) => Some(Err(Error::Io(e))),
                }
            }
            Err(err) => {
                self.done = true;
                Some(Err(Error::Io(err)))
            }
        }
    }
}

/// Decodes a single zero-based frame into RGBA8.
pub fn decode_single_frame(
    path: impl AsRef<Path>,
    frame_index: u64,
    options: &FfmpegOptions,
) -> Result<FrameOwned> {
    let path = path.as_ref();
    let probed = probe_video(path, options)?;
    let dims = options.scale.unwrap_or(probed.dimensions()?);
    let frame_bytes = dims
        .pixels()?
        .checked_mul(4)
        .ok_or_else(|| Error::invalid_frame("frame byte size overflow"))?;
    let mut command = Command::new(&options.ffmpeg);
    command.args(["-hide_banner", "-loglevel", "error"]);
    command.args(&options.input_args);
    command.arg("-i").arg(path);
    command.args(["-map", &format!("0:v:{}", options.stream_index)]);

    let mut filters = format!("select=eq(n\\,{frame_index})");
    if let Some(scale) = options.scale {
        filters.push_str(&format!(",scale={}:{}", scale.width, scale.height));
    }
    command.args([
        "-vf",
        &filters,
        "-vsync",
        "0",
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgba",
        "-f",
        "rawvideo",
        "pipe:1",
    ]);
    let output = command
        .output()
        .map_err(|error| process_start_failed(&options.ffmpeg, "--ffmpeg", error))?;
    if !output.status.success() {
        return Err(Error::ProcessFailed {
            program: options.ffmpeg.display().to_string(),
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    if output.stdout.len() != frame_bytes {
        return Err(Error::invalid_frame(format!(
            "expected {frame_bytes} bytes for one frame, got {}",
            output.stdout.len()
        )));
    }
    FrameOwned::packed_tight(output.stdout, dims.width, dims.height, PixelFormat::Rgba8)
}

fn ensure_input_exists(path: &Path) -> Result<()> {
    if path.exists() {
        Ok(())
    } else {
        Err(Error::input_not_found(path.display().to_string()))
    }
}

fn process_start_failed(program: &Path, option: &str, source: io::Error) -> Error {
    Error::external_tool_start_failed(program.display().to_string(), option, source)
}

/// Compares two videos by decoding raw RGBA8 frames and feeding them to the core metrics.
pub fn compare_videos(
    reference: impl AsRef<Path>,
    distorted: impl AsRef<Path>,
    ffmpeg: &FfmpegOptions,
    compare: &VideoCompareOptions,
    metrics: &MetricSet,
) -> Result<VideoReport> {
    let reference_path = reference.as_ref();
    let distorted_path = distorted.as_ref();
    let mut ref_iter = FfmpegFrameIter::spawn(reference_path, ffmpeg)?;
    let mut dist_iter = FfmpegFrameIter::spawn(distorted_path, ffmpeg)?;
    if ref_iter.dimensions() != dist_iter.dimensions() {
        return Err(Error::incompatible(format!(
            "decoded dimensions differ: {:?} vs {:?}",
            ref_iter.dimensions(),
            dist_iter.dimensions()
        )));
    }

    let every = compare.every.max(1);
    let mut frames = Vec::new();
    let mut compared = 0u64;

    loop {
        let Some(ref_item) = ref_iter.next() else {
            break;
        };
        let Some(dist_item) = dist_iter.next() else {
            break;
        };
        let (idx, ref_frame) = ref_item?;
        let (idx_b, dist_frame) = dist_item?;
        let frame_index = idx.min(idx_b);
        if frame_index % every != 0 {
            continue;
        }
        let results = metrics.compare(&ref_frame.as_view(), &dist_frame.as_view())?;
        frames.push(FrameReport {
            frame_index,
            pts_seconds: None,
            metrics: results,
        });
        compared += 1;
        if compare.max_frames.is_some_and(|max| compared >= max) {
            break;
        }
    }

    let mean_metrics = mean_metric_outputs(&frames);
    Ok(VideoReport {
        reference: reference_path.display().to_string(),
        distorted: distorted_path.display().to_string(),
        dimensions: ref_iter.dimensions(),
        compared_frames: compared,
        frames,
        mean_metrics,
    })
}

fn mean_metric_outputs(frames: &[FrameReport]) -> Vec<MetricOutput> {
    let mut sums: BTreeMap<String, (MetricOutput, f64, u64, bool)> = BTreeMap::new();
    for frame in frames {
        for metric in &frame.metrics {
            let entry = sums
                .entry(metric.name.clone())
                .or_insert_with(|| (metric.clone(), 0.0, 0, false));
            if metric.score.is_infinite() {
                entry.3 = true;
            } else if metric.score.is_finite() {
                entry.1 += metric.score;
                entry.2 += 1;
            }
        }
    }
    sums.into_values()
        .map(|(mut metric, sum, count, has_inf)| {
            metric.score = if count == 0 && has_inf {
                f64::INFINITY
            } else if count > 0 {
                sum / count as f64
            } else {
                f64::NAN
            };
            metric.details.clear();
            metric
                .details
                .insert("frames".to_string(), frames.len() as f64);
            metric.name = format!("mean_{}", metric.name);
            metric
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("imq-video-test-{}-{name}", std::process::id()))
    }

    #[test]
    fn probe_reports_missing_input_before_ffprobe() {
        let path = temp_path("missing.mp4");
        let err = probe_video(&path, &FfmpegOptions::default()).unwrap_err();

        assert!(matches!(err, Error::InputNotFound { .. }));
        assert!(err.to_string().contains("input file not found"));
    }

    #[test]
    fn probe_reports_missing_ffprobe_separately() {
        let path = temp_path("empty.mp4");
        fs::write(&path, []).unwrap();
        let options = FfmpegOptions {
            ffprobe: temp_path("missing-ffprobe"),
            ..Default::default()
        };

        let err = probe_video(&path, &options).unwrap_err();
        let _ = fs::remove_file(path);

        assert!(matches!(err, Error::ExternalToolNotFound { .. }));
        assert!(err.to_string().contains("external tool"));
        assert!(err.to_string().contains("--ffprobe"));
    }
}
