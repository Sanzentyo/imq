//! External ffmpeg adapter.
//!
//! This module is intentionally not part of the Sans-I/O core. It shells out to
//! `ffprobe` for metadata and `ffmpeg` for raw RGBA frames on stdout, then feeds
//! those frames into the same zero-copy/owned core types as still images.

use crate::frame::{Dimensions, FrameOwned, PixelFormat};
use crate::metrics::{MetricOutput, MetricSet};
use crate::report::{FrameReport, VideoReport};
use crate::video_analysis::{
    TimestampAlignmentReport, TimestampPairingOptions, TimestampTransform, TimestampedFrame,
    align_frames_by_timestamp, estimate_timestamp_transform,
};
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;

const PROCESS_STDERR_LIMIT: usize = 1024 * 1024;
const PROBE_STDOUT_LIMIT: usize = 16 * 1024 * 1024;

fn read_limited_stderr(mut reader: impl Read) -> String {
    let mut bytes = Vec::with_capacity(PROCESS_STDERR_LIMIT.min(8192));
    let mut buffer = [0u8; 8192];
    let mut truncated = false;
    loop {
        match reader.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                let remaining = PROCESS_STDERR_LIMIT.saturating_sub(bytes.len());
                let retained = read.min(remaining);
                bytes.extend_from_slice(&buffer[..retained]);
                truncated |= retained != read;
            }
        }
    }
    let mut message = String::from_utf8_lossy(&bytes).into_owned();
    if truncated {
        message.push_str("\n[stderr truncated after 1048576 bytes]");
    }
    message
}

fn run_with_limited_stderr(
    command: &mut Command,
    program: &Path,
    option: &str,
    max_stdout_bytes: usize,
) -> Result<(ExitStatus, Vec<u8>, String)> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| process_start_failed(program, option, error))?;
    let mut stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(Error::invalid_frame("failed to capture process stdout"));
        }
    };
    let stderr_reader = child
        .stderr
        .take()
        .map(|stderr| std::thread::spawn(move || read_limited_stderr(stderr)));
    let mut stdout_bytes = Vec::new();
    let read_limit = u64::try_from(max_stdout_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    if let Err(error) = stdout
        .by_ref()
        .take(read_limit)
        .read_to_end(&mut stdout_bytes)
    {
        let _ = child.kill();
        let _ = child.wait();
        if let Some(reader) = stderr_reader {
            let _ = reader.join();
        }
        return Err(error.into());
    }
    drop(stdout);
    if stdout_bytes.len() > max_stdout_bytes {
        let _ = child.kill();
        let _ = child.wait();
        let stderr = stderr_reader
            .and_then(|reader| reader.join().ok())
            .unwrap_or_default();
        return Err(Error::invalid_frame(format!(
            "process `{}` stdout exceeded the {max_stdout_bytes}-byte limit{}",
            program.display(),
            if stderr.trim().is_empty() {
                String::new()
            } else {
                format!(": {}", stderr.trim())
            }
        )));
    }
    let status = match child.wait() {
        Ok(status) => status,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            if let Some(reader) = stderr_reader {
                let _ = reader.join();
            }
            return Err(error.into());
        }
    };
    let stderr = stderr_reader
        .and_then(|reader| reader.join().ok())
        .unwrap_or_default();
    Ok((status, stdout_bytes, stderr))
}

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
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub avg_frame_rate: Option<f64>,
    /// Declared/nominal frame rate, if ffprobe reports it.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub nominal_frame_rate: Option<f64>,
    /// Stream time base in seconds per timestamp tick.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub time_base_seconds: Option<f64>,
    /// Stream start timestamp in seconds.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub start_time_seconds: Option<f64>,
    /// Number of frames, if container metadata reports it.
    pub nb_frames: Option<u64>,
    /// Codec name, if known.
    pub codec_name: Option<String>,
    /// Duration in seconds, if known.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub duration_seconds: Option<f64>,
    /// Stream bit rate in bits per second, if known.
    pub bit_rate: Option<u64>,
    /// Decoder pixel-format name, such as `yuv420p`.
    pub pixel_format: Option<String>,
    /// ffprobe color-range name.
    pub color_range: Option<String>,
    /// ffprobe color-space/matrix name.
    pub color_space: Option<String>,
    /// ffprobe color-transfer name.
    pub color_transfer: Option<String>,
    /// ffprobe color-primaries name.
    pub color_primaries: Option<String>,
    /// Field order, such as `progressive` or `tt`.
    pub field_order: Option<String>,
}

/// Metadata for one decoded video frame as reported by ffprobe.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VideoFrameInfo {
    /// Decode/output-order frame index used by the rawvideo iterator.
    pub frame_index: u64,
    /// Best-effort presentation timestamp in seconds.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub pts_seconds: Option<f64>,
    /// Packet/frame duration in seconds, when present.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub duration_seconds: Option<f64>,
    /// Whether this is a key frame.
    pub key_frame: Option<bool>,
    /// Coded picture type (`I`, `P`, `B`, etc.).
    pub picture_type: Option<String>,
}

impl VideoFrameInfo {
    fn effective_pts_seconds(&self, video: &VideoInfo) -> Option<f64> {
        self.pts_seconds.or_else(|| {
            video.avg_frame_rate.and_then(|rate| {
                (rate.is_finite() && rate > 0.0).then(|| {
                    video.start_time_seconds.unwrap_or(0.0) + self.frame_index as f64 / rate
                })
            })
        })
    }
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
    #[serde(default)]
    format: Option<ProbeFormat>,
}

#[derive(Debug, serde::Deserialize)]
struct ProbeStream {
    width: Option<u32>,
    height: Option<u32>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
    time_base: Option<String>,
    start_time: Option<String>,
    nb_frames: Option<String>,
    codec_name: Option<String>,
    duration: Option<String>,
    bit_rate: Option<String>,
    pix_fmt: Option<String>,
    color_range: Option<String>,
    color_space: Option<String>,
    color_transfer: Option<String>,
    color_primaries: Option<String>,
    field_order: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct ProbeFormat {
    duration: Option<String>,
    start_time: Option<String>,
    bit_rate: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct ProbeFrameRoot {
    #[serde(default)]
    frames: Vec<ProbeFrame>,
}

#[derive(Debug, serde::Deserialize)]
struct ProbeFrame {
    best_effort_timestamp_time: Option<String>,
    pts_time: Option<String>,
    pkt_dts_time: Option<String>,
    pkt_duration_time: Option<String>,
    duration_time: Option<String>,
    key_frame: Option<u8>,
    pict_type: Option<String>,
}

/// Probes a video using ffprobe.
pub fn probe_video(path: impl AsRef<Path>, options: &FfmpegOptions) -> Result<VideoInfo> {
    let path = path.as_ref();
    ensure_input_exists(path)?;
    let mut command = Command::new(&options.ffprobe);
    command
        .args([
            "-v",
            "error",
            "-select_streams",
            &format!("v:{}", options.stream_index),
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,time_base,start_time,nb_frames,codec_name,duration,bit_rate,pix_fmt,color_range,color_space,color_transfer,color_primaries,field_order:format=duration,start_time,bit_rate",
            "-of",
            "json",
        ])
        .arg(path);
    let (status, stdout, stderr) = run_with_limited_stderr(
        &mut command,
        &options.ffprobe,
        "--ffprobe",
        PROBE_STDOUT_LIMIT,
    )?;

    if !status.success() {
        return Err(Error::ProcessFailed {
            program: options.ffprobe.display().to_string(),
            status: status.to_string(),
            stderr,
        });
    }

    let root: ProbeRoot = serde_json::from_slice(&stdout)?;
    let stream = root
        .streams
        .into_iter()
        .next()
        .ok_or_else(|| Error::invalid_frame("ffprobe did not return a video stream"))?;

    let format = root.format;
    Ok(VideoInfo {
        width: stream
            .width
            .ok_or_else(|| Error::invalid_frame("ffprobe stream has no width"))?,
        height: stream
            .height
            .ok_or_else(|| Error::invalid_frame("ffprobe stream has no height"))?,
        avg_frame_rate: stream.avg_frame_rate.as_deref().and_then(parse_rational),
        nominal_frame_rate: stream.r_frame_rate.as_deref().and_then(parse_rational),
        time_base_seconds: stream.time_base.as_deref().and_then(parse_rational),
        start_time_seconds: stream
            .start_time
            .as_deref()
            .and_then(parse_finite)
            .or_else(|| {
                format
                    .as_ref()
                    .and_then(|format| format.start_time.as_deref())
                    .and_then(parse_finite)
            }),
        nb_frames: stream.nb_frames.as_deref().and_then(|s| s.parse().ok()),
        codec_name: stream.codec_name,
        duration_seconds: stream
            .duration
            .as_deref()
            .and_then(parse_finite)
            .or_else(|| {
                format
                    .as_ref()
                    .and_then(|format| format.duration.as_deref())
                    .and_then(parse_finite)
            }),
        bit_rate: stream
            .bit_rate
            .as_deref()
            .and_then(|value| value.parse().ok())
            .or_else(|| {
                format
                    .as_ref()
                    .and_then(|format| format.bit_rate.as_deref())
                    .and_then(|value| value.parse().ok())
            }),
        pixel_format: stream.pix_fmt,
        color_range: stream.color_range,
        color_space: stream.color_space,
        color_transfer: stream.color_transfer,
        color_primaries: stream.color_primaries,
        field_order: stream.field_order,
    })
}

/// Probes presentation timestamps and basic coding metadata for every video frame.
pub fn probe_video_frames(
    path: impl AsRef<Path>,
    options: &FfmpegOptions,
) -> Result<Vec<VideoFrameInfo>> {
    let path = path.as_ref();
    ensure_input_exists(path)?;
    let mut command = Command::new(&options.ffprobe);
    command
        .args([
            "-v",
            "error",
            "-select_streams",
            &format!("v:{}", options.stream_index),
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time,pts_time,pkt_dts_time,pkt_duration_time,duration_time,key_frame,pict_type",
            "-of",
            "json",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| process_start_failed(&options.ffprobe, "--ffprobe", error))?;
    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(Error::invalid_frame(
                "failed to capture ffprobe frame metadata",
            ));
        }
    };
    let stderr_reader = child
        .stderr
        .take()
        .map(|stderr| std::thread::spawn(move || read_limited_stderr(stderr)));
    let parsed = serde_json::from_reader::<_, ProbeFrameRoot>(stdout);
    let status = match child.wait() {
        Ok(status) => status,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            if let Some(reader) = stderr_reader {
                let _ = reader.join();
            }
            return Err(error.into());
        }
    };
    let stderr = stderr_reader
        .and_then(|reader| reader.join().ok())
        .unwrap_or_default();
    if !status.success() {
        return Err(Error::ProcessFailed {
            program: options.ffprobe.display().to_string(),
            status: status.to_string(),
            stderr,
        });
    }
    let root = parsed?;
    root.frames
        .into_iter()
        .enumerate()
        .map(|(index, frame)| {
            Ok(VideoFrameInfo {
                frame_index: u64::try_from(index)
                    .map_err(|_| Error::invalid_frame("frame index overflows u64"))?,
                pts_seconds: frame
                    .best_effort_timestamp_time
                    .as_deref()
                    .and_then(parse_finite)
                    .or_else(|| frame.pts_time.as_deref().and_then(parse_finite))
                    .or_else(|| frame.pkt_dts_time.as_deref().and_then(parse_finite)),
                duration_seconds: frame
                    .pkt_duration_time
                    .as_deref()
                    .and_then(parse_finite)
                    .or_else(|| frame.duration_time.as_deref().and_then(parse_finite)),
                key_frame: frame.key_frame.map(|value| value != 0),
                picture_type: frame.pict_type,
            })
        })
        .collect()
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

fn parse_finite(value: &str) -> Option<f64> {
    value
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
}

/// Iterator over decoded RGBA8 frames from ffmpeg stdout.
pub struct FfmpegFrameIter {
    child: Child,
    stdout: ChildStdout,
    stderr_reader: Option<JoinHandle<String>>,
    dims: Dimensions,
    frame_bytes: usize,
    avg_frame_rate: Option<f64>,
    start_time_seconds: f64,
    ffmpeg_program: String,
    next_index: u64,
    done: bool,
}

impl FfmpegFrameIter {
    /// Spawns ffmpeg and starts reading raw RGBA frames.
    pub fn spawn(path: impl AsRef<Path>, options: &FfmpegOptions) -> Result<Self> {
        let path = path.as_ref();
        let probed = probe_video(path, options)?;
        Self::spawn_with_info(path, options, &probed)
    }

    fn spawn_with_info(path: &Path, options: &FfmpegOptions, probed: &VideoInfo) -> Result<Self> {
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
        command.args([
            "-vsync", "0", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1",
        ]);
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command
            .spawn()
            .map_err(|error| process_start_failed(&options.ffmpeg, "--ffmpeg", error))?;
        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Error::invalid_frame("failed to capture ffmpeg stdout"));
            }
        };
        let stderr_reader = child
            .stderr
            .take()
            .map(|stderr| std::thread::spawn(move || read_limited_stderr(stderr)));
        Ok(Self {
            child,
            stdout,
            stderr_reader,
            dims,
            frame_bytes,
            avg_frame_rate: probed.avg_frame_rate,
            start_time_seconds: probed.start_time_seconds.unwrap_or(0.0),
            ffmpeg_program: options.ffmpeg.display().to_string(),
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

    /// Estimates a presentation timestamp from the average frame rate.
    ///
    /// Use [`FfmpegTimestampedFrameIter`] when exact per-frame ffprobe PTS is
    /// required for VFR material.
    pub fn estimated_pts_seconds(&self, frame_index: u64) -> Option<f64> {
        self.avg_frame_rate.and_then(|rate| {
            (rate.is_finite() && rate > 0.0)
                .then(|| self.start_time_seconds + frame_index as f64 / rate)
        })
    }

    fn finish(&mut self, incomplete_frame: bool) -> Result<()> {
        self.done = true;
        let status = match self.child.wait() {
            Ok(status) => status,
            Err(error) => {
                let _ = self.child.kill();
                let _ = self.child.wait();
                if let Some(reader) = self.stderr_reader.take() {
                    let _ = reader.join();
                }
                return Err(error.into());
            }
        };
        let mut stderr = self
            .stderr_reader
            .take()
            .and_then(|reader| reader.join().ok())
            .unwrap_or_default();
        if status.success() && !incomplete_frame {
            Ok(())
        } else {
            if stderr.trim().is_empty() && incomplete_frame {
                stderr = "ffmpeg ended before a complete raw frame was available".to_string();
            }
            Err(Error::ProcessFailed {
                program: self.ffmpeg_program.clone(),
                status: status.to_string(),
                stderr,
            })
        }
    }

    fn terminate(&mut self) {
        self.done = true;
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.stderr_reader.take() {
            let _ = reader.join();
        }
    }
}

impl Iterator for FfmpegFrameIter {
    type Item = Result<(u64, FrameOwned)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let mut buf = vec![0u8; self.frame_bytes];
        let mut filled = 0usize;
        loop {
            match self.stdout.read(&mut buf[filled..]) {
                Ok(0) => {
                    return match self.finish(filled != 0) {
                        Ok(()) => None,
                        Err(error) => Some(Err(error)),
                    };
                }
                Ok(read) => {
                    filled += read;
                    if filled != self.frame_bytes {
                        continue;
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    self.terminate();
                    return Some(Err(Error::Io(error)));
                }
            }

            if filled == self.frame_bytes {
                let idx = self.next_index;
                self.next_index += 1;
                return Some(
                    FrameOwned::packed_tight(
                        buf,
                        self.dims.width,
                        self.dims.height,
                        PixelFormat::Rgba8,
                    )
                    .map(|f| (idx, f)),
                );
            }
        }
    }
}

impl Drop for FfmpegFrameIter {
    fn drop(&mut self) {
        if !self.done {
            self.terminate();
        }
    }
}

/// One decoded RGBA8 frame with ffprobe timing/coding metadata.
#[derive(Debug, Clone)]
pub struct DecodedVideoFrame {
    /// Decode/output-order frame index.
    pub frame_index: u64,
    /// Presentation timestamp, exact where ffprobe reported one and estimated
    /// from average frame rate otherwise.
    pub pts_seconds: Option<f64>,
    /// Packet/frame duration in seconds.
    pub duration_seconds: Option<f64>,
    /// Whether this is a key frame.
    pub key_frame: Option<bool>,
    /// Coded picture type (`I`, `P`, `B`, etc.).
    pub picture_type: Option<String>,
    /// Decoded RGBA8 pixels.
    pub frame: FrameOwned,
}

/// Streaming RGBA8 decoder that attaches exact per-frame ffprobe timestamps.
pub struct FfmpegTimestampedFrameIter {
    frames: FfmpegFrameIter,
    info: VideoInfo,
    frame_info: Vec<VideoFrameInfo>,
}

impl FfmpegTimestampedFrameIter {
    /// Probes frame metadata, then starts the rawvideo decoder.
    pub fn spawn(path: impl AsRef<Path>, options: &FfmpegOptions) -> Result<Self> {
        let path = path.as_ref();
        let info = probe_video(path, options)?;
        let frame_info = probe_video_frames(path, options)?;
        let frames = FfmpegFrameIter::spawn_with_info(path, options, &info)?;
        Ok(Self {
            frames,
            info,
            frame_info,
        })
    }

    /// Decoded dimensions.
    pub fn dimensions(&self) -> Dimensions {
        self.frames.dimensions()
    }

    /// Stream-level metadata.
    pub fn video_info(&self) -> &VideoInfo {
        &self.info
    }

    /// Probed per-frame metadata.
    pub fn frame_info(&self) -> &[VideoFrameInfo] {
        &self.frame_info
    }

    /// Returns timestamp descriptors suitable for timeline pairing.
    ///
    /// Missing per-frame timestamps become `NaN` and are reported as
    /// unmatched. Average-frame-rate timestamps are deliberately not invented:
    /// doing so would make VFR alignment appear exact when timing metadata is
    /// unavailable.
    pub fn timestamped_frames(&self) -> Vec<TimestampedFrame> {
        exact_timestamped_frames(&self.frame_info)
    }
}

fn exact_timestamped_frames(frame_info: &[VideoFrameInfo]) -> Vec<TimestampedFrame> {
    frame_info
        .iter()
        .map(|frame| TimestampedFrame {
            index: frame.frame_index,
            pts_seconds: frame
                .pts_seconds
                .filter(|pts_seconds| pts_seconds.is_finite())
                .unwrap_or(f64::NAN),
        })
        .collect()
}

impl Iterator for FfmpegTimestampedFrameIter {
    type Item = Result<DecodedVideoFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        let decoded = self.frames.next()?;
        Some(decoded.map(|(frame_index, frame)| {
            let metadata = usize::try_from(frame_index)
                .ok()
                .and_then(|index| self.frame_info.get(index));
            DecodedVideoFrame {
                frame_index,
                pts_seconds: metadata
                    .and_then(|metadata| metadata.effective_pts_seconds(&self.info))
                    .or_else(|| self.frames.estimated_pts_seconds(frame_index)),
                duration_seconds: metadata.and_then(|metadata| metadata.duration_seconds),
                key_frame: metadata.and_then(|metadata| metadata.key_frame),
                picture_type: metadata.and_then(|metadata| metadata.picture_type.clone()),
                frame,
            }
        }))
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
    let (status, stdout, stderr) =
        run_with_limited_stderr(&mut command, &options.ffmpeg, "--ffmpeg", frame_bytes)?;
    if !status.success() {
        return Err(Error::ProcessFailed {
            program: options.ffmpeg.display().to_string(),
            status: status.to_string(),
            stderr,
        });
    }
    if stdout.len() != frame_bytes {
        return Err(Error::invalid_frame(format!(
            "expected {frame_bytes} bytes for one frame, got {}",
            stdout.len()
        )));
    }
    FrameOwned::packed_tight(stdout, dims.width, dims.height, PixelFormat::Rgba8)
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

/// Control returned by a streaming video comparison callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoStreamControl {
    /// Continue decoding and comparing frames.
    Continue,
    /// Stop after the frame most recently delivered to the callback.
    Stop,
}

/// Bounded-memory summary returned by [`compare_videos_streaming`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VideoStreamSummary {
    /// Reference path.
    pub reference: String,
    /// Distorted path.
    pub distorted: String,
    /// Compared dimensions.
    pub dimensions: Dimensions,
    /// Number of frame pairs delivered to the callback.
    pub compared_frames: u64,
    /// Number of reference frames pulled from the decoder, including skipped samples.
    pub decoded_reference_frames: u64,
    /// Number of distorted frames pulled from the decoder, including skipped samples.
    pub decoded_distorted_frames: u64,
    /// Whether the reference decoder reached end-of-stream.
    pub reference_eof: bool,
    /// Whether the distorted decoder reached end-of-stream.
    pub distorted_eof: bool,
    /// Whether exactly one decoder reached end-of-stream before comparison stopped.
    pub length_mismatch_detected: bool,
    /// Whether a callback stop or frame cap ended comparison before end-of-stream.
    pub stopped_early: bool,
    /// Mean metric outputs accumulated without retaining per-frame reports.
    pub mean_metrics: Vec<MetricOutput>,
}

/// Options for VFR-safe timestamp-aligned comparison.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimestampVideoCompareOptions {
    /// Sampling and output-frame cap applied after timestamp pairing.
    pub compare: VideoCompareOptions,
    /// Pairing tolerance and reuse policy.
    pub pairing: TimestampPairingOptions,
    /// Explicit distorted-to-reference transform, if known.
    pub transform: Option<TimestampTransform>,
    /// Estimate offset and clock drift when no explicit transform is supplied.
    pub estimate_transform: bool,
}

impl Default for TimestampVideoCompareOptions {
    fn default() -> Self {
        Self {
            compare: VideoCompareOptions::default(),
            pairing: TimestampPairingOptions::default(),
            transform: None,
            estimate_transform: true,
        }
    }
}

/// Timestamp-aligned video report plus pairing diagnostics.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimestampVideoComparison {
    /// Frame-level metric report.
    pub video: VideoReport,
    /// Accepted pairs, unmatched frames, transform, and residuals.
    pub alignment: TimestampAlignmentReport,
}

/// Compares videos incrementally and does not retain per-frame reports.
///
/// The callback can serialize, gate, or otherwise consume each report as soon
/// as it is produced. This keeps memory bounded for long captures.
pub fn compare_videos_streaming<F>(
    reference: impl AsRef<Path>,
    distorted: impl AsRef<Path>,
    ffmpeg: &FfmpegOptions,
    compare: &VideoCompareOptions,
    metrics: &MetricSet,
    mut on_frame: F,
) -> Result<VideoStreamSummary>
where
    F: FnMut(&FrameReport) -> Result<VideoStreamControl>,
{
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

    let dimensions = ref_iter.dimensions();
    let every = compare.every.max(1);
    let mut compared = 0u64;
    let mut decoded_reference_frames = 0u64;
    let mut decoded_distorted_frames = 0u64;
    let mut reference_eof = false;
    let mut distorted_eof = false;
    let mut stopped_early = false;
    let mut accumulator = MetricMeanAccumulator::default();
    loop {
        if compare.max_frames.is_some_and(|max| compared >= max) {
            stopped_early = true;
            break;
        }
        let reference_item = ref_iter.next();
        let distorted_item = dist_iter.next();
        decoded_reference_frames += u64::from(matches!(&reference_item, Some(Ok(_))));
        decoded_distorted_frames += u64::from(matches!(&distorted_item, Some(Ok(_))));
        reference_eof |= reference_item.is_none();
        distorted_eof |= distorted_item.is_none();
        let (ref_item, dist_item) = match (reference_item, distorted_item) {
            (Some(Err(error)), _) | (_, Some(Err(error))) => return Err(error),
            (Some(Ok(reference)), Some(Ok(distorted))) => (reference, distorted),
            (None, None) | (None, Some(Ok(_))) | (Some(Ok(_)), None) => break,
        };
        let (reference_index, ref_frame) = ref_item;
        let (distorted_index, dist_frame) = dist_item;
        let frame_index = reference_index.min(distorted_index);
        if frame_index % every != 0 {
            continue;
        }
        let report = FrameReport {
            frame_index,
            pts_seconds: ref_iter.estimated_pts_seconds(reference_index),
            metrics: metrics.compare(&ref_frame.as_view(), &dist_frame.as_view())?,
        };
        accumulator.push(&report);
        compared += 1;
        let control = on_frame(&report)?;
        if matches!(control, VideoStreamControl::Stop) {
            stopped_early = true;
            break;
        }
    }

    Ok(VideoStreamSummary {
        reference: reference_path.display().to_string(),
        distorted: distorted_path.display().to_string(),
        dimensions,
        compared_frames: compared,
        decoded_reference_frames,
        decoded_distorted_frames,
        reference_eof,
        distorted_eof,
        length_mismatch_detected: reference_eof ^ distorted_eof,
        stopped_early,
        mean_metrics: accumulator.finish(),
    })
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
    let mut frames = Vec::new();
    let summary = compare_videos_streaming(
        reference_path,
        distorted_path,
        ffmpeg,
        compare,
        metrics,
        |frame| {
            frames.push(frame.clone());
            Ok(VideoStreamControl::Continue)
        },
    )?;
    Ok(VideoReport {
        reference: summary.reference,
        distorted: summary.distorted,
        dimensions: summary.dimensions,
        compared_frames: summary.compared_frames,
        frames,
        mean_metrics: summary.mean_metrics,
    })
}

/// Compares videos by ffprobe presentation timestamps instead of decode order.
///
/// This path is intended for VFR material and captures with duplicated or
/// dropped frames. Decoded frames are still streamed; only compact ffprobe
/// metadata and the returned frame reports are retained.
pub fn compare_videos_by_timestamp(
    reference: impl AsRef<Path>,
    distorted: impl AsRef<Path>,
    ffmpeg: &FfmpegOptions,
    options: &TimestampVideoCompareOptions,
    metrics: &MetricSet,
) -> Result<TimestampVideoComparison> {
    validate_timestamp_compare_options(options)?;
    let reference_path = reference.as_ref();
    let distorted_path = distorted.as_ref();
    let mut reference_frames = FfmpegTimestampedFrameIter::spawn(reference_path, ffmpeg)?;
    let mut distorted_frames = FfmpegTimestampedFrameIter::spawn(distorted_path, ffmpeg)?;
    if reference_frames.dimensions() != distorted_frames.dimensions() {
        return Err(Error::incompatible(format!(
            "decoded dimensions differ: {:?} vs {:?}",
            reference_frames.dimensions(),
            distorted_frames.dimensions()
        )));
    }
    let dimensions = reference_frames.dimensions();
    let reference_timestamps = reference_frames.timestamped_frames();
    let distorted_timestamps = distorted_frames.timestamped_frames();
    require_usable_timestamps("reference", &reference_timestamps)?;
    require_usable_timestamps("distorted", &distorted_timestamps)?;
    let transform = options.transform.unwrap_or_else(|| {
        if options.estimate_transform {
            estimate_timestamp_transform(&reference_timestamps, &distorted_timestamps)
                .unwrap_or_default()
        } else {
            TimestampTransform::default()
        }
    });
    let alignment = align_frames_by_timestamp(
        &reference_timestamps,
        &distorted_timestamps,
        options.pairing,
        transform,
    );

    let every = options.compare.every.max(1);
    let mut frames = Vec::new();
    let mut reference_current = reference_frames.next().transpose()?;
    let mut distorted_current = distorted_frames.next().transpose()?;
    let mut last_distorted: Option<DecodedVideoFrame> = None;
    for (pair_index, pair) in alignment.pairs.iter().enumerate() {
        if u64::try_from(pair_index).unwrap_or(u64::MAX) % every != 0 {
            continue;
        }
        if options
            .compare
            .max_frames
            .is_some_and(|max| frames.len() as u64 >= max)
        {
            break;
        }
        advance_to_frame(
            &mut reference_frames,
            &mut reference_current,
            pair.reference_index,
        )?;
        let reference_frame = reference_current.as_ref().ok_or_else(|| {
            Error::invalid_frame(format!(
                "reference decoder ended before paired frame {}",
                pair.reference_index
            ))
        })?;

        let distorted_frame = if last_distorted
            .as_ref()
            .is_some_and(|frame| frame.frame_index == pair.distorted_index)
        {
            last_distorted.as_ref().expect("checked Some")
        } else {
            advance_to_frame(
                &mut distorted_frames,
                &mut distorted_current,
                pair.distorted_index,
            )?;
            let frame = distorted_current.as_ref().ok_or_else(|| {
                Error::invalid_frame(format!(
                    "distorted decoder ended before paired frame {}",
                    pair.distorted_index
                ))
            })?;
            last_distorted = Some(frame.clone());
            last_distorted.as_ref().expect("just assigned")
        };

        let mut outputs = metrics.compare(
            &reference_frame.frame.as_view(),
            &distorted_frame.frame.as_view(),
        )?;
        for output in &mut outputs {
            output.details.insert(
                "reference_frame_index".to_string(),
                pair.reference_index as f64,
            );
            output.details.insert(
                "distorted_frame_index".to_string(),
                pair.distorted_index as f64,
            );
            output.details.insert(
                "distorted_pts_seconds".to_string(),
                pair.distorted_pts_seconds,
            );
            output
                .details
                .insert("timestamp_delta_seconds".to_string(), pair.delta_seconds);
        }
        frames.push(FrameReport {
            frame_index: pair.reference_index,
            pts_seconds: Some(pair.reference_pts_seconds),
            metrics: outputs,
        });
    }

    Ok(TimestampVideoComparison {
        video: VideoReport {
            reference: reference_path.display().to_string(),
            distorted: distorted_path.display().to_string(),
            dimensions,
            compared_frames: frames.len() as u64,
            mean_metrics: mean_metric_outputs(&frames),
            frames,
        },
        alignment,
    })
}

fn require_usable_timestamps(label: &str, timestamps: &[TimestampedFrame]) -> Result<()> {
    if timestamps.iter().any(|frame| frame.pts_seconds.is_finite()) {
        Ok(())
    } else {
        Err(Error::invalid_frame(format!(
            "{label} video has no usable per-frame presentation timestamps"
        )))
    }
}

fn validate_timestamp_compare_options(options: &TimestampVideoCompareOptions) -> Result<()> {
    if !options.pairing.max_delta_seconds.is_finite() || options.pairing.max_delta_seconds < 0.0 {
        return Err(Error::unsupported(
            "timestamp pairing tolerance must be finite and non-negative",
        ));
    }
    if let Some(transform) = options.transform {
        if !transform.scale.is_finite() || transform.scale <= 0.0 {
            return Err(Error::unsupported(
                "timestamp transform scale must be finite and greater than zero",
            ));
        }
        if !transform.offset_seconds.is_finite() {
            return Err(Error::unsupported(
                "timestamp transform offset must be finite",
            ));
        }
    }
    Ok(())
}

fn advance_to_frame(
    frames: &mut FfmpegTimestampedFrameIter,
    current: &mut Option<DecodedVideoFrame>,
    target: u64,
) -> Result<()> {
    if current
        .as_ref()
        .is_some_and(|frame| frame.frame_index > target)
    {
        return Err(Error::invalid_frame(
            "timestamp pair plan is not monotonic in decode order",
        ));
    }
    while current
        .as_ref()
        .is_some_and(|frame| frame.frame_index < target)
    {
        *current = frames.next().transpose()?;
    }
    Ok(())
}

fn mean_metric_outputs(frames: &[FrameReport]) -> Vec<MetricOutput> {
    let mut accumulator = MetricMeanAccumulator::default();
    for frame in frames {
        accumulator.push(frame);
    }
    accumulator.finish()
}

#[derive(Default)]
struct MetricMeanAccumulator {
    sums: BTreeMap<String, (MetricOutput, f64, u64, bool)>,
    frames: u64,
}

impl MetricMeanAccumulator {
    fn push(&mut self, frame: &FrameReport) {
        self.frames += 1;
        for metric in &frame.metrics {
            let entry = self
                .sums
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

    fn finish(self) -> Vec<MetricOutput> {
        self.sums
            .into_values()
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
                    .insert("frames".to_string(), self.frames as f64);
                metric.name = format!("mean_{}", metric.name);
                metric
            })
            .collect()
    }
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

    #[test]
    fn timestamp_descriptors_require_per_frame_pts() {
        let descriptors = exact_timestamped_frames(&[
            VideoFrameInfo {
                frame_index: 0,
                pts_seconds: None,
                duration_seconds: Some(1.0 / 30.0),
                key_frame: Some(true),
                picture_type: Some("I".to_string()),
            },
            VideoFrameInfo {
                frame_index: 1,
                pts_seconds: Some(0.125),
                duration_seconds: None,
                key_frame: None,
                picture_type: None,
            },
        ]);

        assert!(descriptors[0].pts_seconds.is_nan());
        assert_eq!(descriptors[1].pts_seconds, 0.125);
        assert!(require_usable_timestamps("reference", &[]).is_err());
        assert!(require_usable_timestamps("reference", &descriptors[..1]).is_err());
        assert!(require_usable_timestamps("reference", &descriptors).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn limited_process_capture_kills_on_oversized_stdout() {
        use std::os::unix::fs::PermissionsExt;

        let script = temp_path("oversized-stdout.sh");
        fs::write(
            &script,
            b"#!/bin/sh\nwhile :; do printf '0123456789abcdef'; done\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&script, permissions).unwrap();

        let mut command = Command::new(&script);
        let error = run_with_limited_stderr(&mut command, &script, "--ffmpeg", 128).unwrap_err();
        let _ = fs::remove_file(script);

        assert!(
            error
                .to_string()
                .contains("stdout exceeded the 128-byte limit")
        );
    }

    #[test]
    fn timestamp_compare_options_reject_non_finite_or_reversed_time() {
        for max_delta_seconds in [f64::NAN, f64::INFINITY, -0.001] {
            let options = TimestampVideoCompareOptions {
                pairing: TimestampPairingOptions {
                    max_delta_seconds,
                    allow_reuse_distorted: false,
                },
                ..TimestampVideoCompareOptions::default()
            };
            assert!(validate_timestamp_compare_options(&options).is_err());
        }
        for transform in [
            TimestampTransform {
                scale: 0.0,
                offset_seconds: 0.0,
            },
            TimestampTransform {
                scale: -1.0,
                offset_seconds: 0.0,
            },
            TimestampTransform {
                scale: f64::NAN,
                offset_seconds: 0.0,
            },
            TimestampTransform {
                scale: 1.0,
                offset_seconds: f64::INFINITY,
            },
        ] {
            let options = TimestampVideoCompareOptions {
                transform: Some(transform),
                ..TimestampVideoCompareOptions::default()
            };
            assert!(validate_timestamp_compare_options(&options).is_err());
        }
        assert!(
            validate_timestamp_compare_options(&TimestampVideoCompareOptions::default()).is_ok()
        );
    }

    #[test]
    #[ignore = "requires local ffmpeg and ffprobe executables"]
    fn timestamp_compare_matches_identical_vfr_timeline() {
        let path = temp_path("timestamp-vfr.mkv");
        let _ = fs::remove_file(&path);
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=16x16:rate=10:duration=1",
                "-vf",
                "select=eq(n\\,0)+eq(n\\,1)+eq(n\\,3)+eq(n\\,6)+eq(n\\,9)",
                "-fps_mode",
                "vfr",
                "-c:v",
                "ffv1",
                "-y",
            ])
            .arg(&path)
            .status()
            .unwrap();
        assert!(status.success());

        let ffmpeg = FfmpegOptions::default();
        let metadata = probe_video_frames(&path, &ffmpeg).unwrap();
        let pts = metadata
            .iter()
            .filter_map(|frame| frame.pts_seconds)
            .collect::<Vec<_>>();
        assert_eq!(pts.len(), 5);
        let deltas = pts
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect::<Vec<_>>();
        assert!(deltas.iter().any(|delta| (*delta - deltas[0]).abs() > 1e-6));

        let report = compare_videos_by_timestamp(
            &path,
            &path,
            &ffmpeg,
            &TimestampVideoCompareOptions {
                pairing: TimestampPairingOptions {
                    max_delta_seconds: 1e-9,
                    allow_reuse_distorted: false,
                },
                ..TimestampVideoCompareOptions::default()
            },
            &MetricSet::from_csv("mse").unwrap(),
        )
        .unwrap();
        let _ = fs::remove_file(path);
        assert_eq!(report.video.compared_frames, 5);
        assert_eq!(report.alignment.pairs.len(), 5);
        assert!(report.alignment.unmatched_reference_indices.is_empty());
        assert!(report.alignment.unmatched_distorted_indices.is_empty());
        assert!(
            report
                .video
                .frames
                .iter()
                .all(|frame| frame.metrics[0].score == 0.0)
        );
    }
}
