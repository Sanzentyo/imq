//! Command-line and optional TUI frontend for `imq`.

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use imq::adapters::bit_packed_gray::{BitOrder, BitPackedGrayFormat};
use imq::adapters::image_crate;
use imq::metrics::MetricSet;
use imq::report::{
    ComparisonGateReport, ComparisonInput, ComparisonReport, ComparisonThresholds,
    VideoFramePairComparisonReport, VideoReport,
};
use imq::{
    Dimensions, InputSpec, MetricOutput, RemoteFrameFormat, RemoteOptions, RemoteTransferMode,
    SshInput, VideoFramePair,
};
use serde::Serialize;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

const REMOTE_STDERR_TAIL_BYTES: usize = 64 * 1024;
static REMOTE_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Parser)]
#[command(name = "imq")]
#[command(about = "Image/video quality metrics with a Sans-I/O Rust core")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compare or inspect media, automatically selecting image/video handling by extension.
    #[command(alias = "c", alias = "cmp")]
    Compare(Box<CompareCmd>),
    /// Compare two still images decoded by the image crate.
    #[command(alias = "i")]
    Image(Box<ImageCmd>),
    /// Compare, gate, and rank multiple candidate images against one reference.
    #[command(alias = "rank", alias = "batch")]
    Suite(Box<SuiteCmd>),
    /// Inspect wgpu capabilities or run reusable GPU comparisons.
    #[cfg(feature = "gpu")]
    #[command(alias = "wgpu")]
    Gpu(GpuCmd),
    /// Report still-image statistics, color balance, histograms, and tendencies.
    #[command(alias = "s", alias = "stat")]
    Stats(StatsCmd),
    /// Compare two videos by piping RGBA frames from ffmpeg stdout.
    #[cfg(feature = "ffmpeg")]
    #[command(alias = "v")]
    Video(VideoCmd),
    /// Decode one video frame with ffmpeg and save it as PNG.
    #[cfg(feature = "ffmpeg")]
    #[command(alias = "x", alias = "extract")]
    ExtractFrame(ExtractFrameCmd),
    /// Probe a video stream with ffprobe.
    #[cfg(feature = "ffmpeg")]
    #[command(alias = "info")]
    Probe(ProbeCmd),
    /// Show image formats available through the image adapter.
    #[command(alias = "fmt")]
    Formats(FormatsCmd),
    /// Pack one or more images into an imqraw lossless raw bundle.
    #[command(alias = "raw-pack", alias = "bundle")]
    Pack(PackCmd),
    /// Inspect an imqraw lossless raw bundle.
    #[command(alias = "raw-info")]
    BundleInfo(BundleInfoCmd),
    /// Preview images or video thumbnails in the terminal.
    #[command(alias = "p")]
    Preview(Box<PreviewCmd>),
    /// Interactive terminal comparison view.
    #[command(alias = "t")]
    Tui(TuiCmd),
}

#[derive(Debug, Args)]
struct ImageCmd {
    /// Reference/original image.
    reference: PathBuf,
    /// Distorted/test image.
    distorted: PathBuf,
    /// Comma-separated metrics: psnr,ssim,wssim,mse,rmse,mae,maxae; optional domains: psnr:color,mse:all.
    #[arg(short, long, default_value = "psnr,ssim,mse,mae,maxae")]
    metrics: String,
    /// Print JSON instead of a text table.
    #[arg(short, long)]
    json: bool,
    /// Include per-image statistics, color balance, histograms, and tendencies.
    #[arg(short, long)]
    stats: bool,
    /// Number of histogram bins to emit when --stats is used.
    #[arg(long, default_value_t = 16)]
    histogram_bins: usize,
    #[command(flatten)]
    stdin: StdinImageArgs,
    #[command(flatten)]
    gate: GateArgs,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Debug, Args)]
struct SuiteCmd {
    /// Reference/original image.
    reference: PathBuf,
    /// Candidate images to compare and rank.
    #[arg(required = true, num_args = 1..)]
    candidates: Vec<PathBuf>,
    /// Comma-separated metrics shared by every candidate.
    #[arg(
        short,
        long,
        default_value = "psnr,ssim,wssim,ms-ssim,mse,rmse,mae,maxae"
    )]
    metrics: String,
    /// Metric output name used for the top-level rank; defaults to consensus rank.
    #[arg(long)]
    primary_metric: Option<String>,
    /// Consensus-rank metric weight, e.g. --weight psnr=2 or --weight mse=0.
    #[arg(long = "weight")]
    weights: Vec<String>,
    /// Candidate path used as the score-delta baseline.
    #[arg(long)]
    baseline: Option<PathBuf>,
    /// Threshold rule applied to every candidate. Supports absolute values and
    /// baseline-relative expressions such as psnr>=baseline-0.5.
    #[arg(long = "rule")]
    rules: Vec<String>,
    /// Maximum comparison workers; 0 uses available parallelism.
    #[arg(short = 'j', long, default_value_t = 0)]
    jobs: usize,
    /// Abort the suite on the first candidate comparison error.
    #[arg(long)]
    fail_fast: bool,
    /// Absolute tolerance used to assign tied ranks.
    #[arg(long, default_value_t = 1e-12)]
    rank_abs_tolerance: f64,
    /// Relative tolerance used to assign tied ranks.
    #[arg(long, default_value_t = 1e-9)]
    rank_rel_tolerance: f64,
    /// Print JSON instead of text.
    #[arg(long)]
    json: bool,
    /// Structured output format.
    #[arg(long, value_enum, default_value_t = OutputFormatArg::Text)]
    format: OutputFormatArg,
    /// Write output to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Args)]
struct GpuCmd {
    #[command(subcommand)]
    command: GpuSubcommand,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Subcommand)]
enum GpuSubcommand {
    /// Report the selected adapter, backend, driver, and effective limits.
    Info(GpuInfoCmd),
    /// Compare one reference against one or more candidates with one reusable context.
    Compare(GpuCompareCmd),
}

#[cfg(feature = "gpu")]
#[derive(Debug, Args)]
struct GpuInfoCmd {
    #[command(flatten)]
    adapter: GpuAdapterArgs,
    /// Structured output format.
    #[arg(long, value_enum, default_value_t = OutputFormatArg::Text)]
    format: OutputFormatArg,
    /// Write output to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Args)]
struct GpuCompareCmd {
    /// Reference/original image.
    reference: PathBuf,
    /// Candidate images compared in caller-provided order.
    #[arg(required = true, num_args = 1..)]
    candidates: Vec<PathBuf>,
    /// GPU metrics: mse,rmse,psnr,mae,maxae share one reduction; wssim uses a cached window kernel.
    #[arg(short, long, default_value = "mse,rmse,psnr,mae,maxae,wssim")]
    metrics: String,
    /// Error sample domain. Color compares RGB; all includes alpha.
    #[arg(long, value_enum, default_value_t = GpuDomainArg::Color)]
    domain: GpuDomainArg,
    /// Square window size used by wssim.
    #[arg(long, default_value_t = 8)]
    window: usize,
    /// Horizontal and vertical stride used by wssim.
    #[arg(long, default_value_t = 8)]
    window_stride: usize,
    #[command(flatten)]
    adapter: GpuAdapterArgs,
    /// CPU fallback policy for initialization or execution errors.
    #[arg(long, value_enum, default_value_t = GpuFallbackArg::Any)]
    fallback: GpuFallbackArg,
    /// Structured output format.
    #[arg(long, value_enum, default_value_t = OutputFormatArg::Text)]
    format: OutputFormatArg,
    /// Write output to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Args)]
struct GpuAdapterArgs {
    /// Adapter power preference.
    #[arg(long, value_enum, default_value_t = GpuPowerArg::High)]
    power: GpuPowerArg,
    /// Require a fallback/software adapter.
    #[arg(long)]
    fallback_adapter: bool,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, ValueEnum)]
enum GpuPowerArg {
    None,
    Low,
    High,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, ValueEnum)]
enum GpuFallbackArg {
    Never,
    Initialization,
    Any,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, ValueEnum)]
enum GpuDomainArg {
    Color,
    All,
}

#[derive(Debug, Args)]
struct StatsCmd {
    /// Input image.
    input: PathBuf,
    /// Number of histogram bins to emit.
    #[arg(long, default_value_t = 16)]
    histogram_bins: usize,
    /// Print JSON instead of text.
    #[arg(short, long)]
    json: bool,
    #[command(flatten)]
    stdin: StdinImageArgs,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Debug, Args)]
struct TuiCmd {
    /// Reference/original image, or an initial directory when TARGETS is omitted.
    reference: Option<PathBuf>,
    /// Distorted/test images to compare against the reference.
    targets: Vec<PathBuf>,
    /// Comma-separated metrics.
    #[arg(short, long, default_value = "psnr,ssim,mse,mae,maxae")]
    metrics: String,
    /// Number of decoded previews to keep in memory.
    #[arg(short = 'C', long, default_value_t = 32)]
    preview_cache: usize,
    /// Display previews at their exact decoded pixel dimensions.
    #[arg(short, long)]
    actual_size: bool,
}

#[derive(Debug, Args)]
struct CompareCmd {
    /// Reference/original image or video. With --stats and no DISTORTED, reports image statistics.
    reference: PathBuf,
    /// Distorted/test image or video.
    distorted: Option<PathBuf>,
    /// Comma-separated metrics.
    #[arg(short, long, default_value = "psnr,ssim,mse,mae,maxae")]
    metrics: String,
    /// Include image statistics, color balance, histograms, and tendencies.
    #[arg(short, long)]
    stats: bool,
    /// Number of histogram bins to emit when --stats is used.
    #[arg(long, default_value_t = 16)]
    histogram_bins: usize,
    #[command(flatten)]
    stdin: StdinImageArgs,
    #[command(flatten)]
    gate: GateArgs,
    #[command(flatten)]
    remote: RemoteArgs,
    /// Extract and compare a single video frame. Shorthand for --video-frames N.
    #[cfg(feature = "ffmpeg")]
    #[arg(long, conflicts_with = "video_frames")]
    video_frame: Option<u64>,
    /// Ordered video frame pairs, e.g. 0:1,30:31 or '(0,1),(30,31)'.
    #[cfg(feature = "ffmpeg")]
    #[arg(long, conflicts_with_all = ["video_frame", "every", "max_frames"])]
    video_frames: Option<String>,
    /// Compare every Nth decoded video frame.
    #[cfg(feature = "ffmpeg")]
    #[arg(long, default_value_t = 1)]
    every: u64,
    /// Maximum number of decoded video frame pairs to compare.
    #[cfg(feature = "ffmpeg")]
    #[arg(long)]
    max_frames: Option<u64>,
    /// Force a video comparison width. Requires --height.
    #[cfg(feature = "ffmpeg")]
    #[arg(long)]
    width: Option<u32>,
    /// Force a video comparison height. Requires --width.
    #[cfg(feature = "ffmpeg")]
    #[arg(long)]
    height: Option<u32>,
    /// ffmpeg executable.
    #[cfg(feature = "ffmpeg")]
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: PathBuf,
    /// ffprobe executable.
    #[cfg(feature = "ffmpeg")]
    #[arg(long, default_value = "ffprobe")]
    ffprobe: PathBuf,
    /// Video stream index.
    #[cfg(feature = "ffmpeg")]
    #[arg(long, default_value_t = 0)]
    stream: usize,
    /// Video frame alignment: decode-order pairing or VFR-safe timestamp pairing.
    #[cfg(feature = "ffmpeg")]
    #[arg(long, value_enum, default_value_t = VideoAlignmentArg::Decode)]
    align: VideoAlignmentArg,
    /// Maximum timestamp residual accepted when --align timestamp is selected.
    #[cfg(feature = "ffmpeg")]
    #[arg(long)]
    max_timestamp_delta: Option<f64>,
    /// Permit one distorted frame to match multiple reference frames.
    #[cfg(feature = "ffmpeg")]
    #[arg(long)]
    allow_reuse_distorted: bool,
    /// Disable automatic timestamp offset and clock-drift estimation.
    #[cfg(feature = "ffmpeg")]
    #[arg(long)]
    no_estimate_timestamp_transform: bool,
    /// Explicit distorted-timeline scale for timestamp alignment.
    #[cfg(feature = "ffmpeg")]
    #[arg(long)]
    timestamp_scale: Option<f64>,
    /// Explicit distorted-timeline offset in seconds for timestamp alignment.
    #[cfg(feature = "ffmpeg")]
    #[arg(long, allow_hyphen_values = true)]
    timestamp_offset: Option<f64>,
    /// Print JSON instead of text.
    #[arg(short, long)]
    json: bool,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Debug, Args, Clone)]
struct RemoteArgs {
    /// Shorthand host for non-URI remote paths.
    #[arg(long)]
    ssh: Option<String>,
    /// Remote transfer mode. The default never falls back to copying.
    #[arg(long, value_enum, default_value_t = RemoteTransferModeArg::Stream)]
    remote_transfer: RemoteTransferModeArg,
    /// Remote video frame stream format.
    #[arg(long, value_enum, default_value_t = RemoteFrameFormatArg::Png)]
    remote_frame_format: RemoteFrameFormatArg,
    /// SSH executable.
    #[arg(long, default_value = "ssh")]
    ssh_bin: PathBuf,
    /// SCP executable.
    #[arg(long, default_value = "scp")]
    scp_bin: PathBuf,
    /// SSH connect timeout in seconds.
    #[arg(long)]
    ssh_connect_timeout: Option<u64>,
    /// Pass BatchMode=yes to SSH.
    #[arg(long)]
    ssh_batch_mode: bool,
    /// Local directory for explicit remote copy modes.
    #[arg(long)]
    remote_copy_dir: Option<PathBuf>,
    /// Keep temporary copy-mode files.
    #[arg(long)]
    keep_temp: bool,
    /// Maximum remote stdout bytes to read.
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    remote_max_bytes: usize,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RemoteTransferModeArg {
    Stream,
    CopyInput,
    CopyFrame,
    CopySource,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RemoteFrameFormatArg {
    Png,
    Rgba,
}

impl RemoteArgs {
    fn options(&self) -> RemoteOptions {
        RemoteOptions {
            transfer: match self.remote_transfer {
                RemoteTransferModeArg::Stream => RemoteTransferMode::Stream,
                RemoteTransferModeArg::CopyInput => RemoteTransferMode::CopyInput,
                RemoteTransferModeArg::CopyFrame => RemoteTransferMode::CopyFrame,
                RemoteTransferModeArg::CopySource => RemoteTransferMode::CopySource,
            },
            frame_format: match self.remote_frame_format {
                RemoteFrameFormatArg::Png => RemoteFrameFormat::Png,
                RemoteFrameFormatArg::Rgba => RemoteFrameFormat::Rgba,
            },
            ssh: self.ssh_bin.clone(),
            scp: self.scp_bin.clone(),
            connect_timeout_seconds: self.ssh_connect_timeout,
            batch_mode: self.ssh_batch_mode,
            copy_dir: self.remote_copy_dir.clone(),
            keep_temp: self.keep_temp,
            max_bytes: self.remote_max_bytes,
        }
    }
}

#[derive(Debug, Args, Clone, Default)]
struct GateArgs {
    /// Metric key or domain used for pass/fail gates, e.g. psnr:rgb-visible or rgb-visible.
    #[arg(long)]
    selected_metric: Option<String>,
    /// Fail when the selected metric score is below this value.
    #[arg(long)]
    fail_under: Option<f64>,
    /// Fail when the selected metric max channel delta exceeds this 8-bit code value.
    #[arg(long)]
    max_selected_channel_delta: Option<f64>,
    /// Fail when the maximum alpha delta exceeds this 8-bit code value.
    #[arg(long)]
    max_alpha_delta: Option<u8>,
    /// Fail when alpha mismatch count exceeds this value.
    #[arg(long)]
    max_alpha_mismatches: Option<u64>,
    /// Fail when alpha mismatches above one LSB exceed this value.
    #[arg(long)]
    max_alpha_mismatches_beyond_one_lsb: Option<u64>,
}

impl GateArgs {
    fn thresholds(&self) -> ComparisonThresholds {
        ComparisonThresholds {
            selected_metric: self.selected_metric.clone(),
            fail_under: self.fail_under,
            max_selected_channel_delta: self.max_selected_channel_delta,
            max_alpha_delta: self.max_alpha_delta,
            max_alpha_mismatches: self.max_alpha_mismatches,
            max_alpha_mismatches_beyond_one_lsb: self.max_alpha_mismatches_beyond_one_lsb,
        }
    }
}

#[derive(Debug, Args)]
struct PreviewCmd {
    /// Image or video files to preview.
    inputs: Vec<PathBuf>,
    /// Terminal display mode.
    #[arg(short = 'D', long, value_enum, default_value_t = PreviewDisplayArg::Auto)]
    display: PreviewDisplayArg,
    /// Video decode policy.
    #[arg(long, value_enum, default_value_t = PreviewDecodeArg::Auto)]
    decode: PreviewDecodeArg,
    /// Preview size for each input, for example 120x60.
    #[arg(short, long, value_parser = parse_preview_size, conflicts_with = "actual_size")]
    size: Option<PreviewSize>,
    /// Maximum preview width for each input. Overridden by --size.
    #[arg(short = 'W', long, conflicts_with = "actual_size")]
    width: Option<u32>,
    /// Maximum preview height for each input. Overridden by --size.
    #[arg(short = 'H', long, conflicts_with = "actual_size")]
    height: Option<u32>,
    /// Fit policy for the requested preview size.
    #[arg(short, long, value_enum, default_value_t = PreviewFitArg::Contain, conflicts_with = "actual_size")]
    fit: PreviewFitArg,
    /// Number of montage rows.
    #[arg(short, long)]
    rows: Option<usize>,
    /// Number of montage columns.
    #[arg(short, long)]
    cols: Option<usize>,
    /// ffmpeg executable for video thumbnails.
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: PathBuf,
    /// Remote video frame to preview.
    #[arg(long)]
    video_frame: Option<u64>,
    #[command(flatten)]
    remote: RemoteArgs,
    /// Display at exact decoded pixel dimensions instead of fitting terminal area.
    #[arg(short, long)]
    actual_size: bool,
}

#[derive(Debug, Args)]
struct FormatsCmd {
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Debug, Args)]
struct PackCmd {
    /// Input image files to pack.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Write bundle bytes to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Per-input labels in input order. Defaults to each input path.
    #[arg(long = "label")]
    labels: Vec<String>,
    /// Tags to attach. Use TAG for all inputs, all:TAG for all inputs, or 1:TAG for a one-based input index.
    #[arg(long = "tag")]
    tags: Vec<String>,
}

#[derive(Debug, Args)]
struct BundleInfoCmd {
    /// imqraw bundle path, or `-` for stdin.
    #[arg(default_value = "-")]
    input: PathBuf,
    /// Structured output format.
    #[arg(long, value_enum, default_value_t = OutputFormatArg::Text)]
    format: OutputFormatArg,
    /// Write output to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct OutputArgs {
    /// Structured output format.
    #[arg(long, value_enum, default_value_t = OutputFormatArg::Text)]
    format: OutputFormatArg,
    /// Write output to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Append the report to a SQLite database.
    #[arg(long)]
    sqlite: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormatArg {
    Text,
    Json,
    Yaml,
    Toml,
    Csv,
}

#[cfg(feature = "ffmpeg")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum VideoAlignmentArg {
    /// Pair frames by decode order (fast, compatible default).
    Decode,
    /// Pair frames by presentation timestamp with alignment diagnostics.
    Timestamp,
}

#[derive(Debug, Args, Clone)]
struct StdinImageArgs {
    /// How to decode `-` image input from stdin.
    #[arg(long, value_enum, default_value_t = StdinImageFormatArg::Encoded)]
    stdin_format: StdinImageFormatArg,
    /// Raw stdin image width in pixels.
    #[arg(long)]
    raw_width: Option<u32>,
    /// Raw stdin image height in pixels.
    #[arg(long)]
    raw_height: Option<u32>,
    /// Raw stdin pixel format.
    #[arg(long, value_enum, default_value_t = RawPixelFormatArg::Rgba8)]
    raw_pixel_format: RawPixelFormatArg,
    /// Raw stdin row stride in bytes. Defaults to the tight row size for the selected raw pixel format.
    #[arg(long)]
    raw_stride: Option<usize>,
    /// Zero-based image index used when --stdin-format imqraw feeds one image argument.
    #[arg(long)]
    stdin_index: Option<usize>,
    /// Tag used when --stdin-format imqraw feeds one image argument.
    #[arg(long)]
    stdin_tag: Option<String>,
    /// Zero-based reference image index when an imqraw bundle feeds both image arguments.
    #[arg(long)]
    stdin_reference_index: Option<usize>,
    /// Reference tag when an imqraw bundle feeds both image arguments.
    #[arg(long)]
    stdin_reference_tag: Option<String>,
    /// Zero-based distorted image index when an imqraw bundle feeds both image arguments.
    #[arg(long)]
    stdin_distorted_index: Option<usize>,
    /// Distorted tag when an imqraw bundle feeds both image arguments.
    #[arg(long)]
    stdin_distorted_tag: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum StdinImageFormatArg {
    /// Decode stdin as an encoded image such as PNG/JPEG/WebP.
    Encoded,
    /// Interpret stdin as raw packed pixel bytes.
    Raw,
    /// Decode stdin as an imqraw lossless raw bundle.
    Imqraw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum RawPixelFormatArg {
    Rgb8,
    Rgba8,
    Bgr8,
    Bgra8,
    Luma8,
    Hsv8,
    Hsva8,
    Binary1Lsb,
    Binary1Msb,
    Gray1Lsb,
    Gray1Msb,
    Gray2Lsb,
    Gray2Msb,
    Gray4Lsb,
    Gray4Msb,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PreviewDisplayArg {
    Auto,
    Kitty,
    Sixel,
    Iterm2,
    Ansi,
    None,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PreviewDecodeArg {
    Auto,
    Hardware,
    Cpu,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PreviewFitArg {
    Contain,
    Cover,
    Stretch,
}

#[cfg_attr(not(feature = "preview"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
struct PreviewSize {
    width: u32,
    height: u32,
}

#[cfg(feature = "ffmpeg")]
#[derive(Debug, Args)]
struct VideoCmd {
    /// Reference/original video.
    reference: PathBuf,
    /// Distorted/test video.
    distorted: PathBuf,
    /// Comma-separated metrics.
    #[arg(short, long, default_value = "psnr,ssim,mse,mae,maxae")]
    metrics: String,
    /// Compare every Nth decoded frame.
    #[arg(long, default_value_t = 1)]
    every: u64,
    /// Maximum number of decoded frame pairs to compare.
    #[arg(long)]
    max_frames: Option<u64>,
    /// Force a comparison width. Requires --height.
    #[arg(long)]
    width: Option<u32>,
    /// Force a comparison height. Requires --width.
    #[arg(long)]
    height: Option<u32>,
    /// ffmpeg executable.
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: PathBuf,
    /// ffprobe executable.
    #[arg(long, default_value = "ffprobe")]
    ffprobe: PathBuf,
    /// Video stream index.
    #[arg(long, default_value_t = 0)]
    stream: usize,
    /// Frame alignment: decode-order pairing or VFR-safe timestamp pairing.
    #[arg(long, value_enum, default_value_t = VideoAlignmentArg::Decode)]
    align: VideoAlignmentArg,
    /// Maximum timestamp residual accepted when --align timestamp is selected.
    #[arg(long)]
    max_timestamp_delta: Option<f64>,
    /// Permit one distorted frame to match multiple reference frames.
    #[arg(long)]
    allow_reuse_distorted: bool,
    /// Disable automatic timestamp offset and clock-drift estimation.
    #[arg(long)]
    no_estimate_timestamp_transform: bool,
    /// Explicit distorted-timeline scale for timestamp alignment.
    #[arg(long)]
    timestamp_scale: Option<f64>,
    /// Explicit distorted-timeline offset in seconds for timestamp alignment.
    #[arg(long, allow_hyphen_values = true)]
    timestamp_offset: Option<f64>,
    /// Print JSON instead of a text table.
    #[arg(short, long)]
    json: bool,
    #[command(flatten)]
    output: OutputArgs,
}

#[cfg(feature = "ffmpeg")]
#[derive(Debug, Args)]
struct ExtractFrameCmd {
    /// Input video.
    input: PathBuf,
    /// Zero-based frame index in decode order.
    index: u64,
    /// Output PNG path.
    output: PathBuf,
    /// Optional ffmpeg executable.
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: PathBuf,
    /// Optional ffprobe executable.
    #[arg(long, default_value = "ffprobe")]
    ffprobe: PathBuf,
}

#[cfg(feature = "ffmpeg")]
#[derive(Debug, Args)]
struct ProbeCmd {
    /// Input video.
    input: PathBuf,
    /// Optional ffmpeg executable.
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: PathBuf,
    /// Optional ffprobe executable.
    #[arg(long, default_value = "ffprobe")]
    ffprobe: PathBuf,
    /// Video stream index.
    #[arg(long, default_value_t = 0)]
    stream: usize,
    /// Print JSON.
    #[arg(short, long)]
    json: bool,
    #[command(flatten)]
    output: OutputArgs,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .without_time()
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Compare(cmd) => run_compare(*cmd),
        Command::Image(cmd) => run_image(*cmd),
        Command::Suite(cmd) => run_suite(*cmd),
        #[cfg(feature = "gpu")]
        Command::Gpu(cmd) => run_gpu(cmd),
        Command::Stats(cmd) => run_stats(cmd),
        #[cfg(feature = "ffmpeg")]
        Command::Video(cmd) => run_video(cmd),
        #[cfg(feature = "ffmpeg")]
        Command::ExtractFrame(cmd) => run_extract_frame(cmd),
        #[cfg(feature = "ffmpeg")]
        Command::Probe(cmd) => run_probe(cmd),
        Command::Formats(cmd) => run_formats(cmd),
        Command::Pack(cmd) => run_pack(cmd),
        Command::BundleInfo(cmd) => run_bundle_info(cmd),
        Command::Preview(cmd) => run_preview(*cmd),
        Command::Tui(cmd) => run_tui(cmd),
    }
}

fn run_stats(cmd: StatsCmd) -> Result<()> {
    run_image_stats(
        &cmd.input,
        cmd.histogram_bins,
        cmd.stdin,
        output_format(cmd.json, cmd.output.format),
        &cmd.output,
    )
}

fn run_compare(cmd: CompareCmd) -> Result<()> {
    let Some(distorted) = cmd.distorted.as_ref() else {
        #[cfg(feature = "ffmpeg")]
        if compare_uses_timestamp_options(&cmd) {
            bail!("timestamp alignment options require two local video inputs")
        }
        if cmd.stats {
            return run_image_stats(
                &cmd.reference,
                cmd.histogram_bins,
                cmd.stdin,
                output_format(cmd.json, cmd.output.format),
                &cmd.output,
            );
        }
        bail!("DISTORTED is required unless --stats is used for a single image")
    };

    let reference = parse_input_spec(&cmd.reference, &cmd.remote)?;
    let distorted_input = parse_input_spec(distorted, &cmd.remote)?;
    let reference_is_video = is_video_input(&reference);
    let distorted_is_video = is_video_input(&distorted_input);
    #[cfg(feature = "ffmpeg")]
    if compare_uses_timestamp_options(&cmd)
        && !(reference_is_video
            && distorted_is_video
            && reference.is_local()
            && distorted_input.is_local()
            && cmd.video_frame.is_none()
            && cmd.video_frames.is_none())
    {
        bail!(
            "timestamp alignment options require full comparison of two local videos; omit --video-frame/--video-frames"
        )
    }
    match (reference_is_video, distorted_is_video) {
        (false, false) => {
            let (report, stats) = compare_image_inputs(
                &reference,
                &distorted_input,
                &cmd.metrics,
                cmd.stdin,
                cmd.stats.then_some(cmd.histogram_bins),
                cmd.gate.thresholds(),
                &cmd.remote.options(),
            )?;
            write_sqlite_report(cmd.output.sqlite.as_deref(), SqlReport::Image(&report))?;
            if let Some(stats) = &stats {
                write_sqlite_stats(cmd.output.sqlite.as_deref(), &stats.reference)?;
                write_sqlite_stats(cmd.output.sqlite.as_deref(), &stats.distorted)?;
            }
            if let Some(stats) = stats {
                emit_image_comparison_stats_report(
                    &ImageComparisonStatsReport {
                        comparison: report.clone(),
                        stats,
                    },
                    output_format(cmd.json, cmd.output.format),
                    &cmd.output,
                )?;
            } else {
                emit_comparison_report(
                    &report,
                    output_format(cmd.json, cmd.output.format),
                    &cmd.output,
                )?;
            }
            exit_if_gate_failed(&report);
            Ok(())
        }
        (true, true) => {
            if cmd.stats {
                bail!("--stats currently applies to still images; omit it for video comparison")
            }
            #[cfg(feature = "ffmpeg")]
            {
                let frame_pairs = selected_video_frame_pairs(cmd.video_frame, cmd.video_frames)?;
                if let Some(pairs) = frame_pairs {
                    if cmd.every != 1 || cmd.max_frames.is_some() {
                        bail!("--video-frame(s) cannot be combined with --every or --max-frames")
                    }
                    let report = compare_video_frame_pair_inputs(
                        &reference,
                        &distorted_input,
                        &pairs,
                        &cmd.metrics,
                        FfmpegCliOptions {
                            ffmpeg: cmd.ffmpeg,
                            ffprobe: cmd.ffprobe,
                            stream: cmd.stream,
                            width: cmd.width,
                            height: cmd.height,
                        },
                        &cmd.remote.options(),
                    )?;
                    write_sqlite_report(
                        cmd.output.sqlite.as_deref(),
                        SqlReport::VideoPairs(&report),
                    )?;
                    emit_video_pair_report(
                        &report,
                        output_format(cmd.json, cmd.output.format),
                        &cmd.output,
                    )
                } else if reference.is_local() && distorted_input.is_local() {
                    let (InputSpec::Local(reference_path), InputSpec::Local(distorted_path)) =
                        (reference, distorted_input)
                    else {
                        unreachable!("checked is_local")
                    };
                    run_video(VideoCmd {
                        reference: reference_path,
                        distorted: distorted_path,
                        metrics: cmd.metrics,
                        every: cmd.every,
                        max_frames: cmd.max_frames,
                        width: cmd.width,
                        height: cmd.height,
                        ffmpeg: cmd.ffmpeg,
                        ffprobe: cmd.ffprobe,
                        stream: cmd.stream,
                        align: cmd.align,
                        max_timestamp_delta: cmd.max_timestamp_delta,
                        allow_reuse_distorted: cmd.allow_reuse_distorted,
                        no_estimate_timestamp_transform: cmd.no_estimate_timestamp_transform,
                        timestamp_scale: cmd.timestamp_scale,
                        timestamp_offset: cmd.timestamp_offset,
                        json: cmd.json,
                        output: cmd.output,
                    })
                } else {
                    bail!(
                        "remote video comparison requires --video-frame or --video-frames. Full remote video comparison is intentionally not enabled by default."
                    )
                }
            }
            #[cfg(not(feature = "ffmpeg"))]
            {
                bail!("video support is disabled; rebuild with `--features ffmpeg`")
            }
        }
        _ => bail!("reference and distorted must both be images or both be videos"),
    }
}

#[cfg(feature = "ffmpeg")]
fn compare_uses_timestamp_options(cmd: &CompareCmd) -> bool {
    cmd.align != VideoAlignmentArg::Decode
        || cmd.max_timestamp_delta.is_some()
        || cmd.allow_reuse_distorted
        || cmd.no_estimate_timestamp_transform
        || cmd.timestamp_scale.is_some()
        || cmd.timestamp_offset.is_some()
}

fn run_image(cmd: ImageCmd) -> Result<()> {
    let (report, stats) = compare_image_paths(
        &cmd.reference,
        &cmd.distorted,
        &cmd.metrics,
        cmd.stdin,
        cmd.stats.then_some(cmd.histogram_bins),
        cmd.gate.thresholds(),
    )?;
    write_sqlite_report(cmd.output.sqlite.as_deref(), SqlReport::Image(&report))?;
    if let Some(stats) = stats {
        write_sqlite_stats(cmd.output.sqlite.as_deref(), &stats.reference)?;
        write_sqlite_stats(cmd.output.sqlite.as_deref(), &stats.distorted)?;
        emit_image_comparison_stats_report(
            &ImageComparisonStatsReport {
                comparison: report.clone(),
                stats,
            },
            output_format(cmd.json, cmd.output.format),
            &cmd.output,
        )?;
    } else {
        emit_comparison_report(
            &report,
            output_format(cmd.json, cmd.output.format),
            &cmd.output,
        )?;
    }
    exit_if_gate_failed(&report);
    Ok(())
}

fn run_suite(cmd: SuiteCmd) -> Result<()> {
    let reference = image_crate::load_image_path(&cmd.reference).with_context(|| {
        format!(
            "failed to decode reference image `{}`",
            cmd.reference.display()
        )
    })?;
    let loaded_candidates = cmd
        .candidates
        .iter()
        .map(|path| {
            image_crate::load_image_path(path)
                .with_context(|| format!("failed to decode candidate image `{}`", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let candidates = cmd
        .candidates
        .iter()
        .zip(&loaded_candidates)
        .map(|(path, frame)| imq::ComparisonCandidate::new(path.to_string_lossy(), frame.as_view()))
        .collect::<Vec<_>>();
    let metrics = MetricSet::from_csv(&cmd.metrics)?;
    let (rules, baseline_rules) = parse_suite_rules(&cmd.rules)?;
    let metric_weights = cmd
        .weights
        .iter()
        .map(|spec| {
            let (metric, weight) = spec.split_once('=').ok_or_else(|| {
                anyhow::anyhow!("invalid metric weight `{spec}`; expected metric=value")
            })?;
            let metric = metric.trim();
            if metric.is_empty() {
                bail!("metric weight name cannot be empty");
            }
            let weight = weight
                .trim()
                .parse::<f64>()
                .with_context(|| format!("invalid metric weight `{spec}`"))?;
            Ok::<_, anyhow::Error>((metric.to_string(), weight))
        })
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
    let max_threads = if cmd.jobs == 0 {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    } else {
        cmd.jobs
    };
    let baseline_candidate = match cmd.baseline.as_deref() {
        Some(baseline) => Some(
            resolve_baseline_label(baseline, &cmd.candidates).ok_or_else(|| {
                anyhow::anyhow!("baseline candidate `{}` does not exist", baseline.display())
            })?,
        ),
        None => None,
    };
    let options = imq::ComparisonSuiteOptions {
        primary_metric: cmd.primary_metric,
        baseline_candidate,
        metric_weights,
        rules,
        baseline_rules,
        failure_policy: if cmd.fail_fast {
            imq::CandidateFailurePolicy::FailFast
        } else {
            imq::CandidateFailurePolicy::Continue
        },
        max_threads,
        score_tolerance: imq::ScoreTolerance::new(cmd.rank_abs_tolerance, cmd.rank_rel_tolerance)?,
    };
    let report = imq::compare_candidate_suite(
        Some(cmd.reference.to_string_lossy().into_owned()),
        &reference.as_view(),
        &candidates,
        &metrics,
        &options,
    )?;
    emit_suite_report(
        &report,
        output_format(cmd.json, cmd.format),
        cmd.output.as_deref(),
    )?;
    if !report.passed {
        std::process::exit(1);
    }
    Ok(())
}

/// Splits suite rules into absolute and baseline-relative rules.
///
/// Routing inspects only the right-hand side of the operator, so metric names
/// that contain `baseline`, such as `baseline_ssim>=0.9`, stay absolute rules.
fn parse_suite_rules(
    rules: &[String],
) -> Result<(
    Vec<imq::MetricThresholdRule>,
    Vec<imq::BaselineThresholdRule>,
)> {
    let mut absolute = Vec::new();
    let mut baseline = Vec::new();
    for rule in rules {
        if rule_threshold_uses_baseline(rule) {
            baseline.push(imq::BaselineThresholdRule::parse(rule)?);
        } else {
            absolute.push(imq::MetricThresholdRule::parse(rule)?);
        }
    }
    Ok((absolute, baseline))
}

fn rule_threshold_uses_baseline(rule: &str) -> bool {
    ["<=", ">=", "<", ">"]
        .into_iter()
        .find_map(|token| rule.split_once(token).map(|(_, rhs)| rhs))
        .is_some_and(|rhs| rhs.trim().starts_with("baseline"))
}

/// Resolves the `--baseline` path to a candidate label.
///
/// Literal matches win; otherwise both sides are resolved against the
/// filesystem (`.` and `..` included), so `a.png`, `./a.png`, and
/// `sub/../a.png` select the same candidate even across symlinked parents.
fn resolve_baseline_label(baseline: &Path, candidates: &[PathBuf]) -> Option<String> {
    candidates
        .iter()
        .find(|candidate| candidate.as_path() == baseline)
        .map(|candidate| candidate.to_string_lossy().into_owned())
        .or_else(|| {
            let baseline_key = resolved_path_key(baseline)?;
            candidates
                .iter()
                .find(|candidate| {
                    resolved_path_key(candidate).is_some_and(|key| key == baseline_key)
                })
                .map(|candidate| candidate.to_string_lossy().into_owned())
        })
}

fn resolved_path_key(path: &Path) -> Option<Vec<std::ffi::OsString>> {
    let resolved = std::fs::canonicalize(path)
        .or_else(|_| std::path::absolute(path))
        .ok()?;
    let mut components: Vec<std::ffi::OsString> = Vec::new();
    for component in resolved.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                components.pop();
            }
            other => components.push(other.as_os_str().to_os_string()),
        }
    }
    Some(components)
}

#[cfg(feature = "gpu")]
fn run_gpu(cmd: GpuCmd) -> Result<()> {
    match cmd.command {
        GpuSubcommand::Info(cmd) => run_gpu_info(cmd),
        GpuSubcommand::Compare(cmd) => run_gpu_compare(cmd),
    }
}

#[cfg(feature = "gpu")]
fn run_gpu_info(cmd: GpuInfoCmd) -> Result<()> {
    let context = imq::gpu::GpuContext::new_with_options(gpu_context_options(&cmd.adapter))?;
    let capabilities = context.capabilities();
    let content = match cmd.format {
        OutputFormatArg::Text => render_gpu_capabilities(capabilities),
        OutputFormatArg::Json => serde_json::to_string_pretty(capabilities)?,
        OutputFormatArg::Yaml => serde_yaml_ng::to_string(capabilities)?,
        OutputFormatArg::Toml => toml::to_string_pretty(capabilities)?,
        OutputFormatArg::Csv => csv_string([capabilities])?,
    };
    write_output(cmd.output.as_deref(), &content)
}

#[cfg(feature = "gpu")]
fn run_gpu_compare(cmd: GpuCompareCmd) -> Result<()> {
    let (error_metrics, include_wssim) = parse_gpu_metrics(&cmd.metrics)?;
    let options = imq::gpu::GpuComparatorOptions {
        context: gpu_context_options(&cmd.adapter),
        fallback: match cmd.fallback {
            GpuFallbackArg::Never => imq::gpu::GpuFallbackPolicy::Never,
            GpuFallbackArg::Initialization => imq::gpu::GpuFallbackPolicy::InitializationOnly,
            GpuFallbackArg::Any => imq::gpu::GpuFallbackPolicy::AnyGpuError,
        },
    };
    let comparator = imq::gpu::GpuComparator::new_with_options(options)?;
    let reference = image_crate::load_image_path(&cmd.reference).with_context(|| {
        format!(
            "failed to decode GPU reference image `{}`",
            cmd.reference.display()
        )
    })?;
    let loaded_candidates = cmd
        .candidates
        .iter()
        .map(|candidate_path| {
            image_crate::load_image_path(candidate_path).with_context(|| {
                format!(
                    "failed to decode GPU candidate image `{}`",
                    candidate_path.display()
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let reference_view = reference.as_view();
    let candidate_views = loaded_candidates
        .iter()
        .map(imq::FrameOwned::as_view)
        .collect::<Vec<_>>();
    let candidate_refs = candidate_views.iter().collect::<Vec<_>>();
    let error_results = if error_metrics.is_empty() {
        vec![None; candidate_refs.len()]
    } else {
        comparator
            .compare_many_rgba8_in_domain(
                &reference_view,
                &candidate_refs,
                &error_metrics,
                match cmd.domain {
                    GpuDomainArg::Color => imq::gpu::GpuErrorDomain::Color,
                    GpuDomainArg::All => imq::gpu::GpuErrorDomain::All,
                },
            )?
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>()
    };
    let wssim_options = imq::gpu::GpuWindowedSsimOptions {
        window_width: cmd.window,
        window_height: cmd.window,
        stride_x: cmd.window_stride,
        stride_y: cmd.window_stride,
    };
    let comparisons = cmd
        .candidates
        .iter()
        .zip(&candidate_views)
        .zip(error_results)
        .map(|((candidate_path, candidate_view), error_result)| {
            let wssim_result = include_wssim
                .then(|| {
                    comparator.compare_windowed_ssim_rgba8(
                        &reference_view,
                        candidate_view,
                        wssim_options,
                    )
                })
                .transpose()?;
            Ok::<_, anyhow::Error>(GpuCliComparison {
                candidate: candidate_path.to_string_lossy().into_owned(),
                error_result,
                wssim_result,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let report = GpuCliReport {
        capabilities: comparator.capabilities().cloned(),
        initialization_error: comparator.initialization_error().map(str::to_string),
        reference: cmd.reference.to_string_lossy().into_owned(),
        comparisons,
    };
    emit_gpu_report(&report, cmd.format, cmd.output.as_deref())
}

#[cfg(feature = "gpu")]
fn gpu_context_options(adapter: &GpuAdapterArgs) -> imq::gpu::GpuContextOptions {
    imq::gpu::GpuContextOptions {
        power_preference: match adapter.power {
            GpuPowerArg::None => imq::gpu::GpuPowerPreference::None,
            GpuPowerArg::Low => imq::gpu::GpuPowerPreference::LowPower,
            GpuPowerArg::High => imq::gpu::GpuPowerPreference::HighPerformance,
        },
        force_fallback_adapter: adapter.fallback_adapter,
    }
}

#[cfg(feature = "gpu")]
fn parse_gpu_metrics(input: &str) -> Result<(Vec<imq::gpu::GpuErrorMetric>, bool)> {
    let mut error_metrics = Vec::new();
    let mut include_wssim = false;
    for metric in input
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        match metric.to_ascii_lowercase().as_str() {
            "mse" | "mse:all" => error_metrics.push(imq::gpu::GpuErrorMetric::Mse),
            "rmse" | "rmse:all" => error_metrics.push(imq::gpu::GpuErrorMetric::Rmse),
            "psnr" | "psnr:all" => error_metrics.push(imq::gpu::GpuErrorMetric::Psnr),
            "mae" | "mae:all" => error_metrics.push(imq::gpu::GpuErrorMetric::Mae),
            "maxae" | "max_ae" | "max-error" | "maxae:all" => {
                error_metrics.push(imq::gpu::GpuErrorMetric::MaxAbsoluteError);
            }
            "wssim" | "windowed-ssim" | "windowed_ssim" => include_wssim = true,
            _ => bail!("unsupported GPU metric `{metric}`; expected mse,rmse,psnr,mae,maxae,wssim"),
        }
    }
    if error_metrics.is_empty() && !include_wssim {
        bail!("at least one GPU metric is required");
    }
    Ok((error_metrics, include_wssim))
}

fn compare_image_paths(
    reference_path: &Path,
    distorted_path: &Path,
    metrics_csv: &str,
    stdin: StdinImageArgs,
    histogram_bins: Option<usize>,
    thresholds: ComparisonThresholds,
) -> Result<(ComparisonReport, Option<ImageComparisonStats>)> {
    let reference = InputSpec::Local(reference_path.to_path_buf());
    let distorted = InputSpec::Local(distorted_path.to_path_buf());
    compare_image_inputs(
        &reference,
        &distorted,
        metrics_csv,
        stdin,
        histogram_bins,
        thresholds,
        &RemoteOptions::default(),
    )
}

fn compare_image_inputs(
    reference_input: &InputSpec,
    distorted_input: &InputSpec,
    metrics_csv: &str,
    stdin: StdinImageArgs,
    histogram_bins: Option<usize>,
    thresholds: ComparisonThresholds,
    remote: &RemoteOptions,
) -> Result<(ComparisonReport, Option<ImageComparisonStats>)> {
    let (reference, distorted) = if matches!(reference_input, InputSpec::Stdin)
        && matches!(distorted_input, InputSpec::Stdin)
        && stdin.stdin_format == StdinImageFormatArg::Imqraw
    {
        load_imqraw_image_pair_from_stdin(&stdin)?
    } else {
        reject_double_stdin_specs(reference_input, Some(distorted_input))?;
        let reference =
            load_image_input_spec(reference_input, &stdin, remote).with_context(|| {
                format!(
                    "failed to decode reference image `{}`",
                    reference_input.display_label()
                )
            })?;
        let distorted =
            load_image_input_spec(distorted_input, &stdin, remote).with_context(|| {
                format!(
                    "failed to decode distorted image `{}`",
                    distorted_input.display_label()
                )
            })?;
        (reference, distorted)
    };
    let reference_label = reference.display_label.clone();
    let distorted_label = distorted.display_label.clone();
    let metrics = MetricSet::from_csv(metrics_csv)?;
    let outputs = metrics.compare(&reference.frame.as_view(), &distorted.frame.as_view())?;
    let alpha = imq::alpha_diagnostics(&reference.frame.as_view(), &distorted.frame.as_view())?;
    let mut report = ComparisonReport::new(
        reference.frame.dimensions(),
        reference.frame.format(),
        distorted.frame.format(),
        outputs,
    )
    .with_labels(reference_label.clone(), distorted_label.clone())
    .with_inputs(reference.input.clone(), distorted.input.clone())
    .with_alpha(alpha);
    if thresholds_have_any(&thresholds) {
        let gate = evaluate_gate(&report, thresholds)?;
        report = report.with_gate(gate);
    }
    let stats = histogram_bins
        .map(|histogram_bins| {
            Ok::<_, anyhow::Error>(ImageComparisonStats {
                reference: image_stats_for_frame(
                    reference_label.clone(),
                    &reference.frame,
                    histogram_bins,
                )?,
                distorted: image_stats_for_frame(
                    distorted_label.clone(),
                    &distorted.frame,
                    histogram_bins,
                )?,
            })
        })
        .transpose()?;
    Ok((report, stats))
}

fn thresholds_have_any(thresholds: &ComparisonThresholds) -> bool {
    thresholds.selected_metric.is_some()
        || thresholds.fail_under.is_some()
        || thresholds.max_selected_channel_delta.is_some()
        || thresholds.max_alpha_delta.is_some()
        || thresholds.max_alpha_mismatches.is_some()
        || thresholds.max_alpha_mismatches_beyond_one_lsb.is_some()
}

fn evaluate_gate(
    report: &ComparisonReport,
    thresholds: ComparisonThresholds,
) -> Result<ComparisonGateReport> {
    if thresholds
        .fail_under
        .is_some_and(|minimum| !minimum.is_finite())
    {
        bail!("--fail-under must be finite");
    }
    if thresholds
        .max_selected_channel_delta
        .is_some_and(|maximum| !maximum.is_finite())
    {
        bail!("--max-selected-channel-delta must be finite");
    }
    let selected = selected_metric(report, thresholds.selected_metric.as_deref())?;
    let mut failures = Vec::new();
    if let Some(minimum) = thresholds.fail_under
        && selected.score < minimum
    {
        failures.push(format!(
            "{} score {:.8} is below --fail-under {:.8}",
            selected.name, selected.score, minimum
        ));
    }
    if let Some(max_delta) = thresholds.max_selected_channel_delta {
        let actual = selected
            .details
            .get("max_channel_delta_code")
            .copied()
            .unwrap_or_else(|| {
                selected
                    .details
                    .get("max_channel_delta")
                    .copied()
                    .unwrap_or(0.0)
                    * 255.0
            });
        if actual > max_delta {
            failures.push(format!(
                "{} max channel delta {:.8} exceeds --max-selected-channel-delta {:.8}",
                selected.name, actual, max_delta
            ));
        }
    }
    if let Some(alpha) = report.alpha {
        if let Some(max_delta) = thresholds.max_alpha_delta
            && alpha.max_delta > max_delta
        {
            failures.push(format!(
                "alpha max delta {} exceeds --max-alpha-delta {}",
                alpha.max_delta, max_delta
            ));
        }
        if let Some(max_mismatches) = thresholds.max_alpha_mismatches
            && alpha.mismatch_count > max_mismatches
        {
            failures.push(format!(
                "alpha mismatches {} exceeds --max-alpha-mismatches {}",
                alpha.mismatch_count, max_mismatches
            ));
        }
        if let Some(max_mismatches) = thresholds.max_alpha_mismatches_beyond_one_lsb
            && alpha.mismatches_beyond_one_lsb > max_mismatches
        {
            failures.push(format!(
                "alpha mismatches beyond one LSB {} exceeds --max-alpha-mismatches-beyond-one-lsb {}",
                alpha.mismatches_beyond_one_lsb, max_mismatches
            ));
        }
    }
    Ok(ComparisonGateReport {
        thresholds,
        passed: failures.is_empty(),
        failures,
    })
}

fn selected_metric<'a>(
    report: &'a ComparisonReport,
    requested: Option<&str>,
) -> Result<&'a MetricOutput> {
    let Some(requested) = requested.map(str::trim).filter(|value| !value.is_empty()) else {
        return report
            .metrics
            .first()
            .ok_or_else(|| anyhow::anyhow!("cannot apply threshold gate without metrics"));
    };
    report
        .metrics
        .iter()
        .find(|metric| {
            metric.name == requested
                || metric
                    .name
                    .strip_prefix("psnr:")
                    .is_some_and(|domain| domain == requested)
                || metric
                    .name
                    .split_once(':')
                    .is_some_and(|(_, domain)| domain == requested)
        })
        .ok_or_else(|| anyhow::anyhow!("selected metric `{requested}` was not computed"))
}

fn exit_if_gate_failed(report: &ComparisonReport) {
    if report.gate.as_ref().is_some_and(|gate| !gate.passed) {
        std::process::exit(1);
    }
}

fn run_image_stats(
    input: &Path,
    histogram_bins: usize,
    stdin: StdinImageArgs,
    format: OutputFormatArg,
    output: &OutputArgs,
) -> Result<()> {
    let report = image_stats_for_path(input, histogram_bins, stdin)?;
    write_sqlite_stats(output.sqlite.as_deref(), &report)?;
    emit_stats_report(&report, format, output)
}

fn image_stats_for_path(
    input: &Path,
    histogram_bins: usize,
    stdin: StdinImageArgs,
) -> Result<ImageStatsReport> {
    let image = load_image_input(input, &stdin)
        .with_context(|| format!("failed to decode image `{}`", image_input_label(input)))?;
    image_stats_for_frame(image_input_label(input), &image, histogram_bins)
}

fn image_stats_for_frame(
    input: String,
    image: &imq::FrameOwned,
    histogram_bins: usize,
) -> Result<ImageStatsReport> {
    let stats = imq::image_statistics(
        &image.as_view(),
        imq::ImageStatisticsOptions { histogram_bins },
    )?;
    Ok(ImageStatsReport { input, stats })
}

fn load_image_input_spec(
    input: &InputSpec,
    stdin: &StdinImageArgs,
    remote: &RemoteOptions,
) -> Result<LoadedImage> {
    match input {
        InputSpec::Local(path) => {
            let frame = image_crate::load_image_path(path)?;
            Ok(LoadedImage::from_input(input, frame))
        }
        InputSpec::Stdin => {
            let frame = load_image_input(Path::new("-"), stdin)?;
            Ok(LoadedImage::from_input(input, frame))
        }
        InputSpec::Ssh(spec) => {
            let frame = load_remote_image(spec, remote)?;
            Ok(LoadedImage::from_input(input, frame))
        }
    }
}

fn load_image_input(path: &Path, stdin: &StdinImageArgs) -> Result<imq::FrameOwned> {
    if !is_stdin_path(path) {
        return Ok(image_crate::load_image_path(path)?);
    }

    let mut bytes = Vec::new();
    std::io::stdin()
        .read_to_end(&mut bytes)
        .with_context(|| "failed to read image bytes from stdin")?;
    match stdin.stdin_format {
        StdinImageFormatArg::Encoded => Ok(image_crate::decode_image_bytes(&bytes)?),
        StdinImageFormatArg::Imqraw => {
            let bundle = imq::decode_imqraw_bundle(&bytes)?;
            Ok(bundle.select(&single_stdin_selector(stdin))?.frame.clone())
        }
        StdinImageFormatArg::Raw => {
            let width = stdin
                .raw_width
                .ok_or_else(|| anyhow::anyhow!("--raw-width is required for raw stdin"))?;
            let height = stdin
                .raw_height
                .ok_or_else(|| anyhow::anyhow!("--raw-height is required for raw stdin"))?;
            decode_raw_stdin_frame(
                bytes,
                width,
                height,
                stdin.raw_pixel_format,
                stdin.raw_stride,
            )
        }
    }
}

fn decode_raw_stdin_frame(
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    pixel_format: RawPixelFormatArg,
    stride: Option<usize>,
) -> Result<imq::FrameOwned> {
    if let Some(format) = pixel_format.bit_packed_gray_format()? {
        return Ok(imq::adapters::bit_packed_gray::decode_bit_packed_gray(
            &bytes, width, height, format, stride,
        )?);
    }
    if let Some(format) = pixel_format.binary_pixel_format() {
        let width = usize::try_from(width).map_err(|_| anyhow::anyhow!("width overflows usize"))?;
        let height =
            usize::try_from(height).map_err(|_| anyhow::anyhow!("height overflows usize"))?;
        let tight_stride = width.div_ceil(8);
        let stride = stride.unwrap_or(tight_stride);
        if stride < tight_stride {
            bail!("raw stride {stride} is smaller than tight binary row size {tight_stride}");
        }
        return Ok(imq::FrameOwned::new(
            imq::Dimensions::new(width as u32, height as u32)?,
            imq::FormatSpec::new(format),
            vec![imq::OwnedPlane::new(bytes, stride)],
        )?);
    }
    let pixel_format = pixel_format
        .pixel_format()
        .ok_or_else(|| anyhow::anyhow!("raw pixel format is not a packed imq pixel format"))?;
    if let Some(stride) = stride {
        Ok(imq::FrameOwned::packed(
            bytes,
            width,
            height,
            pixel_format,
            stride,
        )?)
    } else {
        Ok(imq::FrameOwned::packed_tight(
            bytes,
            width,
            height,
            pixel_format,
        )?)
    }
}

struct LoadedImage {
    frame: imq::FrameOwned,
    display_label: String,
    input: ComparisonInput,
}

impl LoadedImage {
    fn from_input(input: &InputSpec, frame: imq::FrameOwned) -> Self {
        let display_label = input.display_label();
        let report_input = match input {
            InputSpec::Local(path) => ComparisonInput::path(path.display().to_string()),
            InputSpec::Stdin => ComparisonInput::label("stdin"),
            InputSpec::Ssh(spec) => ComparisonInput {
                path: Some(spec.uri()),
                ..ComparisonInput::default()
            },
        };
        Self {
            frame,
            display_label,
            input: report_input,
        }
    }
}

fn load_imqraw_image_pair_from_stdin(stdin: &StdinImageArgs) -> Result<(LoadedImage, LoadedImage)> {
    let mut bytes = Vec::new();
    std::io::stdin()
        .read_to_end(&mut bytes)
        .with_context(|| "failed to read imqraw bundle from stdin")?;
    let bundle = imq::decode_imqraw_bundle(&bytes)?;
    let (reference_selector, distorted_selector) = pair_stdin_selectors(stdin);
    let reference = bundle.select(&reference_selector)?;
    let distorted = bundle.select(&distorted_selector)?;
    Ok((
        loaded_imqraw_image(reference, &reference_selector),
        loaded_imqraw_image(distorted, &distorted_selector),
    ))
}

fn loaded_imqraw_image(
    record: &imq::RawImageRecord,
    selector: &imq::RawImageSelector,
) -> LoadedImage {
    LoadedImage {
        frame: record.frame.clone(),
        display_label: imqraw_record_label(record, selector),
        input: imqraw_comparison_input(record, selector),
    }
}

fn single_stdin_selector(stdin: &StdinImageArgs) -> imq::RawImageSelector {
    stdin.stdin_tag.as_ref().map_or_else(
        || imq::RawImageSelector::Index(stdin.stdin_index.unwrap_or(0)),
        |tag| imq::RawImageSelector::Tag(tag.clone()),
    )
}

fn pair_stdin_selectors(stdin: &StdinImageArgs) -> (imq::RawImageSelector, imq::RawImageSelector) {
    let reference = stdin.stdin_reference_tag.as_ref().map_or_else(
        || {
            imq::RawImageSelector::Index(
                stdin
                    .stdin_reference_index
                    .or(stdin.stdin_index)
                    .unwrap_or(0),
            )
        },
        |tag| imq::RawImageSelector::Tag(tag.clone()),
    );
    let distorted = stdin.stdin_distorted_tag.as_ref().map_or_else(
        || imq::RawImageSelector::Index(stdin.stdin_distorted_index.unwrap_or(1)),
        |tag| imq::RawImageSelector::Tag(tag.clone()),
    );
    (reference, distorted)
}

fn imqraw_record_label(record: &imq::RawImageRecord, selector: &imq::RawImageSelector) -> String {
    record.label.clone().unwrap_or_else(|| match selector {
        imq::RawImageSelector::Index(index) => format!("imqraw[{index}]"),
        imq::RawImageSelector::Tag(tag) => format!("imqraw:{tag}"),
    })
}

fn imqraw_comparison_input(
    record: &imq::RawImageRecord,
    selector: &imq::RawImageSelector,
) -> ComparisonInput {
    match selector {
        imq::RawImageSelector::Index(index) => {
            ComparisonInput::imqraw_index(*index, record.label.clone())
        }
        imq::RawImageSelector::Tag(tag) => {
            ComparisonInput::imqraw_tag(tag.clone(), record.label.clone())
        }
    }
}

fn parse_input_spec(path: &Path, remote: &RemoteArgs) -> Result<InputSpec> {
    let raw = path.to_string_lossy();
    let input = InputSpec::parse(raw.as_ref())?;
    if matches!(input, InputSpec::Local(_))
        && remote.ssh.is_some()
        && !is_stdin_path(path)
        && path.is_absolute()
    {
        return Ok(InputSpec::Ssh(SshInput {
            user: None,
            host: remote.ssh.clone().expect("checked is_some"),
            port: None,
            path: raw.into_owned(),
        }));
    }
    Ok(input)
}

fn is_video_input(input: &InputSpec) -> bool {
    input
        .extension()
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "avi" | "m4v" | "mkv" | "mov" | "mp4" | "mpeg" | "mpg" | "webm" | "wmv"
            )
        })
        .unwrap_or(false)
}

fn load_remote_image(spec: &SshInput, remote: &RemoteOptions) -> Result<imq::FrameOwned> {
    match remote.transfer {
        RemoteTransferMode::Stream => {
            let bytes = ssh_capture_stdout(
                spec,
                &format!("cat -- {}", imq::shell_quote_posix(&spec.path)),
                remote,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "remote stream failed for {}.\ncopy fallback is disabled by default.\nUse --remote-transfer copy-input to copy the input explicitly.\n{error}",
                    spec.uri()
                )
            })?;
            Ok(image_crate::decode_image_bytes(&bytes)?)
        }
        RemoteTransferMode::CopyInput => {
            let temp = scp_to_temp(spec, remote, "input")?;
            Ok(image_crate::load_image_path(&temp.path)?)
        }
        RemoteTransferMode::CopyFrame | RemoteTransferMode::CopySource => {
            bail!("remote transfer mode is not valid for still image input")
        }
    }
}

fn ssh_capture_stdout(
    spec: &SshInput,
    remote_command: &str,
    remote: &RemoteOptions,
) -> Result<Vec<u8>> {
    spec.validate()?;
    let mut command = ProcessCommand::new(&remote.ssh);
    command.arg("-T");
    if remote.batch_mode {
        command.args(["-o", "BatchMode=yes"]);
    }
    if let Some(timeout) = remote.connect_timeout_seconds {
        command.args(["-o", &format!("ConnectTimeout={timeout}")]);
    }
    if let Some(port) = spec.port {
        command.args(["-p", &port.to_string()]);
    }
    command.arg(spec.ssh_target());
    command.arg(remote_command);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let child = command
        .spawn()
        .with_context(|| format!("failed to start `{}`", remote.ssh.display()))?;
    let mut child = CapturedChild::new(child);
    let mut stdout = child
        .child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture ssh stdout"))?;
    let mut bytes = Vec::new();
    let read_limit = u64::try_from(remote.max_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let read_result = stdout.by_ref().take(read_limit).read_to_end(&mut bytes);
    drop(stdout);
    if let Err(error) = read_result {
        child.terminate();
        let stderr = child.take_stderr();
        bail!(
            "failed to read ssh stdout for {}: {error}\n{}",
            spec.uri(),
            stderr.trim()
        )
    }
    let exceeded_limit = bytes.len() > remote.max_bytes;
    if exceeded_limit {
        child.terminate();
        let stderr = child.take_stderr();
        bail!(
            "remote stream exceeded --remote-max-bytes for {}\n{}",
            spec.uri(),
            stderr.trim()
        )
    }
    let status = match child.wait() {
        Ok(status) => status,
        Err(error) => {
            child.terminate();
            let stderr = child.take_stderr();
            bail!(
                "failed to wait for ssh command for {}: {error}\n{}",
                spec.uri(),
                stderr.trim()
            )
        }
    };
    let stderr = child.take_stderr();
    if !status.success() {
        bail!(
            "ssh command failed for {}: {status}\n{}",
            spec.uri(),
            stderr.trim()
        )
    }
    Ok(bytes)
}

struct CapturedChild {
    child: Child,
    stderr_reader: Option<JoinHandle<String>>,
    reaped: bool,
}

impl CapturedChild {
    fn new(mut child: Child) -> Self {
        let stderr_reader = child.stderr.take().map(|stderr| {
            std::thread::spawn(move || read_bounded_stderr_tail(stderr, REMOTE_STDERR_TAIL_BYTES))
        });
        Self {
            child,
            stderr_reader,
            reaped: false,
        }
    }

    fn wait(&mut self) -> std::io::Result<ExitStatus> {
        let status = self.child.wait()?;
        self.reaped = true;
        Ok(status)
    }

    fn terminate(&mut self) {
        if !self.reaped {
            let _ = self.child.kill();
            let _ = self.child.wait();
            self.reaped = true;
        }
    }

    fn take_stderr(&mut self) -> String {
        self.stderr_reader
            .take()
            .and_then(|reader| reader.join().ok())
            .unwrap_or_default()
    }
}

impl Drop for CapturedChild {
    fn drop(&mut self) {
        self.terminate();
        if let Some(reader) = self.stderr_reader.take() {
            let _ = reader.join();
        }
    }
}

fn read_bounded_stderr_tail(mut reader: impl Read, max_bytes: usize) -> String {
    let mut tail = Vec::with_capacity(max_bytes.min(8192));
    let mut chunk = [0u8; 8192];
    let mut truncated = false;
    let mut read_error = None;
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) if max_bytes == 0 => truncated |= read != 0,
            Ok(read) if read >= max_bytes => {
                truncated |= !tail.is_empty() || read > max_bytes;
                tail.clear();
                tail.extend_from_slice(&chunk[read - max_bytes..read]);
            }
            Ok(read) => {
                let overflow = tail.len().saturating_add(read).saturating_sub(max_bytes);
                if overflow != 0 {
                    truncated = true;
                    tail.copy_within(overflow.., 0);
                    tail.truncate(tail.len() - overflow);
                }
                tail.extend_from_slice(&chunk[..read]);
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => {
                read_error = Some(error);
                break;
            }
        }
    }
    let mut output = String::new();
    if truncated {
        output.push_str("[stderr truncated; showing tail]\n");
    }
    if let Some(error) = read_error {
        output.push_str(&format!("[failed to read stderr: {error}]\n"));
    }
    output.push_str(&String::from_utf8_lossy(&tail));
    output
}

#[derive(Debug)]
struct ManagedTempPath {
    path: PathBuf,
    directory: PathBuf,
    keep: bool,
}

impl Drop for ManagedTempPath {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }
}

fn scp_to_temp(spec: &SshInput, remote: &RemoteOptions, kind: &str) -> Result<ManagedTempPath> {
    scp_path_to_temp(spec, &spec.path, remote, kind)
}

fn scp_path_to_temp(
    spec: &SshInput,
    remote_path: &str,
    remote: &RemoteOptions,
    kind: &str,
) -> Result<ManagedTempPath> {
    spec.validate()?;
    let base_dir = remote.copy_dir.clone().unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&base_dir)?;
    let directory = create_private_temp_dir(&base_dir, kind)?;
    let path = directory.join("payload");
    let mut temp = ManagedTempPath {
        path,
        directory,
        keep: false,
    };
    let mut command = ProcessCommand::new(&remote.scp);
    if let Some(port) = spec.port {
        command.args(["-P", &port.to_string()]);
    }
    command.arg(format!(
        "{}:{}",
        spec.scp_target(),
        imq::shell_quote_posix(remote_path)
    ));
    command.arg(&temp.path);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let child = command
        .spawn()
        .with_context(|| format!("failed to start `{}`", remote.scp.display()))?;
    let mut child = CapturedChild::new(child);
    let status = match child.wait() {
        Ok(status) => status,
        Err(error) => {
            child.terminate();
            let stderr = child.take_stderr();
            bail!(
                "failed to wait for scp from {}: {error}\n{}",
                spec.uri(),
                stderr.trim()
            )
        }
    };
    let stderr = child.take_stderr();
    if !status.success() {
        bail!(
            "scp failed for {}: {}\n{}",
            spec.uri(),
            status,
            stderr.trim()
        );
    }
    temp.keep = remote.keep_temp;
    if remote.keep_temp {
        eprintln!("kept remote copy temp: {}", temp.path.display());
    }
    Ok(temp)
}

fn create_private_temp_dir(base: &Path, kind: &str) -> Result<PathBuf> {
    let safe_kind = kind
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    for _ in 0..128 {
        let counter = REMOTE_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let nonce = RandomState::new().hash_one((std::process::id(), counter, timestamp));
        let path = base.join(format!("imq-{safe_kind}-{nonce:016x}"));
        #[cfg(unix)]
        let created = {
            use std::os::unix::fs::DirBuilderExt;
            let mut builder = std::fs::DirBuilder::new();
            builder.mode(0o700).create(&path)
        };
        #[cfg(not(unix))]
        let created = std::fs::create_dir(&path);
        match created {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    bail!("failed to allocate a unique private remote-copy directory")
}

fn reject_double_stdin_specs(first: &InputSpec, second: Option<&InputSpec>) -> Result<()> {
    if matches!(first, InputSpec::Stdin)
        && second.is_some_and(|input| matches!(input, InputSpec::Stdin))
    {
        bail!("stdin input `-` can only be used for one image argument")
    }
    Ok(())
}

fn is_stdin_path(path: &Path) -> bool {
    path == Path::new("-")
}

fn image_input_label(path: &Path) -> String {
    if is_stdin_path(path) {
        "stdin".to_string()
    } else {
        path.display().to_string()
    }
}

impl RawPixelFormatArg {
    fn pixel_format(self) -> Option<imq::PixelFormat> {
        match self {
            Self::Rgb8 => Some(imq::PixelFormat::Rgb8),
            Self::Rgba8 => Some(imq::PixelFormat::Rgba8),
            Self::Bgr8 => Some(imq::PixelFormat::Bgr8),
            Self::Bgra8 => Some(imq::PixelFormat::Bgra8),
            Self::Luma8 => Some(imq::PixelFormat::Luma8),
            Self::Hsv8 => Some(imq::PixelFormat::Hsv8),
            Self::Hsva8 => Some(imq::PixelFormat::Hsva8),
            Self::Binary1Lsb | Self::Binary1Msb => None,
            Self::Gray1Lsb
            | Self::Gray1Msb
            | Self::Gray2Lsb
            | Self::Gray2Msb
            | Self::Gray4Lsb
            | Self::Gray4Msb => None,
        }
    }

    fn binary_pixel_format(self) -> Option<imq::PixelFormat> {
        match self {
            Self::Binary1Lsb => Some(imq::PixelFormat::Binary1Lsb),
            Self::Binary1Msb => Some(imq::PixelFormat::Binary1Msb),
            _ => None,
        }
    }

    fn bit_packed_gray_format(self) -> Result<Option<BitPackedGrayFormat>> {
        let format = match self {
            Self::Gray1Lsb => Some(BitPackedGrayFormat::new(1, BitOrder::LsbFirst)?),
            Self::Gray1Msb => Some(BitPackedGrayFormat::new(1, BitOrder::MsbFirst)?),
            Self::Gray2Lsb => Some(BitPackedGrayFormat::new(2, BitOrder::LsbFirst)?),
            Self::Gray2Msb => Some(BitPackedGrayFormat::new(2, BitOrder::MsbFirst)?),
            Self::Gray4Lsb => Some(BitPackedGrayFormat::new(4, BitOrder::LsbFirst)?),
            Self::Gray4Msb => Some(BitPackedGrayFormat::new(4, BitOrder::MsbFirst)?),
            Self::Rgb8
            | Self::Rgba8
            | Self::Bgr8
            | Self::Bgra8
            | Self::Luma8
            | Self::Hsv8
            | Self::Hsva8
            | Self::Binary1Lsb
            | Self::Binary1Msb => None,
        };
        Ok(format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sqlite_test_path(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("imq-{label}-{}-{nonce}.sqlite", std::process::id()))
    }

    fn plane_bytes(frame: &imq::FrameOwned) -> &[u8] {
        &frame.owned_planes()[0].data
    }

    #[test]
    fn suite_rules_route_by_threshold_side() {
        let rules = [
            "psnr>=35".to_string(),
            "psnr>=baseline-0.5".to_string(),
            "baseline_ssim>=0.9".to_string(),
            "mse<=baseline*1.1".to_string(),
        ];

        let (absolute, baseline) = parse_suite_rules(&rules).unwrap();

        assert_eq!(
            absolute
                .iter()
                .map(|rule| rule.metric.as_str())
                .collect::<Vec<_>>(),
            ["psnr", "baseline_ssim"]
        );
        assert_eq!(absolute[0].threshold, 35.0);
        assert_eq!(absolute[1].threshold, 0.9);
        assert_eq!(
            baseline
                .iter()
                .map(|rule| rule.metric.as_str())
                .collect::<Vec<_>>(),
            ["psnr", "mse"]
        );
        let scaled =
            imq::BaselineThresholdRule::new("mse", imq::GateOperator::LessOrEqual, 1.1, 0.0);
        assert_eq!(baseline[1], scaled);
    }

    #[test]
    fn suite_rules_reject_invalid_thresholds() {
        assert!(parse_suite_rules(&["psnr>=abc".to_string()]).is_err());
        assert!(parse_suite_rules(&["psnr>=baseline=oops".to_string()]).is_err());
    }

    #[test]
    fn resolves_baseline_flag_against_candidate_labels() {
        let candidates = vec![
            PathBuf::from("ref.png"),
            PathBuf::from("./a.png"),
            PathBuf::from("sub/b.png"),
        ];

        assert_eq!(
            resolve_baseline_label(Path::new("sub/b.png"), &candidates).as_deref(),
            Some("sub/b.png")
        );
        assert_eq!(
            resolve_baseline_label(Path::new("./a.png"), &candidates).as_deref(),
            Some("./a.png")
        );
        assert_eq!(
            resolve_baseline_label(Path::new("a.png"), &candidates).as_deref(),
            Some("./a.png")
        );
        assert_eq!(
            resolve_baseline_label(Path::new("sub/../a.png"), &candidates).as_deref(),
            Some("./a.png")
        );
        assert_eq!(
            resolve_baseline_label(Path::new("missing.png"), &candidates),
            None
        );
    }

    #[test]
    fn raw_stdin_gray1_lsb_decodes_to_luma8() {
        let frame =
            decode_raw_stdin_frame(vec![0b0000_1011], 4, 1, RawPixelFormatArg::Gray1Lsb, None)
                .unwrap();

        assert_eq!(frame.format().pixel_format, imq::PixelFormat::Luma8);
        assert_eq!(plane_bytes(&frame), [255, 255, 0, 255]);
    }

    #[test]
    fn raw_stdin_gray4_msb_honors_stride() {
        let frame = decode_raw_stdin_frame(
            vec![0x12, 0xff, 0x34, 0xee],
            2,
            2,
            RawPixelFormatArg::Gray4Msb,
            Some(2),
        )
        .unwrap();

        assert_eq!(plane_bytes(&frame), [17, 34, 51, 68]);
    }

    #[test]
    fn bit_packed_raw_roundtrips_through_imqraw_as_luma8() {
        let frame =
            decode_raw_stdin_frame(vec![0b0000_1011], 4, 1, RawPixelFormatArg::Gray1Lsb, None)
                .unwrap();
        let bundle = imq::RawImageBundle::new(vec![imq::RawImageRecord::new(
            Some("mask".to_string()),
            vec!["mask".to_string()],
            frame,
        )]);
        let decoded =
            imq::decode_imqraw_bundle(&imq::encode_imqraw_bundle(&bundle).unwrap()).unwrap();
        let record = decoded.select_tag("mask").unwrap();

        assert_eq!(record.frame.format().pixel_format, imq::PixelFormat::Luma8);
        assert_eq!(plane_bytes(&record.frame), [255, 255, 0, 255]);
    }

    #[test]
    fn raw_stdin_binary1_lsb_stays_native_binary() {
        let frame =
            decode_raw_stdin_frame(vec![0b0000_1011], 4, 1, RawPixelFormatArg::Binary1Lsb, None)
                .unwrap();

        assert_eq!(frame.format().pixel_format, imq::PixelFormat::Binary1Lsb);
        assert_eq!(plane_bytes(&frame), [0b0000_1011]);
    }

    #[test]
    fn selected_metric_accepts_domain_only() {
        let metric = MetricOutput::new(
            "psnr:rgb-visible",
            35.0,
            "dB",
            imq::metrics::Direction::HigherIsBetter,
        )
        .with_detail("max_channel_delta_code", 3.0);
        let report = ComparisonReport::new(
            imq::Dimensions::new(1, 1).unwrap(),
            imq::FormatSpec::new(imq::PixelFormat::Rgba8),
            imq::FormatSpec::new(imq::PixelFormat::Rgba8),
            vec![metric],
        );
        let gate = evaluate_gate(
            &report,
            ComparisonThresholds {
                selected_metric: Some("rgb-visible".to_string()),
                fail_under: Some(34.0),
                max_selected_channel_delta: Some(4.0),
                ..ComparisonThresholds::default()
            },
        )
        .unwrap();
        assert!(gate.passed);
    }

    #[test]
    fn gate_rejects_non_finite_thresholds() {
        let report = ComparisonReport::new(
            imq::Dimensions::new(1, 1).unwrap(),
            imq::FormatSpec::new(imq::PixelFormat::Rgba8),
            imq::FormatSpec::new(imq::PixelFormat::Rgba8),
            vec![MetricOutput::new(
                "psnr",
                40.0,
                "dB",
                imq::metrics::Direction::HigherIsBetter,
            )],
        );

        for thresholds in [
            ComparisonThresholds {
                fail_under: Some(f64::NAN),
                ..ComparisonThresholds::default()
            },
            ComparisonThresholds {
                max_selected_channel_delta: Some(f64::INFINITY),
                ..ComparisonThresholds::default()
            },
        ] {
            assert!(evaluate_gate(&report, thresholds).is_err());
        }
    }

    #[test]
    fn comparison_csv_preserves_gate_decision() {
        let report = ComparisonReport::new(
            imq::Dimensions::new(1, 1).unwrap(),
            imq::FormatSpec::new(imq::PixelFormat::Rgba8),
            imq::FormatSpec::new(imq::PixelFormat::Rgba8),
            vec![MetricOutput::new(
                "psnr",
                39.0,
                "dB",
                imq::metrics::Direction::HigherIsBetter,
            )],
        )
        .with_gate(ComparisonGateReport {
            thresholds: ComparisonThresholds {
                fail_under: Some(40.0),
                ..ComparisonThresholds::default()
            },
            passed: false,
            failures: vec!["psnr below threshold".to_string()],
        });

        let csv = csv_string(comparison_csv_rows(&report).unwrap()).unwrap();
        let mut reader = csv::Reader::from_reader(csv.as_bytes());
        let headers = reader.headers().unwrap().clone();
        let gate_index = headers.iter().position(|name| name == "gate_json").unwrap();
        let record = reader.records().next().unwrap().unwrap();
        let gate: serde_json::Value =
            serde_json::from_str(record.get(gate_index).unwrap()).unwrap();

        assert_eq!(gate["passed"], false);
        assert_eq!(gate["thresholds"]["fail_under"], 40.0);
        assert_eq!(gate["failures"][0], "psnr below threshold");
    }

    #[test]
    fn sqlite_preserves_non_finite_scores_as_canonical_text() {
        let path = sqlite_test_path("nonfinite");
        let report = ComparisonReport::new(
            imq::Dimensions::new(1, 1).unwrap(),
            imq::FormatSpec::new(imq::PixelFormat::Rgba8),
            imq::FormatSpec::new(imq::PixelFormat::Rgba8),
            vec![
                MetricOutput::new("finite", 1.25, "", imq::metrics::Direction::Neutral),
                MetricOutput::new("nan", f64::NAN, "", imq::metrics::Direction::Neutral),
                MetricOutput::new(
                    "pos_inf",
                    f64::INFINITY,
                    "",
                    imq::metrics::Direction::Neutral,
                ),
                MetricOutput::new(
                    "neg_inf",
                    f64::NEG_INFINITY,
                    "",
                    imq::metrics::Direction::Neutral,
                ),
            ],
        );
        write_sqlite_report(Some(&path), SqlReport::Image(&report)).unwrap();

        let conn = open_sqlite(&path).unwrap();
        let mut statement = conn
            .prepare("SELECT name, typeof(score), score FROM imq_metrics ORDER BY id")
            .unwrap();
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, rusqlite::types::Value>(2)?,
                ))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            rows[0],
            (
                "finite".into(),
                "real".into(),
                rusqlite::types::Value::Real(1.25)
            )
        );
        for (index, expected) in [(1, "NaN"), (2, "Infinity"), (3, "-Infinity")] {
            assert_eq!(rows[index].1, "text");
            assert_eq!(rows[index].2, rusqlite::types::Value::Text(expected.into()));
        }
        drop(statement);
        drop(conn);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sqlite_report_and_metrics_roll_back_together() {
        let path = sqlite_test_path("rollback");
        let conn = open_sqlite(&path).unwrap();
        init_sqlite(&conn).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER reject_metric BEFORE INSERT ON imq_metrics BEGIN SELECT RAISE(ABORT, 'forced metric failure'); END;",
        )
        .unwrap();
        drop(conn);
        let report = ComparisonReport::new(
            imq::Dimensions::new(1, 1).unwrap(),
            imq::FormatSpec::new(imq::PixelFormat::Rgba8),
            imq::FormatSpec::new(imq::PixelFormat::Rgba8),
            vec![MetricOutput::new(
                "mse",
                0.0,
                "code^2",
                imq::metrics::Direction::LowerIsBetter,
            )],
        );
        assert!(write_sqlite_report(Some(&path), SqlReport::Image(&report)).is_err());

        let conn = open_sqlite(&path).unwrap();
        let reports: i64 = conn
            .query_row("SELECT COUNT(*) FROM imq_reports", [], |row| row.get(0))
            .unwrap();
        let metrics: i64 = conn
            .query_row("SELECT COUNT(*) FROM imq_metrics", [], |row| row.get(0))
            .unwrap();
        assert_eq!((reports, metrics), (0, 0));
        drop(conn);
        let _ = std::fs::remove_file(path);
    }

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn video_cli_accepts_timestamp_alignment_controls() {
        let cli = Cli::try_parse_from([
            "imq",
            "video",
            "reference.mkv",
            "candidate.mkv",
            "--align",
            "timestamp",
            "--max-timestamp-delta",
            "0.02",
            "--allow-reuse-distorted",
            "--timestamp-scale",
            "1.001",
            "--timestamp-offset",
            "-0.25",
        ])
        .unwrap();
        let Command::Video(cmd) = cli.command else {
            panic!("expected video command")
        };
        assert_eq!(cmd.align, VideoAlignmentArg::Timestamp);
        assert_eq!(cmd.max_timestamp_delta, Some(0.02));
        assert!(cmd.allow_reuse_distorted);
        assert_eq!(cmd.timestamp_scale, Some(1.001));
        assert_eq!(cmd.timestamp_offset, Some(-0.25));
    }

    #[cfg(unix)]
    #[test]
    fn remote_capture_kills_child_when_byte_limit_is_exceeded() {
        use std::os::unix::fs::PermissionsExt;

        let script = std::env::temp_dir().join(format!(
            "imq-fake-ssh-limit-{}-{}.sh",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(
            &script,
            b"#!/bin/sh\nwhile :; do printf '0123456789abcdef'; printf 'diagnostic\\n' >&2; done\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&script, permissions).unwrap();

        let spec = SshInput {
            user: None,
            host: "example.com".to_string(),
            port: None,
            path: "/image.png".to_string(),
        };
        let options = RemoteOptions {
            ssh: script.clone(),
            max_bytes: 8,
            ..RemoteOptions::default()
        };
        let error = ssh_capture_stdout(&spec, "ignored", &options).unwrap_err();
        let _ = std::fs::remove_file(script);
        assert!(error.to_string().contains("remote-max-bytes"));
    }

    #[test]
    fn remote_stderr_capture_retains_only_the_bounded_tail() {
        let mut input = vec![b'a'; REMOTE_STDERR_TAIL_BYTES + 1024];
        input.extend_from_slice(b"tail-marker");

        let captured = read_bounded_stderr_tail(input.as_slice(), REMOTE_STDERR_TAIL_BYTES);

        assert!(captured.starts_with("[stderr truncated; showing tail]\n"));
        assert!(captured.ends_with("tail-marker"));
        assert!(captured.len() <= REMOTE_STDERR_TAIL_BYTES + 64);
    }

    #[cfg(unix)]
    #[test]
    fn scp_copy_uses_private_unique_directory_and_removes_partials() {
        use std::os::unix::fs::PermissionsExt;

        let base = create_private_temp_dir(&std::env::temp_dir(), "scp-test-root").unwrap();
        let script = base.join("fake-scp.sh");
        std::fs::write(
            &script,
            b"#!/bin/sh\ndestination=\nfor argument in \"$@\"; do destination=$argument; done\nprintf payload > \"$destination\"\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&script, permissions).unwrap();

        let spec = SshInput {
            user: None,
            host: "example.com".to_string(),
            port: None,
            path: "/../../escape.png".to_string(),
        };
        let options = RemoteOptions {
            scp: script.clone(),
            copy_dir: Some(base.clone()),
            ..RemoteOptions::default()
        };
        let temp = scp_to_temp(&spec, &options, "input").unwrap();
        let private_dir = temp.path.parent().unwrap().to_path_buf();
        assert_eq!(temp.path.file_name().unwrap(), "payload");
        assert_eq!(std::fs::read(&temp.path).unwrap(), b"payload");
        assert_eq!(
            std::fs::metadata(&private_dir)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert_eq!(private_dir.parent(), Some(base.as_path()));
        drop(temp);
        assert!(!private_dir.exists());

        std::fs::write(
            &script,
            b"#!/bin/sh\ndestination=\nfor argument in \"$@\"; do destination=$argument; done\nprintf partial > \"$destination\"\nprintf 'copy failed marker\\n' >&2\nexit 7\n",
        )
        .unwrap();
        let error = scp_to_temp(&spec, &options, "input").unwrap_err();
        assert!(error.to_string().contains("copy failed marker"));
        assert!(
            std::fs::read_dir(&base)
                .unwrap()
                .filter_map(|entry| entry.ok())
                .all(|entry| !entry.file_type().unwrap().is_dir())
        );

        let _ = std::fs::remove_file(script);
        let _ = std::fs::remove_dir(base);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn parses_gpu_error_and_window_metrics() {
        let (metrics, wssim) = parse_gpu_metrics("psnr,maxae,wssim").unwrap();
        assert_eq!(
            metrics,
            vec![
                imq::gpu::GpuErrorMetric::Psnr,
                imq::gpu::GpuErrorMetric::MaxAbsoluteError,
            ]
        );
        assert!(wssim);

        let (metrics, wssim) = parse_gpu_metrics("wssim").unwrap();
        assert!(metrics.is_empty());
        assert!(wssim);
        assert!(parse_gpu_metrics("ssim").is_err());
    }
}

#[cfg(feature = "ffmpeg")]
fn selected_video_frame_pairs(
    video_frame: Option<u64>,
    video_frames: Option<String>,
) -> Result<Option<Vec<VideoFramePair>>> {
    if let Some(frame) = video_frame {
        return Ok(Some(vec![VideoFramePair::new(frame, frame)]));
    }
    video_frames
        .as_deref()
        .map(imq::parse_video_frame_pairs)
        .transpose()
        .map_err(Into::into)
}

#[cfg(feature = "ffmpeg")]
fn compare_video_frame_pair_inputs(
    reference: &InputSpec,
    distorted: &InputSpec,
    pairs: &[VideoFramePair],
    metrics_csv: &str,
    ffmpeg_cli: FfmpegCliOptions,
    remote: &RemoteOptions,
) -> Result<VideoFramePairComparisonReport> {
    if pairs.is_empty() {
        bail!("--video-frames must contain at least one frame or frame pair");
    }
    let metrics = MetricSet::from_csv(metrics_csv)?;
    let ffmpeg = imq::video::FfmpegOptions {
        ffmpeg: ffmpeg_cli.ffmpeg,
        ffprobe: ffmpeg_cli.ffprobe,
        stream_index: ffmpeg_cli.stream,
        scale: parse_optional_scale(ffmpeg_cli.width, ffmpeg_cli.height)?,
        input_args: Vec::new(),
    };
    let mut reports = Vec::with_capacity(pairs.len());
    let mut dimensions = None;
    for (pair_index, pair) in pairs.iter().enumerate() {
        let reference_frame =
            extract_video_frame_input_spec(reference, pair.reference, &ffmpeg, remote)
                .with_context(|| format!("failed to extract reference frame {}", pair.reference))?;
        let distorted_frame =
            extract_video_frame_input_spec(distorted, pair.distorted, &ffmpeg, remote)
                .with_context(|| format!("failed to extract distorted frame {}", pair.distorted))?;
        if reference_frame.dimensions() != distorted_frame.dimensions() {
            bail!(
                "decoded frame dimensions differ: {:?} vs {:?}",
                reference_frame.dimensions(),
                distorted_frame.dimensions()
            );
        }
        if let Some(expected) = dimensions {
            if expected != reference_frame.dimensions() {
                bail!(
                    "decoded frame dimensions changed across pairs: {:?} vs {:?}",
                    expected,
                    reference_frame.dimensions()
                );
            }
        } else {
            dimensions = Some(reference_frame.dimensions());
        }
        let outputs = metrics.compare(&reference_frame.as_view(), &distorted_frame.as_view())?;
        reports.push(imq::FramePairReport {
            pair_index: u64::try_from(pair_index).unwrap_or(u64::MAX),
            reference_frame_index: pair.reference,
            distorted_frame_index: pair.distorted,
            label_frame_index: pair.label_frame_index,
            reference_pts_seconds: None,
            distorted_pts_seconds: None,
            metrics: outputs,
        });
    }
    let mean_metrics = mean_pair_metric_outputs(&reports);
    Ok(VideoFramePairComparisonReport {
        reference: reference.display_label(),
        distorted: distorted.display_label(),
        dimensions: dimensions.ok_or_else(|| anyhow::anyhow!("no frame pairs were compared"))?,
        pairs: reports,
        mean_metrics,
    })
}

#[cfg(feature = "ffmpeg")]
fn extract_video_frame_input_spec(
    input: &InputSpec,
    frame_index: u64,
    ffmpeg: &imq::video::FfmpegOptions,
    remote: &RemoteOptions,
) -> Result<imq::FrameOwned> {
    match input {
        InputSpec::Local(path) => Ok(imq::video::decode_single_frame(path, frame_index, ffmpeg)?),
        InputSpec::Stdin => bail!("stdin video frame extraction is not supported yet"),
        InputSpec::Ssh(spec) => extract_remote_video_frame(spec, frame_index, ffmpeg, remote),
    }
}

#[cfg(feature = "ffmpeg")]
fn extract_remote_video_frame(
    spec: &SshInput,
    frame_index: u64,
    ffmpeg: &imq::video::FfmpegOptions,
    remote: &RemoteOptions,
) -> Result<imq::FrameOwned> {
    match remote.transfer {
        RemoteTransferMode::Stream => {
            extract_remote_video_frame_png(spec, frame_index, ffmpeg, remote)
        }
        RemoteTransferMode::CopyFrame => {
            let temp = remote_extract_frame_to_temp_then_scp(spec, frame_index, ffmpeg, remote)?;
            Ok(image_crate::load_image_path(&temp.path)?)
        }
        RemoteTransferMode::CopySource => {
            eprintln!(
                "warning: --remote-transfer copy-source copies full remote video files to local temporary storage"
            );
            let temp = scp_to_temp(spec, remote, "source")?;
            Ok(imq::video::decode_single_frame(
                &temp.path,
                frame_index,
                ffmpeg,
            )?)
        }
        RemoteTransferMode::CopyInput => {
            bail!(
                "copy-input is not allowed for video inputs; use copy-frame or copy-source explicitly"
            )
        }
    }
}

#[cfg(feature = "ffmpeg")]
fn extract_remote_video_frame_png(
    spec: &SshInput,
    frame_index: u64,
    ffmpeg: &imq::video::FfmpegOptions,
    remote: &RemoteOptions,
) -> Result<imq::FrameOwned> {
    let bytes = remote_video_frame_png_bytes(spec, frame_index, ffmpeg, remote).map_err(|error| {
        anyhow::anyhow!(
            "remote stream failed for {}.\ncopy fallback is disabled by default.\nUse --remote-transfer copy-frame to copy extracted frames, or --remote-transfer copy-source to copy the source file explicitly.\n{error}",
            spec.uri()
        )
    })?;
    Ok(image_crate::decode_image_bytes(&bytes)?)
}

#[cfg(feature = "ffmpeg")]
fn remote_video_frame_png_bytes(
    spec: &SshInput,
    frame_index: u64,
    ffmpeg: &imq::video::FfmpegOptions,
    remote: &RemoteOptions,
) -> Result<Vec<u8>> {
    if remote.frame_format != RemoteFrameFormat::Png {
        bail!("--remote-frame-format rgba is reserved for a future raw RGBA remote path")
    }
    let mut filters = format!("select=eq(n\\,{frame_index})");
    if let Some(scale) = ffmpeg.scale {
        filters.push_str(&format!(",scale={}:{}", scale.width, scale.height));
    }
    let command = format!(
        "{} -hide_banner -loglevel error -i {} -vf {} -vsync 0 -frames:v 1 -f image2pipe -vcodec png pipe:1",
        imq::shell_quote_posix(&ffmpeg.ffmpeg.to_string_lossy()),
        imq::shell_quote_posix(&spec.path),
        imq::shell_quote_posix(&filters),
    );
    ssh_capture_stdout(spec, &command, remote)
}

#[cfg(feature = "ffmpeg")]
fn remote_extract_frame_to_temp_then_scp(
    spec: &SshInput,
    frame_index: u64,
    ffmpeg: &imq::video::FfmpegOptions,
    remote: &RemoteOptions,
) -> Result<ManagedTempPath> {
    if remote.frame_format != RemoteFrameFormat::Png {
        bail!("--remote-frame-format rgba is reserved for a future raw RGBA remote path")
    }
    let mut filters = format!("select=eq(n\\,{frame_index})");
    if let Some(scale) = ffmpeg.scale {
        filters.push_str(&format!(",scale={}:{}", scale.width, scale.height));
    }
    let remote_path = format!("/tmp/imq-frame-{}-{}.png", std::process::id(), frame_index);
    let command = format!(
        "{} -hide_banner -loglevel error -i {} -vf {} -vsync 0 -frames:v 1 -f image2 -vcodec png {} && printf '%s\\n' {}",
        imq::shell_quote_posix(&ffmpeg.ffmpeg.to_string_lossy()),
        imq::shell_quote_posix(&spec.path),
        imq::shell_quote_posix(&filters),
        imq::shell_quote_posix(&remote_path),
        imq::shell_quote_posix(&remote_path),
    );
    let output = ssh_capture_stdout(spec, &command, remote)?;
    let remote_temp = String::from_utf8_lossy(&output).trim().to_string();
    if remote_temp.is_empty() {
        bail!("remote frame extraction did not report a temporary frame path")
    }
    let local = scp_path_to_temp(spec, &remote_temp, remote, "frame")?;
    if !remote.keep_temp {
        let _ = ssh_capture_stdout(
            spec,
            &format!("rm -f -- {}", imq::shell_quote_posix(&remote_temp)),
            remote,
        );
    } else {
        eprintln!("kept remote frame temp: {remote_temp}");
    }
    Ok(local)
}

#[cfg(feature = "ffmpeg")]
fn mean_pair_metric_outputs(frames: &[imq::FramePairReport]) -> Vec<MetricOutput> {
    use std::collections::BTreeMap;
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
                .insert("pairs".to_string(), frames.len() as f64);
            metric.name = format!("mean_{}", metric.name);
            metric
        })
        .collect()
}

#[cfg(feature = "ffmpeg")]
fn run_video(cmd: VideoCmd) -> Result<()> {
    let metrics = MetricSet::from_csv(&cmd.metrics)?;
    let compare = imq::video::VideoCompareOptions {
        every: cmd.every.max(1),
        max_frames: cmd.max_frames,
    };
    let timestamp_options = match cmd.align {
        VideoAlignmentArg::Decode => {
            if cmd.max_timestamp_delta.is_some()
                || cmd.allow_reuse_distorted
                || cmd.no_estimate_timestamp_transform
                || cmd.timestamp_scale.is_some()
                || cmd.timestamp_offset.is_some()
            {
                bail!("timestamp alignment options require `--align timestamp`")
            }
            None
        }
        VideoAlignmentArg::Timestamp => {
            let mut pairing = imq::TimestampPairingOptions::default();
            if let Some(max_delta_seconds) = cmd.max_timestamp_delta {
                if !max_delta_seconds.is_finite() || max_delta_seconds < 0.0 {
                    bail!("--max-timestamp-delta must be finite and non-negative")
                }
                pairing.max_delta_seconds = max_delta_seconds;
            }
            pairing.allow_reuse_distorted = cmd.allow_reuse_distorted;

            let transform = if cmd.timestamp_scale.is_some() || cmd.timestamp_offset.is_some() {
                let scale = cmd.timestamp_scale.unwrap_or(1.0);
                let offset_seconds = cmd.timestamp_offset.unwrap_or(0.0);
                if !scale.is_finite() || scale <= 0.0 {
                    bail!("--timestamp-scale must be finite and greater than zero")
                }
                if !offset_seconds.is_finite() {
                    bail!("--timestamp-offset must be finite")
                }
                Some(imq::TimestampTransform {
                    scale,
                    offset_seconds,
                })
            } else {
                None
            };
            Some(imq::video::TimestampVideoCompareOptions {
                compare: compare.clone(),
                pairing,
                transform,
                estimate_transform: !cmd.no_estimate_timestamp_transform,
            })
        }
    };
    let ffmpeg = imq::video::FfmpegOptions {
        ffmpeg: cmd.ffmpeg,
        ffprobe: cmd.ffprobe,
        stream_index: cmd.stream,
        scale: parse_optional_scale(cmd.width, cmd.height)?,
        input_args: Vec::new(),
    };
    let format = output_format(cmd.json, cmd.output.format);
    if let Some(options) = timestamp_options {
        let report = imq::video::compare_videos_by_timestamp(
            &cmd.reference,
            &cmd.distorted,
            &ffmpeg,
            &options,
            &metrics,
        )
        .with_context(|| "timestamp-aligned video comparison failed")?;
        if report.alignment.pairs.is_empty() && options.compare.max_frames != Some(0) {
            bail!(
                "timestamp alignment accepted no frame pairs; increase --max-timestamp-delta or verify the timeline transform"
            )
        }
        write_sqlite_report(
            cmd.output.sqlite.as_deref(),
            SqlReport::TimestampVideo(&report),
        )?;
        emit_timestamp_video_report(&report, format, &cmd.output)?;
    } else {
        let report =
            imq::video::compare_videos(&cmd.reference, &cmd.distorted, &ffmpeg, &compare, &metrics)
                .with_context(|| "video comparison failed")?;
        write_sqlite_report(cmd.output.sqlite.as_deref(), SqlReport::Video(&report))?;
        emit_video_report(&report, format, &cmd.output)?;
    }
    Ok(())
}

#[cfg(feature = "ffmpeg")]
fn run_extract_frame(cmd: ExtractFrameCmd) -> Result<()> {
    let options = imq::video::FfmpegOptions {
        ffmpeg: cmd.ffmpeg,
        ffprobe: cmd.ffprobe,
        ..Default::default()
    };
    let frame =
        imq::video::decode_single_frame(&cmd.input, cmd.index, &options).with_context(|| {
            format!(
                "failed to decode frame {} from `{}`",
                cmd.index,
                cmd.input.display()
            )
        })?;
    let view = frame.as_view();
    let plane = view.plane(0)?;
    let img = image::RgbaImage::from_raw(
        frame.dimensions().width,
        frame.dimensions().height,
        plane.data.to_vec(),
    )
    .ok_or_else(|| anyhow::anyhow!("decoded RGBA frame could not be represented as an image"))?;
    img.save(&cmd.output)
        .with_context(|| format!("failed to save `{}`", cmd.output.display()))?;
    println!("saved frame {} -> {}", cmd.index, cmd.output.display());
    Ok(())
}

#[cfg(feature = "ffmpeg")]
fn run_probe(cmd: ProbeCmd) -> Result<()> {
    let options = imq::video::FfmpegOptions {
        ffmpeg: cmd.ffmpeg,
        ffprobe: cmd.ffprobe,
        stream_index: cmd.stream,
        ..Default::default()
    };
    let info = imq::video::probe_video(&cmd.input, &options)
        .with_context(|| format!("failed to probe `{}`", cmd.input.display()))?;
    let report = ProbeReport {
        input: cmd.input.display().to_string(),
        info,
    };
    write_sqlite_probe(cmd.output.sqlite.as_deref(), &report)?;
    emit_probe_report(
        &report,
        output_format(cmd.json, cmd.output.format),
        &cmd.output,
    )?;
    Ok(())
}

fn run_formats(cmd: FormatsCmd) -> Result<()> {
    let report = FormatsReport {
        formats: image_crate::enabled_format_hint()
            .iter()
            .map(|format| (*format).to_string())
            .collect(),
    };
    write_sqlite_formats(cmd.output.sqlite.as_deref(), &report)?;
    emit_formats_report(&report, cmd.output.format, &cmd.output)?;
    Ok(())
}

fn run_pack(cmd: PackCmd) -> Result<()> {
    let labels = pack_labels(&cmd.inputs, &cmd.labels)?;
    let tags = parse_pack_tags(cmd.inputs.len(), &cmd.tags)?;
    let records = cmd
        .inputs
        .iter()
        .zip(labels)
        .zip(tags)
        .map(|((input, label), tags)| {
            let frame = image_crate::load_image_path(input)
                .with_context(|| format!("failed to decode `{}`", input.display()))?;
            Ok(imq::RawImageRecord::new(Some(label), tags, frame))
        })
        .collect::<Result<Vec<_>>>()?;
    let bundle = imq::RawImageBundle::new(records);
    let bytes = imq::encode_imqraw_bundle(&bundle)?;
    write_binary_output(cmd.output.as_deref(), &bytes)
}

fn run_bundle_info(cmd: BundleInfoCmd) -> Result<()> {
    let bytes = read_binary_input(&cmd.input)?;
    let bundle = imq::decode_imqraw_bundle(&bytes)
        .with_context(|| format!("failed to decode `{}`", image_input_label(&cmd.input)))?;
    let report = BundleInfoReport::from_bundle(image_input_label(&cmd.input), &bundle);
    emit_bundle_info_report(&report, cmd.format, cmd.output.as_deref())
}

fn pack_labels(inputs: &[PathBuf], labels: &[String]) -> Result<Vec<String>> {
    if !labels.is_empty() && labels.len() != inputs.len() {
        bail!(
            "--label must be passed once per input when used; got {} label(s) for {} input(s)",
            labels.len(),
            inputs.len()
        );
    }
    Ok(if labels.is_empty() {
        inputs
            .iter()
            .map(|input| input.display().to_string())
            .collect()
    } else {
        labels.to_vec()
    })
}

fn parse_pack_tags(input_count: usize, specs: &[String]) -> Result<Vec<Vec<String>>> {
    let mut tags = vec![Vec::new(); input_count];
    for spec in specs {
        match spec.split_once(':') {
            Some((target, tag)) if target.eq_ignore_ascii_case("all") => {
                tags.iter_mut()
                    .for_each(|entry| entry.push(tag.to_string()));
            }
            Some((target, tag)) => {
                let index = target
                    .parse::<usize>()
                    .with_context(|| format!("invalid tag target `{target}` in `{spec}`"))?;
                if !(1..=input_count).contains(&index) {
                    bail!(
                        "tag target `{target}` is out of range; use 1..={}",
                        input_count
                    );
                }
                tags[index - 1].push(tag.to_string());
            }
            None => {
                tags.iter_mut()
                    .for_each(|entry| entry.push(spec.to_string()));
            }
        }
    }
    Ok(tags)
}

#[derive(Debug, Serialize)]
struct ProbeReport {
    input: String,
    info: imq::video::VideoInfo,
}

#[derive(Debug, Serialize)]
struct ImageStatsReport {
    input: String,
    stats: imq::ImageStatistics,
}

#[derive(Debug, Serialize)]
struct ImageComparisonStatsReport {
    comparison: ComparisonReport,
    stats: ImageComparisonStats,
}

#[derive(Debug, Serialize)]
struct ImageComparisonStats {
    reference: ImageStatsReport,
    distorted: ImageStatsReport,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Serialize)]
struct GpuCliReport {
    capabilities: Option<imq::gpu::GpuCapabilities>,
    initialization_error: Option<String>,
    reference: String,
    comparisons: Vec<GpuCliComparison>,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Serialize)]
struct GpuCliComparison {
    candidate: String,
    error_result: Option<imq::gpu::GpuRgba8Comparison>,
    wssim_result: Option<imq::gpu::GpuWindowedSsimComparison>,
}

#[derive(Debug, Serialize)]
struct FormatsReport {
    formats: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BundleInfoReport {
    input: String,
    images: Vec<BundleImageInfo>,
}

impl BundleInfoReport {
    fn from_bundle(input: String, bundle: &imq::RawImageBundle) -> Self {
        let images = bundle
            .records
            .iter()
            .enumerate()
            .map(|(index, record)| {
                let dimensions = record.frame.dimensions();
                let format = record.frame.format();
                BundleImageInfo {
                    index,
                    label: record.label.clone(),
                    tags: record.tags.clone(),
                    width: dimensions.width,
                    height: dimensions.height,
                    pixel_format: format!("{:?}", format.pixel_format),
                    color_space: format!("{:?}", format.color_space),
                    transfer: format!("{:?}", format.transfer),
                    range: format!("{:?}", format.range),
                    planes: record.frame.owned_planes().len(),
                    bytes: record
                        .frame
                        .owned_planes()
                        .iter()
                        .map(|plane| u64::try_from(plane.data.len()).unwrap_or(u64::MAX))
                        .sum(),
                }
            })
            .collect();
        Self { input, images }
    }
}

#[derive(Debug, Serialize)]
struct BundleImageInfo {
    index: usize,
    label: Option<String>,
    tags: Vec<String>,
    width: u32,
    height: u32,
    pixel_format: String,
    color_space: String,
    transfer: String,
    range: String,
    planes: usize,
    bytes: u64,
}

#[derive(Debug, Serialize)]
struct MetricCsvRow {
    report_kind: &'static str,
    reference: String,
    distorted: String,
    width: u32,
    height: u32,
    scope: &'static str,
    frame_index: Option<u64>,
    pts_seconds: Option<f64>,
    pair_index: Option<u64>,
    reference_frame_index: Option<u64>,
    distorted_frame_index: Option<u64>,
    metric: String,
    score: f64,
    unit: String,
    direction: String,
    details_json: String,
}

#[derive(Debug, Serialize)]
struct ComparisonCsvRow {
    report_kind: &'static str,
    reference: String,
    distorted: String,
    width: u32,
    height: u32,
    scope: &'static str,
    frame_index: Option<u64>,
    pts_seconds: Option<f64>,
    pair_index: Option<u64>,
    reference_frame_index: Option<u64>,
    distorted_frame_index: Option<u64>,
    metric: String,
    score: f64,
    unit: String,
    direction: String,
    details_json: String,
    gate_json: Option<String>,
}

#[derive(Debug, Serialize)]
struct SuiteCsvRow {
    reference: String,
    candidate: String,
    candidate_passed: bool,
    overall_rank: Option<usize>,
    mean_rank: Option<f64>,
    pareto_optimal: bool,
    error: Option<String>,
    gate_json: Option<String>,
    baseline_gate_json: Option<String>,
    metric: Option<String>,
    score: Option<f64>,
    unit: Option<String>,
    direction: Option<String>,
    metric_rank: Option<usize>,
    baseline_delta: Option<f64>,
    details_json: Option<String>,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Serialize)]
struct GpuCsvRow {
    reference: String,
    candidate: String,
    execution: String,
    fallback_reason: Option<String>,
    metric: String,
    score: f64,
    unit: String,
    direction: String,
    details_json: String,
}

#[derive(Debug, Serialize)]
struct ProbeCsvRow {
    input: String,
    width: u32,
    height: u32,
    avg_frame_rate: Option<f64>,
    nb_frames: Option<u64>,
    codec_name: Option<String>,
    duration_seconds: Option<f64>,
}

#[derive(Debug, Serialize)]
struct FormatCsvRow {
    format: String,
}

#[derive(Debug, Serialize)]
struct StatsCsvRow {
    input: String,
    width: u32,
    height: u32,
    pixels: u64,
    metric: String,
    value: String,
}

#[derive(Debug, Serialize)]
struct BundleInfoCsvRow {
    input: String,
    index: usize,
    label: String,
    tags: String,
    width: u32,
    height: u32,
    pixel_format: String,
    color_space: String,
    transfer: String,
    range: String,
    planes: usize,
    bytes: u64,
}

struct MetricCsvContext {
    report_kind: &'static str,
    reference: String,
    distorted: String,
    width: u32,
    height: u32,
    scope: &'static str,
    frame_index: Option<u64>,
    pts_seconds: Option<f64>,
    pair_index: Option<u64>,
    reference_frame_index: Option<u64>,
    distorted_frame_index: Option<u64>,
}

enum SqlReport<'a> {
    Image(&'a ComparisonReport),
    Video(&'a VideoReport),
    TimestampVideo(&'a imq::video::TimestampVideoComparison),
    VideoPairs(&'a VideoFramePairComparisonReport),
}

struct FfmpegCliOptions {
    ffmpeg: PathBuf,
    ffprobe: PathBuf,
    stream: usize,
    width: Option<u32>,
    height: Option<u32>,
}

struct SqlReportInsert<'a> {
    kind: &'a str,
    reference: Option<&'a str>,
    distorted: Option<&'a str>,
    width: u32,
    height: u32,
    compared_frames: Option<i64>,
    payload_json: &'a str,
}

fn output_format(json: bool, format: OutputFormatArg) -> OutputFormatArg {
    if json { OutputFormatArg::Json } else { format }
}

fn emit_comparison_report(
    report: &ComparisonReport,
    format: OutputFormatArg,
    output: &OutputArgs,
) -> Result<()> {
    let content = match format {
        OutputFormatArg::Text => render_comparison_text(report),
        OutputFormatArg::Json => serde_json::to_string_pretty(report)?,
        OutputFormatArg::Yaml => serde_yaml_ng::to_string(report)?,
        OutputFormatArg::Toml => toml::to_string_pretty(report)?,
        OutputFormatArg::Csv => csv_string(comparison_csv_rows(report)?)?,
    };
    write_output(output.output.as_deref(), &content)
}

fn emit_suite_report(
    report: &imq::ComparisonSuiteReport,
    format: OutputFormatArg,
    output: Option<&Path>,
) -> Result<()> {
    let content = match format {
        OutputFormatArg::Text => render_suite_text(report),
        OutputFormatArg::Json => serde_json::to_string_pretty(report)?,
        OutputFormatArg::Yaml => serde_yaml_ng::to_string(report)?,
        OutputFormatArg::Toml => toml::to_string_pretty(report)?,
        OutputFormatArg::Csv => csv_string(suite_csv_rows(report))?,
    };
    write_output(output, &content)
}

#[cfg(feature = "gpu")]
fn emit_gpu_report(
    report: &GpuCliReport,
    format: OutputFormatArg,
    output: Option<&Path>,
) -> Result<()> {
    let content = match format {
        OutputFormatArg::Text => render_gpu_report(report),
        OutputFormatArg::Json => serde_json::to_string_pretty(report)?,
        OutputFormatArg::Yaml => serde_yaml_ng::to_string(report)?,
        OutputFormatArg::Toml => toml::to_string_pretty(report)?,
        OutputFormatArg::Csv => csv_string(gpu_csv_rows(report))?,
    };
    write_output(output, &content)
}

fn emit_image_comparison_stats_report(
    report: &ImageComparisonStatsReport,
    format: OutputFormatArg,
    output: &OutputArgs,
) -> Result<()> {
    let content = match format {
        OutputFormatArg::Text => {
            let mut text = render_comparison_text(&report.comparison);
            text.push('\n');
            text.push_str(&render_stats_text(&report.stats.reference));
            text.push('\n');
            text.push_str(&render_stats_text(&report.stats.distorted));
            text
        }
        OutputFormatArg::Json => serde_json::to_string_pretty(report)?,
        OutputFormatArg::Yaml => serde_yaml_ng::to_string(report)?,
        OutputFormatArg::Toml => toml::to_string_pretty(report)?,
        OutputFormatArg::Csv => {
            let mut csv = csv_string(comparison_csv_rows(&report.comparison)?)?;
            csv.push('\n');
            csv.push_str(&csv_string(stats_csv_rows(&report.stats.reference))?);
            csv.push('\n');
            csv.push_str(&csv_string(stats_csv_rows(&report.stats.distorted))?);
            csv
        }
    };
    write_output(output.output.as_deref(), &content)
}

fn emit_stats_report(
    report: &ImageStatsReport,
    format: OutputFormatArg,
    output: &OutputArgs,
) -> Result<()> {
    let content = match format {
        OutputFormatArg::Text => render_stats_text(report),
        OutputFormatArg::Json => serde_json::to_string_pretty(report)?,
        OutputFormatArg::Yaml => serde_yaml_ng::to_string(report)?,
        OutputFormatArg::Toml => toml::to_string_pretty(report)?,
        OutputFormatArg::Csv => csv_string(stats_csv_rows(report))?,
    };
    write_output(output.output.as_deref(), &content)
}

fn emit_video_report(
    report: &VideoReport,
    format: OutputFormatArg,
    output: &OutputArgs,
) -> Result<()> {
    let content = match format {
        OutputFormatArg::Text => render_video_text(report),
        OutputFormatArg::Json => serde_json::to_string_pretty(report)?,
        OutputFormatArg::Yaml => serde_yaml_ng::to_string(report)?,
        OutputFormatArg::Toml => toml::to_string_pretty(report)?,
        OutputFormatArg::Csv => csv_string(video_csv_rows(report))?,
    };
    write_output(output.output.as_deref(), &content)
}

#[cfg(feature = "ffmpeg")]
fn emit_timestamp_video_report(
    report: &imq::video::TimestampVideoComparison,
    format: OutputFormatArg,
    output: &OutputArgs,
) -> Result<()> {
    let content = match format {
        OutputFormatArg::Text => render_timestamp_video_text(report),
        OutputFormatArg::Json => serde_json::to_string_pretty(report)?,
        OutputFormatArg::Yaml => serde_yaml_ng::to_string(report)?,
        OutputFormatArg::Toml => toml::to_string_pretty(report)?,
        OutputFormatArg::Csv => csv_string(timestamp_video_csv_rows(report))?,
    };
    write_output(output.output.as_deref(), &content)
}

fn emit_video_pair_report(
    report: &VideoFramePairComparisonReport,
    format: OutputFormatArg,
    output: &OutputArgs,
) -> Result<()> {
    let content = match format {
        OutputFormatArg::Text => render_video_pair_text(report),
        OutputFormatArg::Json => serde_json::to_string_pretty(report)?,
        OutputFormatArg::Yaml => serde_yaml_ng::to_string(report)?,
        OutputFormatArg::Toml => toml::to_string_pretty(report)?,
        OutputFormatArg::Csv => csv_string(video_pair_csv_rows(report))?,
    };
    write_output(output.output.as_deref(), &content)
}

fn emit_probe_report(
    report: &ProbeReport,
    format: OutputFormatArg,
    output: &OutputArgs,
) -> Result<()> {
    let content = match format {
        OutputFormatArg::Text => render_probe_text(report),
        OutputFormatArg::Json => serde_json::to_string_pretty(report)?,
        OutputFormatArg::Yaml => serde_yaml_ng::to_string(report)?,
        OutputFormatArg::Toml => toml::to_string_pretty(report)?,
        OutputFormatArg::Csv => csv_string([ProbeCsvRow {
            input: report.input.clone(),
            width: report.info.width,
            height: report.info.height,
            avg_frame_rate: report.info.avg_frame_rate,
            nb_frames: report.info.nb_frames,
            codec_name: report.info.codec_name.clone(),
            duration_seconds: report.info.duration_seconds,
        }])?,
    };
    write_output(output.output.as_deref(), &content)
}

fn emit_formats_report(
    report: &FormatsReport,
    format: OutputFormatArg,
    output: &OutputArgs,
) -> Result<()> {
    let content = match format {
        OutputFormatArg::Text => render_formats_text(report),
        OutputFormatArg::Json => serde_json::to_string_pretty(report)?,
        OutputFormatArg::Yaml => serde_yaml_ng::to_string(report)?,
        OutputFormatArg::Toml => toml::to_string_pretty(report)?,
        OutputFormatArg::Csv => csv_string(report.formats.iter().map(|format| FormatCsvRow {
            format: format.clone(),
        }))?,
    };
    write_output(output.output.as_deref(), &content)
}

fn emit_bundle_info_report(
    report: &BundleInfoReport,
    format: OutputFormatArg,
    output: Option<&Path>,
) -> Result<()> {
    let content = match format {
        OutputFormatArg::Text => render_bundle_info_text(report),
        OutputFormatArg::Json => serde_json::to_string_pretty(report)?,
        OutputFormatArg::Yaml => serde_yaml_ng::to_string(report)?,
        OutputFormatArg::Toml => toml::to_string_pretty(report)?,
        OutputFormatArg::Csv => csv_string(bundle_info_csv_rows(report))?,
    };
    write_output(output, &content)
}

fn write_output(path: Option<&Path>, content: &str) -> Result<()> {
    if let Some(path) = path {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create `{}`", parent.display()))?;
        }
        std::fs::write(path, content)
            .with_context(|| format!("failed to write `{}`", path.display()))?;
    } else {
        print!("{content}");
    }
    Ok(())
}

fn write_binary_output(path: Option<&Path>, content: &[u8]) -> Result<()> {
    if let Some(path) = path {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create `{}`", parent.display()))?;
        }
        std::fs::write(path, content)
            .with_context(|| format!("failed to write `{}`", path.display()))?;
    } else {
        std::io::stdout()
            .write_all(content)
            .with_context(|| "failed to write imqraw bundle to stdout")?;
    }
    Ok(())
}

fn read_binary_input(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    if is_stdin_path(path) {
        std::io::stdin()
            .read_to_end(&mut bytes)
            .with_context(|| "failed to read bytes from stdin")?;
    } else {
        bytes =
            std::fs::read(path).with_context(|| format!("failed to read `{}`", path.display()))?;
    }
    Ok(bytes)
}

fn csv_string<T, I>(rows: I) -> Result<String>
where
    T: Serialize,
    I: IntoIterator<Item = T>,
{
    let mut writer = csv::Writer::from_writer(Vec::new());
    for row in rows {
        writer.serialize(row)?;
    }
    Ok(String::from_utf8(writer.into_inner()?)?)
}

fn render_comparison_text(report: &ComparisonReport) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "reference : {}\n",
        report.reference.as_deref().unwrap_or("<none>")
    ));
    output.push_str(&format!(
        "distorted : {}\n",
        report.distorted.as_deref().unwrap_or("<none>")
    ));
    output.push_str(&format!(
        "size      : {}x{}\n",
        report.dimensions.width, report.dimensions.height
    ));
    output.push_str(&render_metrics(&report.metrics));
    if let Some(alpha) = report.alpha {
        output.push_str(&format!(
            "alpha     : ref t/o/p={}/{}/{}  dist t/o/p={}/{}/{}  mismatches={}  max_delta={}  beyond_1_lsb={}\n",
            alpha.reference.transparent,
            alpha.reference.opaque,
            alpha.reference.partial,
            alpha.distorted.transparent,
            alpha.distorted.opaque,
            alpha.distorted.partial,
            alpha.mismatch_count,
            alpha.max_delta,
            alpha.mismatches_beyond_one_lsb
        ));
    }
    if let Some(gate) = &report.gate {
        output.push_str(&format!(
            "gate      : {}\n",
            if gate.passed { "passed" } else { "failed" }
        ));
        for failure in &gate.failures {
            output.push_str(&format!("  - {failure}\n"));
        }
    }
    output
}

fn render_suite_text(report: &imq::ComparisonSuiteReport) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "reference : {}\n",
        report.reference.as_deref().unwrap_or("<none>")
    ));
    output.push_str(&format!(
        "size      : {}x{}\n",
        report.dimensions.width, report.dimensions.height
    ));
    output.push_str(&format!(
        "suite     : {} ({} candidates)\n",
        if report.passed { "passed" } else { "failed" },
        report.candidates.len()
    ));
    if let Some(primary) = &report.primary_metric {
        output.push_str(&format!("primary   : {primary}\n"));
    } else {
        output.push_str("primary   : consensus mean rank\n");
    }
    if let Some(baseline) = &report.baseline_candidate {
        output.push_str(&format!("baseline  : {baseline}\n"));
    }

    for candidate in &report.candidates {
        output.push_str(&format!(
            "\n[rank {}] {}  status={}  pareto={}  mean_rank={}\n",
            candidate
                .overall_rank
                .map(|rank| rank.to_string())
                .unwrap_or_else(|| "-".to_string()),
            candidate.label,
            if candidate.passed() {
                "passed"
            } else {
                "failed"
            },
            candidate.pareto_optimal,
            candidate
                .mean_rank
                .map(|rank| format!("{rank:.4}"))
                .unwrap_or_else(|| "-".to_string())
        ));
        if let Some(error) = &candidate.error {
            output.push_str(&format!("error     : {error}\n"));
            continue;
        }
        for metric in &candidate.metrics {
            let rank = candidate
                .metric_ranks
                .get(&metric.name)
                .map(|rank| rank.to_string())
                .unwrap_or_else(|| "-".to_string());
            let delta = candidate
                .baseline_deltas
                .get(&metric.name)
                .map(|delta| format!(" delta={delta:+.8}"))
                .unwrap_or_default();
            output.push_str(&format!(
                "  {:<24} {:>14.8} {:<18} rank={}{}\n",
                metric.name, metric.score, metric.unit, rank, delta
            ));
        }
        if let Some(gate) = &candidate.gate {
            for failure in gate.failures() {
                output.push_str(&format!("  gate failure: {}\n", failure.message));
            }
        }
        if let Some(gate) = &candidate.baseline_gate {
            for failure in gate.failures() {
                output.push_str(&format!("  baseline gate failure: {}\n", failure.message));
            }
        }
    }
    output
}

#[cfg(feature = "gpu")]
fn render_gpu_capabilities(capabilities: &imq::gpu::GpuCapabilities) -> String {
    format!(
        "adapter                  : {}\n\
         backend                  : {}\n\
         device type              : {}\n\
         driver                   : {}\n\
         driver info              : {}\n\
         vendor/device            : {:#06x}/{:#06x}\n\
         fallback adapter         : {}\n\
         max buffer bytes         : {}\n\
         max storage binding      : {}\n\
         max workgroups/dimension : {}\n\
         max rgba8 input pixels   : {}\n\
         max error pixels         : {}\n\
         max wssim windows        : {}\n",
        capabilities.adapter_name,
        capabilities.backend,
        capabilities.device_type,
        capabilities.driver,
        capabilities.driver_info,
        capabilities.vendor_id,
        capabilities.device_id,
        capabilities.fallback_adapter,
        capabilities.max_buffer_size,
        capabilities.max_storage_buffer_binding_size,
        capabilities.max_compute_workgroups_per_dimension,
        capabilities.max_rgba8_input_pixels,
        capabilities.max_error_stats_pixels,
        capabilities.max_windowed_ssim_windows,
    )
}

#[cfg(feature = "gpu")]
fn render_gpu_report(report: &GpuCliReport) -> String {
    let mut output = String::new();
    if let Some(capabilities) = &report.capabilities {
        output.push_str(&render_gpu_capabilities(capabilities));
    } else if let Some(error) = &report.initialization_error {
        output.push_str(&format!("GPU unavailable          : {error}\n"));
    }
    output.push_str(&format!(
        "reference                : {}\n",
        report.reference
    ));
    for comparison in &report.comparisons {
        output.push_str(&format!(
            "\ncandidate                : {}\n",
            comparison.candidate
        ));
        if let Some(result) = &comparison.error_result {
            output.push_str(&format!(
                "error execution          : {:?}\n",
                result.execution
            ));
            if let Some(reason) = &result.fallback_reason {
                output.push_str(&format!("error fallback reason    : {reason}\n"));
            }
            output.push_str(&render_metrics(&result.metrics));
        }
        if let Some(result) = &comparison.wssim_result {
            output.push_str(&format!(
                "wssim execution          : {:?}\n",
                result.execution
            ));
            if let Some(reason) = &result.fallback_reason {
                output.push_str(&format!("wssim fallback reason    : {reason}\n"));
            }
            output.push_str(&render_metrics(std::slice::from_ref(&result.metric)));
        }
    }
    output
}

fn render_stats_text(report: &ImageStatsReport) -> String {
    let stats = &report.stats;
    let mut output = String::new();
    output.push_str(&format!("image     : {}\n", report.input));
    output.push_str(&format!("size      : {}x{}\n", stats.width, stats.height));
    output.push_str(&format!("pixels    : {}\n", stats.pixels));
    output.push_str(&format!(
        "exposure  : {}  contrast: {}  saturation: {}\n",
        stats.tendencies.exposure, stats.tendencies.contrast, stats.tendencies.saturation
    ));
    output.push_str(&format!(
        "temp/tint : {} / {}\n",
        stats.tendencies.temperature, stats.tendencies.tint
    ));
    output.push_str(&format!(
        "rgb mean  : r={:.4} g={:.4} b={:.4} dominant={}\n",
        stats.color_balance.red_mean,
        stats.color_balance.green_mean,
        stats.color_balance.blue_mean,
        stats.color_balance.dominant_channel
    ));
    output.push_str(&format!(
        "luma      : mean={:.4} std={:.4} p05={:.4} p95={:.4} entropy={:.4}\n",
        stats.luma.mean,
        stats.luma.std_dev,
        stats.luma.p05,
        stats.luma.p95,
        stats.luma.entropy_bits
    ));
    output.push_str(&format!(
        "hsv       : hue={:.2} sat={:.4} value={:.4}\n",
        stats.hsv.mean_hue_degrees, stats.hsv.mean_saturation, stats.hsv.mean_value
    ));
    output.push_str(&format!(
        "flags     : grayscale={} cast={} shadow_clip={} highlight_clip={} transparency={}\n",
        stats.tendencies.likely_grayscale,
        stats.tendencies.color_cast,
        stats.tendencies.shadow_clipping,
        stats.tendencies.highlight_clipping,
        stats.tendencies.has_transparency
    ));
    output
}

fn render_video_text(report: &VideoReport) -> String {
    let mut output = String::new();
    output.push_str(&format!("reference       : {}\n", report.reference));
    output.push_str(&format!("distorted       : {}\n", report.distorted));
    output.push_str(&format!(
        "size            : {}x{}\n",
        report.dimensions.width, report.dimensions.height
    ));
    output.push_str(&format!("compared frames : {}\n", report.compared_frames));
    output.push_str("\nmean metrics\n");
    output.push_str(&render_metrics(&report.mean_metrics));
    output
}

#[cfg(feature = "ffmpeg")]
fn render_timestamp_video_text(report: &imq::video::TimestampVideoComparison) -> String {
    let alignment = &report.alignment;
    let mut output = render_video_text(&report.video);
    output.push_str("\ntimestamp alignment\n");
    output.push_str(&format!(
        "planned pairs         : {}\n",
        alignment.pairs.len()
    ));
    output.push_str(&format!(
        "unmatched reference   : {}\n",
        alignment.unmatched_reference_indices.len()
    ));
    output.push_str(&format!(
        "unmatched distorted   : {}\n",
        alignment.unmatched_distorted_indices.len()
    ));
    output.push_str(&format!(
        "timeline transform    : distorted * {:.12} {:+.12}s\n",
        alignment.transform.scale, alignment.transform.offset_seconds
    ));
    output.push_str(&format!(
        "mean signed residual  : {}\n",
        format_optional_seconds(alignment.mean_signed_delta_seconds)
    ));
    output.push_str(&format!(
        "mean absolute residual: {}\n",
        format_optional_seconds(alignment.mean_absolute_delta_seconds)
    ));
    output.push_str(&format!(
        "RMS residual          : {}\n",
        format_optional_seconds(alignment.rms_delta_seconds)
    ));
    output.push_str(&format!(
        "maximum residual      : {}\n",
        format_optional_seconds(alignment.max_delta_seconds)
    ));
    output
}

#[cfg(feature = "ffmpeg")]
fn format_optional_seconds(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_string(), |value| format!("{value:.9}s"))
}

fn render_video_pair_text(report: &VideoFramePairComparisonReport) -> String {
    let mut output = String::new();
    output.push_str(&format!("reference       : {}\n", report.reference));
    output.push_str(&format!("distorted       : {}\n", report.distorted));
    output.push_str(&format!(
        "size            : {}x{}\n",
        report.dimensions.width, report.dimensions.height
    ));
    output.push_str(&format!("compared pairs  : {}\n", report.pairs.len()));
    for pair in &report.pairs {
        output.push_str(&format!(
            "\npair {}: ref={} distorted={}",
            pair.pair_index, pair.reference_frame_index, pair.distorted_frame_index
        ));
        if let Some(label) = pair.label_frame_index {
            output.push_str(&format!(" label={label}"));
        }
        output.push('\n');
        output.push_str(&render_metrics(&pair.metrics));
    }
    output.push_str("\nmean metrics\n");
    output.push_str(&render_metrics(&report.mean_metrics));
    output
}

fn render_probe_text(report: &ProbeReport) -> String {
    let info = &report.info;
    let mut output = String::new();
    output.push_str(&format!("file      : {}\n", report.input));
    output.push_str(&format!("size      : {}x{}\n", info.width, info.height));
    output.push_str(&format!(
        "codec     : {}\n",
        info.codec_name.as_deref().unwrap_or("unknown")
    ));
    output.push_str(&format!(
        "fps       : {}\n",
        info.avg_frame_rate
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "unknown".to_string())
    ));
    output.push_str(&format!(
        "frames    : {}\n",
        info.nb_frames
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    ));
    output.push_str(&format!(
        "duration  : {}\n",
        info.duration_seconds
            .map(|value| format!("{value:.3}s"))
            .unwrap_or_else(|| "unknown".to_string())
    ));
    output
}

fn render_formats_text(report: &FormatsReport) -> String {
    let mut output = String::from("image crate adapter format hint:\n");
    report
        .formats
        .iter()
        .for_each(|format| output.push_str(&format!("- {format}\n")));
    output
}

fn render_bundle_info_text(report: &BundleInfoReport) -> String {
    let mut output = String::new();
    output.push_str(&format!("bundle : {}\n", report.input));
    output.push_str(&format!("images : {}\n", report.images.len()));
    report.images.iter().for_each(|image| {
        output.push_str(&format!(
            "[{}] {} {}x{} {} planes={} bytes={}\n",
            image.index,
            image.label.as_deref().unwrap_or("<unlabeled>"),
            image.width,
            image.height,
            image.pixel_format,
            image.planes,
            image.bytes
        ));
        if !image.tags.is_empty() {
            output.push_str(&format!("    tags: {}\n", image.tags.join(",")));
        }
    });
    output
}

fn render_metrics(metrics: &[MetricOutput]) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "{:<32} {:>16}  {:<20}  direction\n",
        "metric", "score", "unit"
    ));
    output.push_str(&format!(
        "{:-<32} {:-<16}  {:-<20}  {:-<12}\n",
        "", "", "", ""
    ));
    metrics.iter().for_each(|metric| {
        output.push_str(&format!(
            "{:<32} {:>16.8}  {:<20}  {:?}\n",
            metric.name, metric.score, metric.unit, metric.direction
        ));
    });
    output
}

fn bundle_info_csv_rows(report: &BundleInfoReport) -> Vec<BundleInfoCsvRow> {
    report
        .images
        .iter()
        .map(|image| BundleInfoCsvRow {
            input: report.input.clone(),
            index: image.index,
            label: image.label.clone().unwrap_or_default(),
            tags: image.tags.join("|"),
            width: image.width,
            height: image.height,
            pixel_format: image.pixel_format.clone(),
            color_space: image.color_space.clone(),
            transfer: image.transfer.clone(),
            range: image.range.clone(),
            planes: image.planes,
            bytes: image.bytes,
        })
        .collect()
}

fn comparison_csv_rows(report: &ComparisonReport) -> Result<Vec<ComparisonCsvRow>> {
    let gate_json = report
        .gate
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?;
    Ok(metric_csv_rows(
        &MetricCsvContext {
            report_kind: "image",
            reference: report.reference.clone().unwrap_or_default(),
            distorted: report.distorted.clone().unwrap_or_default(),
            width: report.dimensions.width,
            height: report.dimensions.height,
            scope: "image",
            frame_index: None,
            pts_seconds: None,
            pair_index: None,
            reference_frame_index: None,
            distorted_frame_index: None,
        },
        &report.metrics,
    )
    .into_iter()
    .map(|row| ComparisonCsvRow {
        report_kind: row.report_kind,
        reference: row.reference,
        distorted: row.distorted,
        width: row.width,
        height: row.height,
        scope: row.scope,
        frame_index: row.frame_index,
        pts_seconds: row.pts_seconds,
        pair_index: row.pair_index,
        reference_frame_index: row.reference_frame_index,
        distorted_frame_index: row.distorted_frame_index,
        metric: row.metric,
        score: row.score,
        unit: row.unit,
        direction: row.direction,
        details_json: row.details_json,
        gate_json: gate_json.clone(),
    })
    .collect())
}

fn suite_csv_rows(report: &imq::ComparisonSuiteReport) -> Vec<SuiteCsvRow> {
    let reference = report.reference.clone().unwrap_or_default();
    report
        .candidates
        .iter()
        .flat_map(|candidate| {
            if candidate.metrics.is_empty() {
                return vec![SuiteCsvRow {
                    reference: reference.clone(),
                    candidate: candidate.label.clone(),
                    candidate_passed: candidate.passed(),
                    overall_rank: candidate.overall_rank,
                    mean_rank: candidate.mean_rank,
                    pareto_optimal: candidate.pareto_optimal,
                    error: candidate.error.clone(),
                    gate_json: candidate
                        .gate
                        .as_ref()
                        .and_then(|gate| serde_json::to_string(gate).ok()),
                    baseline_gate_json: candidate
                        .baseline_gate
                        .as_ref()
                        .and_then(|gate| serde_json::to_string(gate).ok()),
                    metric: None,
                    score: None,
                    unit: None,
                    direction: None,
                    metric_rank: None,
                    baseline_delta: None,
                    details_json: None,
                }];
            }
            candidate
                .metrics
                .iter()
                .map(|metric| SuiteCsvRow {
                    reference: reference.clone(),
                    candidate: candidate.label.clone(),
                    candidate_passed: candidate.passed(),
                    overall_rank: candidate.overall_rank,
                    mean_rank: candidate.mean_rank,
                    pareto_optimal: candidate.pareto_optimal,
                    error: candidate.error.clone(),
                    gate_json: candidate
                        .gate
                        .as_ref()
                        .and_then(|gate| serde_json::to_string(gate).ok()),
                    baseline_gate_json: candidate
                        .baseline_gate
                        .as_ref()
                        .and_then(|gate| serde_json::to_string(gate).ok()),
                    metric: Some(metric.name.clone()),
                    score: Some(metric.score),
                    unit: Some(metric.unit.clone()),
                    direction: Some(format!("{:?}", metric.direction)),
                    metric_rank: candidate.metric_ranks.get(&metric.name).copied(),
                    baseline_delta: candidate.baseline_deltas.get(&metric.name).copied(),
                    details_json: Some(
                        imq::metric_details_to_json(&metric.details).unwrap_or_default(),
                    ),
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn gpu_csv_rows(report: &GpuCliReport) -> Vec<GpuCsvRow> {
    let mut rows = Vec::new();
    for comparison in &report.comparisons {
        if let Some(result) = &comparison.error_result {
            for metric in &result.metrics {
                rows.push(GpuCsvRow {
                    reference: report.reference.clone(),
                    candidate: comparison.candidate.clone(),
                    execution: format!("{:?}", result.execution),
                    fallback_reason: result.fallback_reason.clone(),
                    metric: metric.name.clone(),
                    score: metric.score,
                    unit: metric.unit.clone(),
                    direction: format!("{:?}", metric.direction),
                    details_json: imq::metric_details_to_json(&metric.details).unwrap_or_default(),
                });
            }
        }
        if let Some(result) = &comparison.wssim_result {
            let metric = &result.metric;
            rows.push(GpuCsvRow {
                reference: report.reference.clone(),
                candidate: comparison.candidate.clone(),
                execution: format!("{:?}", result.execution),
                fallback_reason: result.fallback_reason.clone(),
                metric: metric.name.clone(),
                score: metric.score,
                unit: metric.unit.clone(),
                direction: format!("{:?}", metric.direction),
                details_json: imq::metric_details_to_json(&metric.details).unwrap_or_default(),
            });
        }
    }
    rows
}

fn stats_csv_rows(report: &ImageStatsReport) -> Vec<StatsCsvRow> {
    let stats = &report.stats;
    [
        ("red_mean", stats.color_balance.red_mean.to_string()),
        ("green_mean", stats.color_balance.green_mean.to_string()),
        ("blue_mean", stats.color_balance.blue_mean.to_string()),
        ("luma_mean", stats.luma.mean.to_string()),
        ("luma_std_dev", stats.luma.std_dev.to_string()),
        ("luma_p05", stats.luma.p05.to_string()),
        ("luma_p95", stats.luma.p95.to_string()),
        ("hue_degrees", stats.hsv.mean_hue_degrees.to_string()),
        ("saturation_mean", stats.hsv.mean_saturation.to_string()),
        ("value_mean", stats.hsv.mean_value.to_string()),
        ("exposure", stats.tendencies.exposure.clone()),
        ("contrast", stats.tendencies.contrast.clone()),
        ("temperature", stats.tendencies.temperature.clone()),
        ("tint", stats.tendencies.tint.clone()),
        (
            "dominant_channel",
            stats.color_balance.dominant_channel.clone(),
        ),
        (
            "likely_grayscale",
            stats.tendencies.likely_grayscale.to_string(),
        ),
        ("color_cast", stats.tendencies.color_cast.to_string()),
    ]
    .into_iter()
    .map(|(metric, value)| StatsCsvRow {
        input: report.input.clone(),
        width: stats.width,
        height: stats.height,
        pixels: stats.pixels,
        metric: metric.to_string(),
        value,
    })
    .collect()
}

fn video_csv_rows(report: &VideoReport) -> Vec<MetricCsvRow> {
    video_csv_rows_for_kind(report, "video")
}

fn video_csv_rows_for_kind(report: &VideoReport, report_kind: &'static str) -> Vec<MetricCsvRow> {
    let mean_rows = metric_csv_rows(
        &MetricCsvContext {
            report_kind,
            reference: report.reference.clone(),
            distorted: report.distorted.clone(),
            width: report.dimensions.width,
            height: report.dimensions.height,
            scope: "mean",
            frame_index: None,
            pts_seconds: None,
            pair_index: None,
            reference_frame_index: None,
            distorted_frame_index: None,
        },
        &report.mean_metrics,
    );
    let frame_rows = report.frames.iter().flat_map(|frame| {
        metric_csv_rows(
            &MetricCsvContext {
                report_kind,
                reference: report.reference.clone(),
                distorted: report.distorted.clone(),
                width: report.dimensions.width,
                height: report.dimensions.height,
                scope: "frame",
                frame_index: Some(frame.frame_index),
                pts_seconds: frame.pts_seconds,
                pair_index: None,
                reference_frame_index: None,
                distorted_frame_index: None,
            },
            &frame.metrics,
        )
    });
    mean_rows.into_iter().chain(frame_rows).collect()
}

#[cfg(feature = "ffmpeg")]
fn timestamp_video_csv_rows(report: &imq::video::TimestampVideoComparison) -> Vec<MetricCsvRow> {
    let mut rows = video_csv_rows_for_kind(&report.video, "video_timestamp");
    let alignment = &report.alignment;
    let mut diagnostics = vec![
        MetricOutput::new(
            "alignment_planned_pairs",
            alignment.pairs.len() as f64,
            "frames",
            imq::metrics::Direction::Neutral,
        ),
        MetricOutput::new(
            "alignment_unmatched_reference_frames",
            alignment.unmatched_reference_indices.len() as f64,
            "frames",
            imq::metrics::Direction::LowerIsBetter,
        ),
        MetricOutput::new(
            "alignment_unmatched_distorted_frames",
            alignment.unmatched_distorted_indices.len() as f64,
            "frames",
            imq::metrics::Direction::LowerIsBetter,
        ),
        MetricOutput::new(
            "alignment_timestamp_scale",
            alignment.transform.scale,
            "ratio",
            imq::metrics::Direction::Neutral,
        ),
        MetricOutput::new(
            "alignment_timestamp_offset",
            alignment.transform.offset_seconds,
            "seconds",
            imq::metrics::Direction::Neutral,
        ),
    ];
    for (name, value) in [
        (
            "alignment_mean_signed_residual",
            alignment.mean_signed_delta_seconds,
        ),
        (
            "alignment_mean_absolute_residual",
            alignment.mean_absolute_delta_seconds,
        ),
        ("alignment_rms_residual", alignment.rms_delta_seconds),
        ("alignment_max_residual", alignment.max_delta_seconds),
    ] {
        if let Some(value) = value {
            diagnostics.push(MetricOutput::new(
                name,
                value,
                "seconds",
                imq::metrics::Direction::LowerIsBetter,
            ));
        }
    }
    rows.extend(metric_csv_rows(
        &MetricCsvContext {
            report_kind: "video_timestamp",
            reference: report.video.reference.clone(),
            distorted: report.video.distorted.clone(),
            width: report.video.dimensions.width,
            height: report.video.dimensions.height,
            scope: "alignment",
            frame_index: None,
            pts_seconds: None,
            pair_index: None,
            reference_frame_index: None,
            distorted_frame_index: None,
        },
        &diagnostics,
    ));
    rows
}

fn video_pair_csv_rows(report: &VideoFramePairComparisonReport) -> Vec<MetricCsvRow> {
    let mean_rows = metric_csv_rows(
        &MetricCsvContext {
            report_kind: "video_pairs",
            reference: report.reference.clone(),
            distorted: report.distorted.clone(),
            width: report.dimensions.width,
            height: report.dimensions.height,
            scope: "mean",
            frame_index: None,
            pts_seconds: None,
            pair_index: None,
            reference_frame_index: None,
            distorted_frame_index: None,
        },
        &report.mean_metrics,
    );
    let pair_rows = report.pairs.iter().flat_map(|pair| {
        metric_csv_rows(
            &MetricCsvContext {
                report_kind: "video_pairs",
                reference: report.reference.clone(),
                distorted: report.distorted.clone(),
                width: report.dimensions.width,
                height: report.dimensions.height,
                scope: "pair",
                frame_index: pair.label_frame_index,
                pts_seconds: None,
                pair_index: Some(pair.pair_index),
                reference_frame_index: Some(pair.reference_frame_index),
                distorted_frame_index: Some(pair.distorted_frame_index),
            },
            &pair.metrics,
        )
    });
    mean_rows.into_iter().chain(pair_rows).collect()
}

fn metric_csv_rows(ctx: &MetricCsvContext, metrics: &[MetricOutput]) -> Vec<MetricCsvRow> {
    metrics
        .iter()
        .map(|metric| MetricCsvRow {
            report_kind: ctx.report_kind,
            reference: ctx.reference.clone(),
            distorted: ctx.distorted.clone(),
            width: ctx.width,
            height: ctx.height,
            scope: ctx.scope,
            frame_index: ctx.frame_index,
            pts_seconds: ctx.pts_seconds,
            pair_index: ctx.pair_index,
            reference_frame_index: ctx.reference_frame_index,
            distorted_frame_index: ctx.distorted_frame_index,
            metric: metric.name.clone(),
            score: metric.score,
            unit: metric.unit.clone(),
            direction: format!("{:?}", metric.direction),
            details_json: imq::metric_details_to_json(&metric.details).unwrap_or_default(),
        })
        .collect()
}

fn write_sqlite_report(path: Option<&Path>, report: SqlReport<'_>) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let mut conn = open_sqlite(path)?;
    init_sqlite(&conn)?;
    let tx = conn.transaction()?;
    match report {
        SqlReport::Image(report) => {
            let payload = serde_json::to_string(report)?;
            let report_id = insert_sqlite_report(
                &tx,
                &SqlReportInsert {
                    kind: "image",
                    reference: report.reference.as_deref(),
                    distorted: report.distorted.as_deref(),
                    width: report.dimensions.width,
                    height: report.dimensions.height,
                    compared_frames: None,
                    payload_json: &payload,
                },
            )?;
            insert_sqlite_metrics(&tx, report_id, "image", None, None, &report.metrics)?;
        }
        SqlReport::Video(report) => {
            let payload = serde_json::to_string(report)?;
            let report_id = insert_sqlite_report(
                &tx,
                &SqlReportInsert {
                    kind: "video",
                    reference: Some(&report.reference),
                    distorted: Some(&report.distorted),
                    width: report.dimensions.width,
                    height: report.dimensions.height,
                    compared_frames: Some(sql_u64(report.compared_frames)),
                    payload_json: &payload,
                },
            )?;
            insert_sqlite_metrics(&tx, report_id, "mean", None, None, &report.mean_metrics)?;
            for frame in &report.frames {
                insert_sqlite_metrics(
                    &tx,
                    report_id,
                    "frame",
                    Some(sql_u64(frame.frame_index)),
                    frame.pts_seconds,
                    &frame.metrics,
                )?;
            }
        }
        SqlReport::TimestampVideo(report) => {
            let payload = serde_json::to_string(report)?;
            let video = &report.video;
            let report_id = insert_sqlite_report(
                &tx,
                &SqlReportInsert {
                    kind: "video_timestamp",
                    reference: Some(&video.reference),
                    distorted: Some(&video.distorted),
                    width: video.dimensions.width,
                    height: video.dimensions.height,
                    compared_frames: Some(sql_u64(video.compared_frames)),
                    payload_json: &payload,
                },
            )?;
            insert_sqlite_metrics(&tx, report_id, "mean", None, None, &video.mean_metrics)?;
            for frame in &video.frames {
                insert_sqlite_metrics(
                    &tx,
                    report_id,
                    "frame",
                    Some(sql_u64(frame.frame_index)),
                    frame.pts_seconds,
                    &frame.metrics,
                )?;
            }
            let alignment = &report.alignment;
            let mut diagnostics = vec![
                MetricOutput::new(
                    "alignment_planned_pairs",
                    alignment.pairs.len() as f64,
                    "frames",
                    imq::metrics::Direction::Neutral,
                ),
                MetricOutput::new(
                    "alignment_unmatched_reference_frames",
                    alignment.unmatched_reference_indices.len() as f64,
                    "frames",
                    imq::metrics::Direction::LowerIsBetter,
                ),
                MetricOutput::new(
                    "alignment_unmatched_distorted_frames",
                    alignment.unmatched_distorted_indices.len() as f64,
                    "frames",
                    imq::metrics::Direction::LowerIsBetter,
                ),
                MetricOutput::new(
                    "alignment_timestamp_scale",
                    alignment.transform.scale,
                    "ratio",
                    imq::metrics::Direction::Neutral,
                ),
                MetricOutput::new(
                    "alignment_timestamp_offset",
                    alignment.transform.offset_seconds,
                    "seconds",
                    imq::metrics::Direction::Neutral,
                ),
            ];
            for (name, value) in [
                (
                    "alignment_mean_signed_residual",
                    alignment.mean_signed_delta_seconds,
                ),
                (
                    "alignment_mean_absolute_residual",
                    alignment.mean_absolute_delta_seconds,
                ),
                ("alignment_rms_residual", alignment.rms_delta_seconds),
                ("alignment_max_residual", alignment.max_delta_seconds),
            ] {
                if let Some(value) = value {
                    diagnostics.push(MetricOutput::new(
                        name,
                        value,
                        "seconds",
                        imq::metrics::Direction::LowerIsBetter,
                    ));
                }
            }
            insert_sqlite_metrics(&tx, report_id, "alignment", None, None, &diagnostics)?;
        }
        SqlReport::VideoPairs(report) => {
            let payload = serde_json::to_string(report)?;
            let report_id = insert_sqlite_report(
                &tx,
                &SqlReportInsert {
                    kind: "video_pairs",
                    reference: Some(&report.reference),
                    distorted: Some(&report.distorted),
                    width: report.dimensions.width,
                    height: report.dimensions.height,
                    compared_frames: Some(sql_u64(report.pairs.len() as u64)),
                    payload_json: &payload,
                },
            )?;
            insert_sqlite_metrics(&tx, report_id, "mean", None, None, &report.mean_metrics)?;
            for pair in &report.pairs {
                insert_sqlite_metrics(
                    &tx,
                    report_id,
                    "pair",
                    Some(sql_u64(pair.label_frame_index.unwrap_or(pair.pair_index))),
                    None,
                    &pair.metrics,
                )?;
            }
        }
    }
    tx.commit()?;
    Ok(())
}

fn write_sqlite_probe(path: Option<&Path>, report: &ProbeReport) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let conn = open_sqlite(path)?;
    init_sqlite(&conn)?;
    conn.execute(
        "INSERT INTO imq_probe_reports \
         (input, width, height, avg_frame_rate, nb_frames, codec_name, duration_seconds, payload_json) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        rusqlite::params![
            report.input.as_str(),
            i64::from(report.info.width),
            i64::from(report.info.height),
            sqlite_optional_f64(report.info.avg_frame_rate),
            report.info.nb_frames.map(sql_u64),
            report.info.codec_name.as_deref(),
            sqlite_optional_f64(report.info.duration_seconds),
            serde_json::to_string(report)?,
        ],
    )?;
    Ok(())
}

fn write_sqlite_stats(path: Option<&Path>, report: &ImageStatsReport) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let conn = open_sqlite(path)?;
    init_sqlite(&conn)?;
    conn.execute(
        "INSERT INTO imq_image_stats \
         (input, width, height, pixels, exposure, contrast, saturation, temperature, tint, dominant_channel, payload_json) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        rusqlite::params![
            report.input.as_str(),
            i64::from(report.stats.width),
            i64::from(report.stats.height),
            sql_u64(report.stats.pixels),
            report.stats.tendencies.exposure.as_str(),
            report.stats.tendencies.contrast.as_str(),
            report.stats.tendencies.saturation.as_str(),
            report.stats.tendencies.temperature.as_str(),
            report.stats.tendencies.tint.as_str(),
            report.stats.color_balance.dominant_channel.as_str(),
            serde_json::to_string(report)?,
        ],
    )?;
    Ok(())
}

fn write_sqlite_formats(path: Option<&Path>, report: &FormatsReport) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let mut conn = open_sqlite(path)?;
    init_sqlite(&conn)?;
    let tx = conn.transaction()?;
    for format in &report.formats {
        tx.execute(
            "INSERT OR IGNORE INTO imq_formats (format) VALUES (?1)",
            rusqlite::params![format],
        )?;
    }
    tx.commit()?;
    Ok(())
}

fn open_sqlite(path: &Path) -> Result<rusqlite::Connection> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create `{}`", parent.display()))?;
    }
    rusqlite::Connection::open(path)
        .with_context(|| format!("failed to open SQLite database `{}`", path.display()))
}

fn init_sqlite(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS imq_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            reference TEXT,
            distorted TEXT,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            compared_frames INTEGER,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS imq_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER NOT NULL REFERENCES imq_reports(id) ON DELETE CASCADE,
            scope TEXT NOT NULL,
            frame_index INTEGER,
            pts_seconds REAL,
            name TEXT NOT NULL,
            score REAL NOT NULL,
            unit TEXT NOT NULL,
            direction TEXT NOT NULL,
            details_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS imq_probe_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            avg_frame_rate REAL,
            nb_frames INTEGER,
            codec_name TEXT,
            duration_seconds REAL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS imq_formats (
            format TEXT PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS imq_image_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            pixels INTEGER NOT NULL,
            exposure TEXT NOT NULL,
            contrast TEXT NOT NULL,
            saturation TEXT NOT NULL,
            temperature TEXT NOT NULL,
            tint TEXT NOT NULL,
            dominant_channel TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );",
    )?;
    Ok(())
}

fn insert_sqlite_report(conn: &rusqlite::Connection, report: &SqlReportInsert<'_>) -> Result<i64> {
    conn.execute(
        "INSERT INTO imq_reports \
         (kind, reference, distorted, width, height, compared_frames, payload_json) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        rusqlite::params![
            report.kind,
            report.reference,
            report.distorted,
            i64::from(report.width),
            i64::from(report.height),
            report.compared_frames,
            report.payload_json
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

fn insert_sqlite_metrics(
    conn: &rusqlite::Connection,
    report_id: i64,
    scope: &str,
    frame_index: Option<i64>,
    pts_seconds: Option<f64>,
    metrics: &[MetricOutput],
) -> Result<()> {
    for metric in metrics {
        conn.execute(
            "INSERT INTO imq_metrics \
             (report_id, scope, frame_index, pts_seconds, name, score, unit, direction, details_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                report_id,
                scope,
                frame_index,
                sqlite_optional_f64(pts_seconds),
                metric.name.as_str(),
                sqlite_f64(metric.score),
                metric.unit.as_str(),
                format!("{:?}", metric.direction),
                imq::metric_details_to_json(&metric.details)?,
            ],
        )?;
    }
    Ok(())
}

fn sqlite_f64(value: f64) -> rusqlite::types::Value {
    if value.is_finite() {
        rusqlite::types::Value::Real(value)
    } else if value.is_nan() {
        rusqlite::types::Value::Text("NaN".to_string())
    } else if value.is_sign_positive() {
        rusqlite::types::Value::Text("Infinity".to_string())
    } else {
        rusqlite::types::Value::Text("-Infinity".to_string())
    }
}

fn sqlite_optional_f64(value: Option<f64>) -> rusqlite::types::Value {
    value.map_or(rusqlite::types::Value::Null, sqlite_f64)
}

fn sql_u64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

#[cfg(feature = "preview")]
fn run_preview(cmd: PreviewCmd) -> Result<()> {
    use imq::preview::{
        DecodeMode, DisplayMode, FitMode, PreviewOptions, preview_image_bytes, preview_path,
        render_preview,
    };

    if cmd.inputs.is_empty() {
        bail!("at least one input path is required");
    }
    let (columns, rows) = preview_layout(cmd.inputs.len(), cmd.rows, cmd.cols);
    let display = match cmd.display {
        PreviewDisplayArg::Auto => DisplayMode::Auto,
        PreviewDisplayArg::Kitty => DisplayMode::Kitty,
        PreviewDisplayArg::Sixel => DisplayMode::Sixel,
        PreviewDisplayArg::Iterm2 => DisplayMode::Iterm2,
        PreviewDisplayArg::Ansi => DisplayMode::Ansi,
        PreviewDisplayArg::None => DisplayMode::None,
    };
    let default_size = default_preview_size(columns, rows, cmd.display);
    let size = if cmd.actual_size {
        PreviewSize {
            width: u32::MAX,
            height: u32::MAX,
        }
    } else {
        cmd.size.unwrap_or(PreviewSize {
            width: cmd.width.unwrap_or(default_size.width),
            height: cmd.height.unwrap_or(default_size.height),
        })
    };
    let options = PreviewOptions {
        width: size.width,
        height: size.height,
        ffmpeg: cmd.ffmpeg,
        decode: match cmd.decode {
            PreviewDecodeArg::Auto => DecodeMode::Auto,
            PreviewDecodeArg::Hardware => DecodeMode::Hardware,
            PreviewDecodeArg::Cpu => DecodeMode::Cpu,
        },
        fit: match cmd.fit {
            PreviewFitArg::Contain => FitMode::Contain,
            PreviewFitArg::Cover => FitMode::Cover,
            PreviewFitArg::Stretch => FitMode::Stretch,
        },
    };
    let mut previews = Vec::new();
    for input in &cmd.inputs {
        let spec = parse_input_spec(input, &cmd.remote)?;
        let preview = match (&spec, is_video_input(&spec), cmd.video_frame) {
            (InputSpec::Local(path), false, None) => preview_path(path, &options)
                .with_context(|| format!("failed to preview `{}`", path.display()))?,
            (InputSpec::Local(path), true, None) => preview_path(path, &options)
                .with_context(|| format!("failed to preview `{}`", path.display()))?,
            (InputSpec::Local(path), true, Some(frame_index)) => {
                let ffmpeg = imq::video::FfmpegOptions {
                    ffmpeg: options.ffmpeg.clone(),
                    ..Default::default()
                };
                let frame = imq::video::decode_single_frame(path, frame_index, &ffmpeg)?;
                preview_from_rgba_frame(frame, &options, format!("frame:{frame_index}"))?
            }
            (InputSpec::Ssh(spec), false, None) => {
                let remote = cmd.remote.options();
                let bytes = ssh_capture_stdout(
                    spec,
                    &format!("cat -- {}", imq::shell_quote_posix(&spec.path)),
                    &remote,
                )
                .map_err(|error| {
                    anyhow::anyhow!(
                        "remote stream failed for {}.\ncopy fallback is disabled by default.\nUse --remote-transfer copy-input to copy the input explicitly.\n{error}",
                        spec.uri()
                    )
                })?;
                preview_image_bytes(&bytes, &options, spec.uri())?
            }
            (InputSpec::Ssh(spec), true, Some(frame_index)) => {
                let ffmpeg = imq::video::FfmpegOptions {
                    ffmpeg: options.ffmpeg.clone(),
                    ..Default::default()
                };
                let bytes = remote_video_frame_png_bytes(
                    spec,
                    frame_index,
                    &ffmpeg,
                    &cmd.remote.options(),
                )?;
                preview_image_bytes(&bytes, &options, format!("{}#{}", spec.uri(), frame_index))?
            }
            (InputSpec::Ssh(_), true, None) => {
                bail!(
                    "remote video preview requires --video-frame. Use --video-frame N to extract a frame over SSH."
                )
            }
            (InputSpec::Stdin, _, _) => bail!("preview stdin input is not supported yet"),
            (_, false, Some(_)) => bail!("--video-frame can only be used with video inputs"),
        };
        previews.push(preview);
    }

    for (index, input) in cmd.inputs.iter().enumerate() {
        eprintln!("[{}] {}", index + 1, input.display());
    }
    let image = montage_previews(&previews, Some(rows), Some(columns));
    print!("{}", render_preview(&image, display));
    Ok(())
}

#[cfg(feature = "preview")]
fn preview_from_rgba_frame(
    frame: imq::FrameOwned,
    options: &imq::preview::PreviewOptions,
    source: String,
) -> Result<imq::preview::PreviewImage> {
    let view = frame.as_view();
    let plane = view.plane(0)?;
    let image = image::RgbaImage::from_raw(
        frame.dimensions().width,
        frame.dimensions().height,
        plane.data.to_vec(),
    )
    .ok_or_else(|| anyhow::anyhow!("decoded RGBA frame could not be represented as an image"))?;
    Ok(imq::preview::preview_dynamic_image(
        image::DynamicImage::ImageRgba8(image),
        options,
        source,
    ))
}

fn parse_preview_size(input: &str) -> std::result::Result<PreviewSize, String> {
    let Some((width, height)) = input.split_once('x').or_else(|| input.split_once('X')) else {
        return Err("expected WIDTHxHEIGHT, for example 120x60".to_string());
    };
    let width = width
        .parse::<u32>()
        .map_err(|_| "width must be a positive integer".to_string())?;
    let height = height
        .parse::<u32>()
        .map_err(|_| "height must be a positive integer".to_string())?;
    if width == 0 || height == 0 {
        return Err("width and height must be non-zero".to_string());
    }
    Ok(PreviewSize { width, height })
}

#[cfg_attr(not(feature = "preview"), allow(dead_code))]
fn preview_layout(count: usize, rows: Option<usize>, cols: Option<usize>) -> (usize, usize) {
    let count = count.max(1);
    let columns = cols
        .or_else(|| rows.map(|r| count.div_ceil(r.max(1))))
        .unwrap_or_else(|| (count as f64).sqrt().ceil() as usize)
        .max(1);
    let rows = rows
        .unwrap_or_else(|| count.div_ceil(columns))
        .max(count.div_ceil(columns))
        .max(1);
    (columns, rows)
}

#[cfg_attr(not(feature = "preview"), allow(dead_code))]
fn default_preview_size(columns: usize, rows: usize, display: PreviewDisplayArg) -> PreviewSize {
    #[cfg(not(feature = "preview"))]
    let _ = display;
    let native_pixels = {
        #[cfg(feature = "preview")]
        {
            match display {
                PreviewDisplayArg::Kitty | PreviewDisplayArg::Sixel | PreviewDisplayArg::Iterm2 => {
                    true
                }
                PreviewDisplayArg::Auto => {
                    let capabilities = imq::preview::terminal_capabilities();
                    capabilities.kitty || capabilities.sixel || capabilities.iterm2
                }
                PreviewDisplayArg::Ansi | PreviewDisplayArg::None => false,
            }
        }
        #[cfg(not(feature = "preview"))]
        {
            false
        }
    };
    let terminal_width = std::env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(120);
    let terminal_height = std::env::var("LINES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(40);
    let gap = 2 * columns.saturating_sub(1) as u32;
    let cell_width = terminal_width
        .saturating_sub(gap)
        .checked_div(columns as u32)
        .unwrap_or(80)
        .max(16);
    let cell_height = (terminal_height.saturating_sub(rows.saturating_sub(1) as u32))
        .checked_div(rows as u32)
        .unwrap_or(24)
        .max(8);
    let (x_scale, y_scale) = if native_pixels { (10, 20) } else { (1, 2) };
    PreviewSize {
        width: cell_width.saturating_mul(x_scale),
        height: cell_height.saturating_mul(y_scale),
    }
}

#[cfg(not(feature = "preview"))]
fn run_preview(_cmd: PreviewCmd) -> Result<()> {
    bail!("preview support is disabled; rebuild with `--features preview`")
}

#[cfg(feature = "preview")]
fn montage_previews(
    previews: &[imq::preview::PreviewImage],
    rows: Option<usize>,
    cols: Option<usize>,
) -> imq::preview::PreviewImage {
    use imq::preview::PreviewImage;

    let count = previews.len().max(1);
    let columns = cols
        .or_else(|| rows.map(|r| count.div_ceil(r.max(1))))
        .unwrap_or_else(|| (count as f64).sqrt().ceil() as usize)
        .max(1);
    let rows = rows
        .unwrap_or_else(|| count.div_ceil(columns))
        .max(count.div_ceil(columns))
        .max(1);
    let cell_w = previews.iter().map(|p| p.width).max().unwrap_or(1);
    let cell_h = previews.iter().map(|p| p.height).max().unwrap_or(1);
    let gap = 2u32;
    let width = columns as u32 * cell_w + (columns.saturating_sub(1) as u32 * gap);
    let height = rows as u32 * cell_h + (rows.saturating_sub(1) as u32 * gap);
    let mut pixels = vec![[0, 0, 0]; (width * height) as usize];
    for (index, preview) in previews.iter().enumerate() {
        let col = index % columns;
        let row = index / columns;
        let x0 = col as u32 * (cell_w + gap);
        let y0 = row as u32 * (cell_h + gap);
        for y in 0..preview.height {
            for x in 0..preview.width {
                let dst = ((y0 + y) * width + x0 + x) as usize;
                pixels[dst] = preview.pixel(x, y);
            }
        }
    }
    PreviewImage {
        width,
        height,
        source_width: width,
        source_height: height,
        pixels,
        source: "montage".to_string(),
    }
}

fn run_tui(cmd: TuiCmd) -> Result<()> {
    #[cfg(feature = "tui")]
    {
        let (reference, targets, initial_dir) =
            if cmd.targets.is_empty() && cmd.reference.as_ref().is_some_and(|p| p.is_dir()) {
                (None, Vec::new(), cmd.reference)
            } else {
                (cmd.reference, cmd.targets, None)
            };
        tui_app::run(
            reference,
            targets,
            initial_dir,
            cmd.metrics,
            cmd.preview_cache,
            cmd.actual_size,
        )
    }
    #[cfg(not(feature = "tui"))]
    {
        let _ = cmd;
        bail!("TUI support is disabled; rebuild with `--features tui`")
    }
}

fn parse_optional_scale(width: Option<u32>, height: Option<u32>) -> Result<Option<Dimensions>> {
    match (width, height) {
        (Some(w), Some(h)) => Ok(Some(Dimensions::new(w, h)?)),
        (None, None) => Ok(None),
        _ => bail!("--width and --height must be specified together"),
    }
}

#[cfg(feature = "tui")]
mod tui_app {
    use super::*;
    use crossterm::event::{self, Event, KeyCode};
    use crossterm::execute;
    use crossterm::terminal::{
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
    };
    use ratatui::Terminal;
    use ratatui::backend::CrosstermBackend;
    use ratatui::layout::{Constraint, Direction as LayoutDirection, Layout, Rect};
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{
        Block, Borders, Cell, List, ListItem, ListState, Paragraph, Row, Table,
    };
    use ratatui_image::picker::{Picker, ProtocolType};
    use ratatui_image::protocol::StatefulProtocol;
    use ratatui_image::{Resize, StatefulImage};
    use std::collections::{HashMap, VecDeque};
    use std::fs;
    use std::io;
    use std::path::{Path, PathBuf};
    use std::thread;
    use std::time::SystemTime;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Slot {
        Reference,
        Distorted,
    }

    impl Slot {
        fn toggle(self) -> Self {
            match self {
                Self::Reference => Self::Distorted,
                Self::Distorted => Self::Reference,
            }
        }

        fn label(self) -> &'static str {
            match self {
                Self::Reference => "reference",
                Self::Distorted => "distorted",
            }
        }
    }

    #[derive(Debug, Clone)]
    struct FileEntry {
        path: PathBuf,
        name: String,
        is_dir: bool,
        is_image: bool,
        is_video: bool,
        is_other: bool,
        extension: Option<String>,
        modified: Option<SystemTime>,
    }

    #[derive(Debug, Clone)]
    struct Comparison {
        dimensions: Dimensions,
        metrics: Vec<MetricOutput>,
    }

    #[derive(Debug, Clone)]
    struct TargetComparison {
        path: PathBuf,
        comparison: Option<Comparison>,
        error: Option<String>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SortMode {
        Name,
        Kind,
        Extension,
        Modified,
    }

    impl SortMode {
        fn next(self) -> Self {
            match self {
                Self::Name => Self::Kind,
                Self::Kind => Self::Extension,
                Self::Extension => Self::Modified,
                Self::Modified => Self::Name,
            }
        }

        fn label(self) -> &'static str {
            match self {
                Self::Name => "name",
                Self::Kind => "kind",
                Self::Extension => "ext",
                Self::Modified => "modified",
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum PreviewMode {
        Current,
        SideBySide,
        Diff,
    }

    impl PreviewMode {
        fn next(self) -> Self {
            match self {
                Self::Current => Self::SideBySide,
                Self::SideBySide => Self::Diff,
                Self::Diff => Self::Current,
            }
        }

        fn label(self) -> &'static str {
            match self {
                Self::Current => "current",
                Self::SideBySide => "side",
                Self::Diff => "diff",
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum InputMode {
        Normal,
        Filter,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct PreviewCacheKey {
        path: PathBuf,
        width: u32,
        height: u32,
        fit: imq::preview::FitMode,
    }

    #[derive(Debug)]
    struct PreviewCache {
        capacity: usize,
        order: VecDeque<PreviewCacheKey>,
        entries: HashMap<PreviewCacheKey, imq::preview::PreviewImage>,
    }

    impl PreviewCache {
        fn new(capacity: usize) -> Self {
            Self {
                capacity,
                order: VecDeque::new(),
                entries: HashMap::new(),
            }
        }

        fn get(&mut self, key: &PreviewCacheKey) -> Option<imq::preview::PreviewImage> {
            let preview = self.entries.get(key)?.clone();
            self.touch(key);
            Some(preview)
        }

        fn insert(&mut self, key: PreviewCacheKey, preview: imq::preview::PreviewImage) {
            if self.capacity == 0 {
                return;
            }
            if self.entries.contains_key(&key) {
                self.touch(&key);
                self.entries.insert(key, preview);
                return;
            }
            while self.entries.len() >= self.capacity {
                if let Some(oldest) = self.order.pop_front() {
                    self.entries.remove(&oldest);
                } else {
                    break;
                }
            }
            self.order.push_back(key.clone());
            self.entries.insert(key, preview);
        }

        fn touch(&mut self, key: &PreviewCacheKey) {
            if let Some(index) = self.order.iter().position(|existing| existing == key) {
                self.order.remove(index);
            }
            self.order.push_back(key.clone());
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum PreviewResolution {
        Fixed { width: u32, height: u32 },
        Max,
        Actual,
    }

    struct App {
        reference: Option<PathBuf>,
        targets: Vec<PathBuf>,
        metrics_csv: String,
        comparisons: Vec<TargetComparison>,
        preview: Option<imq::preview::PreviewImage>,
        preview_error: Option<String>,
        preview_protocol: Option<StatefulProtocol>,
        preview_picker: Picker,
        preview_cache: PreviewCache,
        preview_resolution: PreviewResolution,
        preview_fit: imq::preview::FitMode,
        preview_mode: PreviewMode,
        cwd: PathBuf,
        entries: Vec<FileEntry>,
        selected: usize,
        active_slot: Slot,
        show_dirs: bool,
        show_other: bool,
        sort_mode: SortMode,
        extension_filter: Option<String>,
        name_filter: String,
        input_mode: InputMode,
        status: String,
        list_state: ListState,
    }

    struct AppConfig {
        reference: Option<PathBuf>,
        targets: Vec<PathBuf>,
        initial_dir: Option<PathBuf>,
        metrics_csv: String,
        preview_picker: Picker,
        preview_cache_capacity: usize,
        preview_resolution: PreviewResolution,
        actual_size: bool,
    }

    impl App {
        fn new(config: AppConfig) -> Result<Self> {
            let cwd = initial_cwd(
                config.initial_dir.as_deref(),
                config.reference.as_deref(),
                config.targets.first().map(PathBuf::as_path),
            )?;
            let mut app = Self {
                reference: config.reference,
                targets: config.targets,
                metrics_csv: config.metrics_csv,
                comparisons: Vec::new(),
                preview: None,
                preview_error: None,
                preview_protocol: None,
                preview_picker: config.preview_picker,
                preview_cache: PreviewCache::new(config.preview_cache_capacity),
                preview_resolution: if config.actual_size {
                    PreviewResolution::Actual
                } else {
                    config.preview_resolution
                },
                preview_fit: imq::preview::FitMode::Contain,
                preview_mode: PreviewMode::Current,
                cwd,
                entries: Vec::new(),
                selected: 0,
                active_slot: Slot::Reference,
                show_dirs: true,
                show_other: false,
                sort_mode: SortMode::Name,
                extension_filter: None,
                name_filter: String::new(),
                input_mode: InputMode::Normal,
                status: String::new(),
                list_state: ListState::default(),
            };
            app.refresh_entries();
            if let Some(path) = app
                .reference
                .clone()
                .or_else(|| app.targets.first().cloned())
            {
                app.select_path(&path);
            }
            app.update_preview();
            app.compare_targets();
            Ok(app)
        }

        fn refresh_entries(&mut self) {
            match read_entries(
                &self.cwd,
                self.show_dirs,
                self.show_other,
                self.extension_filter.as_deref(),
                &self.name_filter,
                self.sort_mode,
            ) {
                Ok(entries) => {
                    self.entries = entries;
                    self.selected = self.selected.min(self.entries.len().saturating_sub(1));
                    self.status = format!(
                        "Browsing {} ({})",
                        self.cwd.display(),
                        self.browser_status()
                    );
                }
                Err(err) => {
                    self.entries.clear();
                    self.selected = 0;
                    self.status = format!("Failed to read {}: {err}", self.cwd.display());
                }
            }
            self.sync_list_state();
            self.update_preview();
        }

        fn browser_status(&self) -> String {
            format!(
                "dirs:{} other:{} sort:{} ext:{} filter:{}",
                on_off(self.show_dirs),
                on_off(self.show_other),
                self.sort_mode.label(),
                self.extension_filter.as_deref().unwrap_or("*"),
                if self.name_filter.is_empty() {
                    "*"
                } else {
                    self.name_filter.as_str()
                }
            )
        }

        fn move_selection(&mut self, delta: isize) {
            if self.entries.is_empty() {
                self.selected = 0;
                self.sync_list_state();
                return;
            }
            let len = self.entries.len() as isize;
            self.selected = (self.selected as isize + delta).rem_euclid(len) as usize;
            self.sync_list_state();
            self.update_preview();
        }

        fn page_selection(&mut self, delta: isize) {
            let step = 10.min(self.entries.len().max(1)) as isize;
            self.move_selection(delta * step);
        }

        fn first_selection(&mut self) {
            self.selected = 0;
            self.sync_list_state();
            self.update_preview();
        }

        fn last_selection(&mut self) {
            self.selected = self.entries.len().saturating_sub(1);
            self.sync_list_state();
            self.update_preview();
        }

        fn parent_directory(&mut self) {
            if let Some(parent) = self.cwd.parent() {
                self.cwd = parent.to_path_buf();
                self.selected = 0;
                self.refresh_entries();
            }
        }

        fn open_or_select(&mut self) {
            let Some(entry) = self.entries.get(self.selected).cloned() else {
                return;
            };
            if entry.is_dir {
                self.cwd = entry.path;
                self.selected = 0;
                self.refresh_entries();
                return;
            }
            if !entry.is_image {
                self.status = "Select a still image file for comparison".to_string();
                return;
            }
            if self.reference.is_none() {
                self.set_reference(entry.path);
            } else {
                self.toggle_target(entry.path);
            }
        }

        fn set_slot(&mut self, slot: Slot) {
            let Some(entry) = self.entries.get(self.selected).cloned() else {
                return;
            };
            if entry.is_dir || !entry.is_image {
                self.status = "Select a still image file for comparison".to_string();
                return;
            }
            match slot {
                Slot::Reference => self.set_reference(entry.path),
                Slot::Distorted => self.toggle_target(entry.path),
            }
            self.active_slot = slot.toggle();
        }

        fn set_reference(&mut self, path: PathBuf) {
            self.reference = Some(path.clone());
            self.targets.retain(|target| target != &path);
            self.status = format!("Set reference {}", short_path(&path));
            self.compare_targets();
        }

        fn toggle_target(&mut self, path: PathBuf) {
            if self.reference.as_ref() == Some(&path) {
                self.status = "Reference is already selected".to_string();
                return;
            }
            if let Some(index) = self.targets.iter().position(|target| target == &path) {
                self.targets.remove(index);
                self.status = format!("Removed target {}", short_path(&path));
            } else {
                self.targets.push(path.clone());
                self.status = format!("Added target {}", short_path(&path));
            }
            self.compare_targets();
        }

        fn compare_targets(&mut self) {
            let Some(reference_path) = &self.reference else {
                self.comparisons.clear();
                self.status = "Select a reference image with r".to_string();
                return;
            };
            let targets = self.targets.clone();
            self.comparisons =
                match compare_target_paths(reference_path, &targets, &self.metrics_csv) {
                    Ok(comparisons) => comparisons,
                    Err(err) => targets
                        .into_iter()
                        .map(|path| TargetComparison {
                            path,
                            comparison: None,
                            error: Some(format!("{err:#}")),
                        })
                        .collect(),
                };
            self.status = format!(
                "Compared {} target(s) with {} worker(s)",
                self.targets.len(),
                comparison_worker_count(self.targets.len())
            );
        }

        fn sync_list_state(&mut self) {
            if self.entries.is_empty() {
                self.list_state.select(None);
            } else {
                self.list_state.select(Some(self.selected));
            }
        }

        fn select_path(&mut self, path: &Path) {
            let target = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            if let Some(index) = self.entries.iter().position(|entry| {
                entry
                    .path
                    .canonicalize()
                    .unwrap_or_else(|_| entry.path.clone())
                    == target
            }) {
                self.selected = index;
                self.sync_list_state();
            }
        }

        fn update_preview(&mut self) {
            let Some(entry) = self.entries.get(self.selected) else {
                self.preview = None;
                self.preview_error = None;
                self.preview_protocol = None;
                return;
            };
            if entry.is_dir || entry.is_other {
                self.preview = None;
                self.preview_error = None;
                self.preview_protocol = None;
                return;
            }
            let (width, height) = self.requested_preview_size();
            let key = PreviewCacheKey {
                path: entry.path.clone(),
                width,
                height,
                fit: self.preview_fit,
            };
            if let Some(preview) = self.preview_cache.get(&key) {
                self.set_preview(preview);
                return;
            }
            let options = imq::preview::PreviewOptions {
                width,
                height,
                fit: self.preview_fit,
                ..Default::default()
            };
            match imq::preview::preview_path(&entry.path, &options) {
                Ok(preview) => {
                    self.preview_cache.insert(key, preview.clone());
                    self.preview_error = None;
                    self.set_preview(preview);
                }
                Err(err) => {
                    self.preview = None;
                    self.preview_error = Some(format!("{err:#}"));
                    self.preview_protocol = None;
                    self.status = format!("Preview failed: {err:#}");
                }
            }
        }

        fn set_preview(&mut self, preview: imq::preview::PreviewImage) {
            self.preview_protocol = preview_to_dynamic(&preview)
                .map(|image| self.preview_picker.new_resize_protocol(image));
            self.preview = Some(preview);
        }

        fn resize_preview(&mut self, larger: bool) {
            self.preview_resolution = match (larger, self.preview_resolution, self.preview.as_ref())
            {
                (true, PreviewResolution::Fixed { width, height }, Some(preview)) => {
                    let next_width = width.saturating_add((width / 4).max(1));
                    let next_height = height.saturating_add((height / 4).max(1));
                    if next_width >= preview.source_width || next_height >= preview.source_height {
                        PreviewResolution::Max
                    } else {
                        PreviewResolution::Fixed {
                            width: next_width,
                            height: next_height,
                        }
                    }
                }
                (true, PreviewResolution::Fixed { width, height }, None) => {
                    PreviewResolution::Fixed {
                        width: width.saturating_add((width / 4).max(1)),
                        height: height.saturating_add((height / 4).max(1)),
                    }
                }
                (true, PreviewResolution::Max, _) => PreviewResolution::Max,
                (true, PreviewResolution::Actual, _) => PreviewResolution::Actual,
                (false, PreviewResolution::Fixed { width, height }, _) => {
                    PreviewResolution::Fixed {
                        width: (width * 4 / 5).max(16),
                        height: (height * 4 / 5).max(8),
                    }
                }
                (false, PreviewResolution::Max, Some(preview)) => PreviewResolution::Fixed {
                    width: (preview.source_width * 4 / 5).max(16),
                    height: (preview.source_height * 4 / 5).max(8),
                },
                (false, PreviewResolution::Max, None) => PreviewResolution::Fixed {
                    width: 192,
                    height: 96,
                },
                (false, PreviewResolution::Actual, _) => PreviewResolution::Actual,
            };
            self.status = format!("Preview size: {}", self.preview_resolution_label());
            self.update_preview();
        }

        fn toggle_actual_size(&mut self) {
            self.preview_resolution = match self.preview_resolution {
                PreviewResolution::Actual => match self.preview.as_ref() {
                    Some(preview) => PreviewResolution::Fixed {
                        width: preview.source_width,
                        height: preview.source_height,
                    },
                    None => PreviewResolution::Fixed {
                        width: 192,
                        height: 96,
                    },
                },
                _ => PreviewResolution::Actual,
            };
            self.status = format!("Preview size: {}", self.preview_resolution_label());
            self.update_preview();
        }

        fn cycle_preview_fit(&mut self) {
            self.preview_fit = match self.preview_fit {
                imq::preview::FitMode::Contain => imq::preview::FitMode::Cover,
                imq::preview::FitMode::Cover => imq::preview::FitMode::Stretch,
                imq::preview::FitMode::Stretch => imq::preview::FitMode::Contain,
            };
            self.status = format!("Preview fit: {}", fit_label(self.preview_fit));
            self.update_preview();
        }

        fn cycle_preview_mode(&mut self) {
            self.preview_mode = self.preview_mode.next();
            self.status = format!("Preview view: {}", self.preview_mode.label());
        }

        fn toggle_dirs(&mut self) {
            self.show_dirs = !self.show_dirs;
            self.refresh_entries();
        }

        fn toggle_other_files(&mut self) {
            self.show_other = !self.show_other;
            self.refresh_entries();
        }

        fn cycle_sort(&mut self) {
            self.sort_mode = self.sort_mode.next();
            self.refresh_entries();
        }

        fn filter_to_selected_extension(&mut self) {
            let Some(entry) = self.entries.get(self.selected) else {
                return;
            };
            if let Some(extension) = &entry.extension {
                self.extension_filter = Some(extension.clone());
                self.refresh_entries();
            }
        }

        fn clear_filters(&mut self) {
            self.extension_filter = None;
            self.name_filter.clear();
            self.input_mode = InputMode::Normal;
            self.refresh_entries();
        }

        fn begin_filter_input(&mut self) {
            self.input_mode = InputMode::Filter;
            self.status = format!("Filter: {}", self.name_filter);
        }

        fn push_filter_char(&mut self, value: char) {
            self.name_filter.push(value);
            self.refresh_entries();
            self.input_mode = InputMode::Filter;
        }

        fn pop_filter_char(&mut self) {
            self.name_filter.pop();
            self.refresh_entries();
            self.input_mode = InputMode::Filter;
        }

        fn selected_target_path(&self) -> Option<&Path> {
            let entry_path = self
                .entries
                .get(self.selected)
                .map(|entry| entry.path.as_path());
            entry_path
                .filter(|path| self.targets.iter().any(|target| target == *path))
                .or_else(|| self.targets.first().map(PathBuf::as_path))
        }

        fn requested_preview_size(&self) -> (u32, u32) {
            match self.preview_resolution {
                PreviewResolution::Fixed { width, height } => (width, height),
                PreviewResolution::Max => (u32::MAX, u32::MAX),
                PreviewResolution::Actual => (u32::MAX, u32::MAX),
            }
        }

        fn preview_resolution_label(&self) -> String {
            match (self.preview_resolution, self.preview.as_ref()) {
                (PreviewResolution::Fixed { width, height }, _) => format!("{width}x{height}"),
                (PreviewResolution::Max, Some(preview)) => {
                    format!("max {}x{}", preview.source_width, preview.source_height)
                }
                (PreviewResolution::Max, None) => "max".to_string(),
                (PreviewResolution::Actual, Some(preview)) => {
                    format!("actual {}x{}", preview.source_width, preview.source_height)
                }
                (PreviewResolution::Actual, None) => "actual".to_string(),
            }
        }
    }

    pub fn run(
        reference: Option<PathBuf>,
        targets: Vec<PathBuf>,
        initial_dir: Option<PathBuf>,
        metrics_csv: String,
        preview_cache_capacity: usize,
        actual_size: bool,
    ) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let preview_picker = Picker::from_query_stdio().unwrap_or_else(|_| picker_from_env());
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        let terminal_size = terminal.size()?;
        let preview_resolution = default_tui_preview_resolution(
            Rect::new(0, 0, terminal_size.width, terminal_size.height),
            &preview_picker,
        );
        let mut app = match App::new(AppConfig {
            reference,
            targets,
            initial_dir,
            metrics_csv,
            preview_picker,
            preview_resolution,
            actual_size,
            preview_cache_capacity,
        }) {
            Ok(app) => app,
            Err(err) => {
                disable_raw_mode()?;
                execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                terminal.show_cursor()?;
                return Err(err);
            }
        };

        let result = loop {
            terminal.draw(|frame| {
                let chunks = Layout::default()
                    .direction(LayoutDirection::Vertical)
                    .constraints([
                        Constraint::Length(7),
                        Constraint::Min(5),
                        Constraint::Length(2),
                    ])
                    .split(frame.area());

                render_header(frame, chunks[0], &app);

                let body = Layout::default()
                    .direction(LayoutDirection::Horizontal)
                    .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
                    .split(chunks[1]);
                render_browser(frame, body[0], &mut app);
                let right = Layout::default()
                    .direction(LayoutDirection::Vertical)
                    .constraints([
                        Constraint::Length(17),
                        Constraint::Length(8),
                        Constraint::Min(5),
                    ])
                    .split(body[1]);
                render_preview_panel(frame, right[0], &mut app);
                render_selection_strip(frame, right[1], &mut app);
                render_metrics(frame, right[2], &app);

                let footer = Paragraph::new(Line::from(vec![
                    Span::styled("/", Style::default().fg(Color::Cyan)),
                    Span::raw(" filter  "),
                    Span::styled("j/k", Style::default().fg(Color::Cyan)),
                    Span::raw(" move  "),
                    Span::styled("h/l", Style::default().fg(Color::Green)),
                    Span::raw(" back/open  "),
                    Span::styled("Enter", Style::default().fg(Color::Green)),
                    Span::raw(" open/set  "),
                    Span::styled("g/G", Style::default().fg(Color::Cyan)),
                    Span::raw(" top/end  "),
                    Span::styled("+/-", Style::default().fg(Color::Magenta)),
                    Span::raw(" size  "),
                    Span::styled("f", Style::default().fg(Color::Magenta)),
                    Span::raw(" fit  "),
                    Span::styled("a", Style::default().fg(Color::Magenta)),
                    Span::raw(" actual  "),
                    Span::styled("v", Style::default().fg(Color::Magenta)),
                    Span::raw(" view  "),
                    Span::styled("r/d", Style::default().fg(Color::Yellow)),
                    Span::raw(" set  "),
                    Span::styled("Space", Style::default().fg(Color::Yellow)),
                    Span::raw(" target  "),
                    Span::styled("q/Esc", Style::default().fg(Color::Red)),
                    Span::raw(" quit"),
                ]));
                frame.render_widget(footer, chunks[2]);
            })?;

            if let Event::Key(key) = event::read()? {
                if app.input_mode == InputMode::Filter {
                    match key.code {
                        KeyCode::Esc | KeyCode::Enter => {
                            app.input_mode = InputMode::Normal;
                            app.status = format!("Filter: {}", app.name_filter);
                        }
                        KeyCode::Backspace => app.pop_filter_char(),
                        KeyCode::Char(value) => app.push_filter_char(value),
                        _ => {}
                    }
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break Ok::<(), anyhow::Error>(()),
                    KeyCode::Up | KeyCode::Char('k') => app.move_selection(-1),
                    KeyCode::Down | KeyCode::Char('j') => app.move_selection(1),
                    KeyCode::PageUp => app.page_selection(-1),
                    KeyCode::PageDown => app.page_selection(1),
                    KeyCode::Home | KeyCode::Char('g') => app.first_selection(),
                    KeyCode::End | KeyCode::Char('G') => app.last_selection(),
                    KeyCode::Left | KeyCode::Char('h') => app.parent_directory(),
                    KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => app.open_or_select(),
                    KeyCode::Tab => {
                        app.active_slot = app.active_slot.toggle();
                        app.status = format!("Target: {}", app.active_slot.label());
                    }
                    KeyCode::Char(' ') => {
                        if let Some(entry) = app.entries.get(app.selected).cloned()
                            && entry.is_image
                        {
                            app.toggle_target(entry.path);
                        }
                    }
                    KeyCode::Char('r') => app.set_slot(Slot::Reference),
                    KeyCode::Char('d') => app.set_slot(Slot::Distorted),
                    KeyCode::Char('c') => app.compare_targets(),
                    KeyCode::Char('+') | KeyCode::Char('=') => app.resize_preview(true),
                    KeyCode::Char('-') => app.resize_preview(false),
                    KeyCode::Char('f') => app.cycle_preview_fit(),
                    KeyCode::Char('a') => app.toggle_actual_size(),
                    KeyCode::Char('v') => app.cycle_preview_mode(),
                    KeyCode::Char('i') => app.toggle_dirs(),
                    KeyCode::Char('o') => app.toggle_other_files(),
                    KeyCode::Char('s') => app.cycle_sort(),
                    KeyCode::Char('e') => app.filter_to_selected_extension(),
                    KeyCode::Char('u') => app.clear_filters(),
                    KeyCode::Char('/') => app.begin_filter_input(),
                    _ => {}
                }
            }
        };

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        result
    }

    fn picker_from_env() -> Picker {
        let mut picker = Picker::halfblocks();
        match imq::preview::auto_display_mode() {
            imq::preview::DisplayMode::Kitty => picker.set_protocol_type(ProtocolType::Kitty),
            imq::preview::DisplayMode::Sixel => picker.set_protocol_type(ProtocolType::Sixel),
            imq::preview::DisplayMode::Iterm2 => picker.set_protocol_type(ProtocolType::Iterm2),
            imq::preview::DisplayMode::Auto
            | imq::preview::DisplayMode::Ansi
            | imq::preview::DisplayMode::None => {}
        }
        picker
    }

    fn render_header(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
        let dims = app
            .comparisons
            .iter()
            .find_map(|target| target.comparison.as_ref())
            .map(|comparison| {
                format!(
                    "{}x{}",
                    comparison.dimensions.width, comparison.dimensions.height
                )
            })
            .unwrap_or_else(|| "--".to_string());
        let active = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let lines = vec![
            Line::from(vec![
                Span::styled("reference ", label_style(Slot::Reference, app.active_slot)),
                Span::raw(display_path(app.reference.as_deref())),
            ]),
            Line::from(vec![
                Span::styled("targets ", label_style(Slot::Distorted, app.active_slot)),
                Span::raw(format!(
                    "{} selected ({})",
                    app.targets.len(),
                    app.preview_mode.label()
                )),
            ]),
            Line::from(vec![
                Span::styled("target ", active),
                Span::raw(app.active_slot.label()),
                Span::raw("    "),
                Span::styled("size ", Style::default().fg(Color::Cyan)),
                Span::raw(dims),
                Span::raw("    "),
                Span::styled("browser ", Style::default().fg(Color::Cyan)),
                Span::raw(app.browser_status()),
                Span::raw("    "),
                Span::styled("metrics ", Style::default().fg(Color::Cyan)),
                Span::raw(app.metrics_csv.as_str()),
            ]),
            Line::from(vec![
                Span::styled("status ", Style::default().fg(Color::Cyan)),
                Span::raw(app.status.as_str()),
            ]),
        ];
        let header = Paragraph::new(lines).block(
            Block::default()
                .title(Span::styled(
                    " imq comparison ",
                    Style::default()
                        .fg(Color::LightCyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue)),
        );
        frame.render_widget(header, area);
    }

    fn render_browser(frame: &mut ratatui::Frame<'_>, area: Rect, app: &mut App) {
        let items = app.entries.iter().map(|entry| {
            let selected_role = entry_role(entry, app);
            let style = if selected_role.is_some() {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD)
            } else if entry.is_dir {
                Style::default().fg(Color::LightBlue)
            } else if entry.is_video {
                Style::default().fg(Color::LightMagenta)
            } else if entry.is_other {
                Style::default().fg(Color::DarkGray)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(Line::from(vec![
                Span::styled(entry_prefix(entry), style),
                Span::styled(selected_role.unwrap_or(" "), style),
                Span::raw(" "),
                Span::styled(entry.name.as_str(), style),
            ]))
        });
        let list = List::new(items)
            .block(
                Block::default()
                    .title(format!(
                        " browser: {} | {} ",
                        app.cwd.display(),
                        app.browser_status()
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Blue)),
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("> ");
        frame.render_stateful_widget(list, area, &mut app.list_state);
    }

    fn render_metrics(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
        if app.comparisons.is_empty() {
            let empty = Paragraph::new(vec![
                Line::from(Span::styled(
                    "No comparison yet",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from("Press r on a reference image, then Space on one or more targets."),
            ])
            .block(
                Block::default()
                    .title(" comparisons ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Blue)),
            );
            frame.render_widget(empty, area);
            return;
        }

        let primary_metric = app
            .comparisons
            .iter()
            .filter_map(|target| target.comparison.as_ref())
            .find_map(|comparison| comparison.metrics.first())
            .map(|metric| metric.name.as_str())
            .unwrap_or("score");
        let rows = app
            .comparisons
            .iter()
            .map(|target| match &target.comparison {
                Some(comparison) => {
                    let primary = comparison.metrics.first();
                    Row::new(vec![
                        Cell::from(short_path(&target.path)),
                        Cell::from(metric_value(comparison, "psnr")),
                        Cell::from(metric_value(comparison, "ssim")),
                        Cell::from(metric_value(comparison, "mse")),
                        Cell::from(primary.map(metric_bar).unwrap_or_default()),
                    ])
                    .style(primary.map(metric_row_style).unwrap_or_default())
                }
                None => Row::new(vec![
                    Cell::from(short_path(&target.path)),
                    Cell::from("err"),
                    Cell::from("err"),
                    Cell::from("err"),
                    Cell::from(target.error.clone().unwrap_or_default()),
                ])
                .style(Style::default().fg(Color::Red)),
            });
        let table = Table::new(
            rows,
            [
                Constraint::Percentage(36),
                Constraint::Length(12),
                Constraint::Length(10),
                Constraint::Length(12),
                Constraint::Min(14),
            ],
        )
        .header(
            Row::new(vec!["target", "psnr", "ssim", "mse", primary_metric]).style(
                Style::default()
                    .fg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD),
            ),
        )
        .block(
            Block::default()
                .title(" comparisons ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue)),
        );
        frame.render_widget(table, area);
    }

    fn render_preview_panel(frame: &mut ratatui::Frame<'_>, area: Rect, app: &mut App) {
        if app.preview_mode != PreviewMode::Current && render_comparison_preview(frame, area, app) {
            return;
        }
        let Some(preview) = &app.preview else {
            let message = app.preview_error.as_deref().map_or_else(
                || "Select an image or video file".to_string(),
                |error| format!("Preview failed:\n{error}"),
            );
            let empty = Paragraph::new(message).block(panel_block(" preview "));
            frame.render_widget(empty, area);
            return;
        };
        let inner = panel_block(format!(
            " preview: {} {} {} ",
            preview.source,
            app.preview_resolution_label(),
            fit_label(app.preview_fit)
        ));
        let content_area = inner.inner(area);
        frame.render_widget(inner, area);
        if let Some(protocol) = &mut app.preview_protocol {
            let resize = match app.preview_resolution {
                PreviewResolution::Actual => Resize::Crop(None),
                _ => Resize::Scale(None),
            };
            frame.render_stateful_widget(
                StatefulImage::default().resize(resize),
                content_area,
                protocol,
            );
            return;
        }
        let (display_cols, display_pixel_rows) = preview_display_cells(preview, content_area, app);
        if display_cols == 0 || display_pixel_rows == 0 {
            return;
        }
        let x_offset = (content_area.width.saturating_sub(display_cols as u16)) / 2;
        let y_offset = (content_area
            .height
            .saturating_sub(display_pixel_rows.div_ceil(2) as u16))
            / 2;
        for y in (0..display_pixel_rows).step_by(2) {
            let spans = (0..display_cols).map(|x| {
                let top =
                    sample_preview_pixel(preview, x, y, display_cols, display_pixel_rows, app);
                let bottom = sample_preview_pixel(
                    preview,
                    x,
                    (y + 1).min(display_pixel_rows.saturating_sub(1)),
                    display_cols,
                    display_pixel_rows,
                    app,
                );
                Span::styled(
                    "▀",
                    Style::default()
                        .fg(Color::Rgb(top[0], top[1], top[2]))
                        .bg(Color::Rgb(bottom[0], bottom[1], bottom[2])),
                )
            });
            frame.render_widget(
                Paragraph::new(Line::from(spans.collect::<Vec<_>>())),
                Rect {
                    x: content_area.x + x_offset,
                    y: content_area.y + y_offset + (y / 2) as u16,
                    width: display_cols as u16,
                    height: 1,
                },
            );
        }
    }

    fn render_comparison_preview(
        frame: &mut ratatui::Frame<'_>,
        area: Rect,
        app: &mut App,
    ) -> bool {
        let (Some(reference), Some(target)) = (
            app.reference.clone(),
            app.selected_target_path().map(Path::to_path_buf),
        ) else {
            return false;
        };
        let block = panel_block(format!(" preview: {} ", app.preview_mode.label()));
        let content_area = block.inner(area);
        frame.render_widget(block, area);
        match app.preview_mode {
            PreviewMode::Current => false,
            PreviewMode::SideBySide => {
                let split = Layout::default()
                    .direction(LayoutDirection::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(content_area);
                render_path_preview(frame, split[0], app, &reference, "reference");
                render_path_preview(frame, split[1], app, &target, "target");
                true
            }
            PreviewMode::Diff => {
                if let Some(diff) = diff_preview_for_area(app, &reference, &target, content_area) {
                    draw_preview_blocks(frame, content_area, &diff, false);
                } else {
                    frame.render_widget(Paragraph::new("Diff preview unavailable"), content_area);
                }
                true
            }
        }
    }

    fn render_path_preview(
        frame: &mut ratatui::Frame<'_>,
        area: Rect,
        app: &mut App,
        path: &Path,
        title: &str,
    ) {
        let block = panel_block(format!(" {title}: {} ", short_path(path)));
        let content_area = block.inner(area);
        frame.render_widget(block, area);
        let Some(preview) = preview_for_area(app, path, content_area) else {
            frame.render_widget(Paragraph::new("preview unavailable"), content_area);
            return;
        };
        draw_preview_blocks(frame, content_area, &preview, false);
    }

    fn render_selection_strip(frame: &mut ratatui::Frame<'_>, area: Rect, app: &mut App) {
        let block = panel_block(" selected ");
        let content_area = block.inner(area);
        frame.render_widget(block, area);
        let mut paths = Vec::new();
        if let Some(reference) = &app.reference {
            paths.push(("ref", reference.clone()));
        }
        paths.extend(app.targets.iter().take(4).map(|path| ("tgt", path.clone())));
        if paths.is_empty() || content_area.width == 0 {
            frame.render_widget(
                Paragraph::new("r: set reference, Space: target"),
                content_area,
            );
            return;
        }
        let width_each = (content_area.width / paths.len() as u16).max(1);
        for (index, (kind, path)) in paths.into_iter().enumerate() {
            let rect = Rect {
                x: content_area.x + index as u16 * width_each,
                y: content_area.y,
                width: if index == 0 {
                    width_each
                } else {
                    width_each.min(
                        content_area
                            .right()
                            .saturating_sub(content_area.x + index as u16 * width_each),
                    )
                },
                height: content_area.height,
            };
            let block = panel_block(format!(" {kind}: {} ", short_path(&path)));
            let inner = block.inner(rect);
            frame.render_widget(block, rect);
            if let Some(preview) = preview_for_area(app, &path, inner) {
                draw_preview_blocks(frame, inner, &preview, false);
            }
        }
    }

    fn preview_display_cells(
        preview: &imq::preview::PreviewImage,
        area: Rect,
        app: &App,
    ) -> (u32, u32) {
        if app.preview_resolution == PreviewResolution::Actual {
            return (
                preview.width.min(u32::from(area.width)),
                preview.height.min(u32::from(area.height).saturating_mul(2)),
            );
        }
        fitted_preview_cells(preview, area)
    }

    fn preview_for_area(
        app: &mut App,
        path: &Path,
        area: Rect,
    ) -> Option<imq::preview::PreviewImage> {
        let width = u32::from(area.width.max(1));
        let height = u32::from(area.height.max(1)).saturating_mul(2);
        let key = PreviewCacheKey {
            path: path.to_path_buf(),
            width,
            height,
            fit: imq::preview::FitMode::Contain,
        };
        if let Some(preview) = app.preview_cache.get(&key) {
            return Some(preview);
        }
        let options = imq::preview::PreviewOptions {
            width,
            height,
            fit: imq::preview::FitMode::Contain,
            ..Default::default()
        };
        let preview = imq::preview::preview_path(path, &options).ok()?;
        app.preview_cache.insert(key, preview.clone());
        Some(preview)
    }

    fn diff_preview_for_area(
        app: &mut App,
        reference: &Path,
        target: &Path,
        area: Rect,
    ) -> Option<imq::preview::PreviewImage> {
        let reference = preview_for_area(app, reference, area)?;
        let target = preview_for_area(app, target, area)?;
        let width = reference.width.min(target.width);
        let height = reference.height.min(target.height);
        if width == 0 || height == 0 {
            return None;
        }
        let pixels = (0..height)
            .flat_map(|y| {
                let reference = &reference;
                let target = &target;
                (0..width).map(move |x| {
                    let a = reference.pixel(x, y);
                    let b = target.pixel(x, y);
                    [
                        a[0].abs_diff(b[0]).saturating_mul(3),
                        a[1].abs_diff(b[1]).saturating_mul(3),
                        a[2].abs_diff(b[2]).saturating_mul(3),
                    ]
                })
            })
            .collect();
        Some(imq::preview::PreviewImage {
            width,
            height,
            source_width: width,
            source_height: height,
            pixels,
            source: "diff".to_string(),
        })
    }

    fn draw_preview_blocks(
        frame: &mut ratatui::Frame<'_>,
        area: Rect,
        preview: &imq::preview::PreviewImage,
        exact: bool,
    ) {
        let (display_cols, display_pixel_rows) = if exact {
            (
                preview.width.min(u32::from(area.width)),
                preview.height.min(u32::from(area.height).saturating_mul(2)),
            )
        } else {
            fitted_preview_cells(preview, area)
        };
        if display_cols == 0 || display_pixel_rows == 0 {
            return;
        }
        let x_offset = (area.width.saturating_sub(display_cols as u16)) / 2;
        let y_offset = (area
            .height
            .saturating_sub(display_pixel_rows.div_ceil(2) as u16))
            / 2;
        for y in (0..display_pixel_rows).step_by(2) {
            let spans = (0..display_cols).map(|x| {
                let top = sample_preview_pixel_scaled(
                    preview,
                    x,
                    y,
                    display_cols,
                    display_pixel_rows,
                    exact,
                );
                let bottom = sample_preview_pixel_scaled(
                    preview,
                    x,
                    (y + 1).min(display_pixel_rows.saturating_sub(1)),
                    display_cols,
                    display_pixel_rows,
                    exact,
                );
                Span::styled(
                    "▀",
                    Style::default()
                        .fg(Color::Rgb(top[0], top[1], top[2]))
                        .bg(Color::Rgb(bottom[0], bottom[1], bottom[2])),
                )
            });
            frame.render_widget(
                Paragraph::new(Line::from(spans.collect::<Vec<_>>())),
                Rect {
                    x: area.x + x_offset,
                    y: area.y + y_offset + (y / 2) as u16,
                    width: display_cols as u16,
                    height: 1,
                },
            );
        }
    }

    fn preview_to_dynamic(preview: &imq::preview::PreviewImage) -> Option<image::DynamicImage> {
        let mut raw = Vec::with_capacity(preview.pixels.len() * 3);
        preview
            .pixels
            .iter()
            .for_each(|pixel| raw.extend_from_slice(pixel));
        image::RgbImage::from_raw(preview.width, preview.height, raw)
            .map(image::DynamicImage::ImageRgb8)
    }

    fn default_tui_preview_resolution(
        terminal_area: Rect,
        preview_picker: &Picker,
    ) -> PreviewResolution {
        let body = Layout::default()
            .direction(LayoutDirection::Vertical)
            .constraints([
                Constraint::Length(7),
                Constraint::Min(5),
                Constraint::Length(2),
            ])
            .split(terminal_area);
        let columns = Layout::default()
            .direction(LayoutDirection::Horizontal)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(body[1]);
        let right = Layout::default()
            .direction(LayoutDirection::Vertical)
            .constraints([
                Constraint::Length(17),
                Constraint::Length(8),
                Constraint::Min(5),
            ])
            .split(columns[1]);
        let content_area = panel_block(" preview ").inner(right[0]);
        let font_size = preview_picker.font_size();
        PreviewResolution::Fixed {
            width: u32::from(content_area.width.max(1)) * u32::from(font_size.width.max(1)),
            height: u32::from(content_area.height.max(1)) * u32::from(font_size.height.max(1)),
        }
    }

    fn fitted_preview_cells(preview: &imq::preview::PreviewImage, area: Rect) -> (u32, u32) {
        let max_width = u32::from(area.width);
        let max_height = u32::from(area.height).saturating_mul(2);
        if max_width == 0 || max_height == 0 || preview.width == 0 || preview.height == 0 {
            return (0, 0);
        }
        let scale = (max_width as f64 / f64::from(preview.width))
            .min(max_height as f64 / f64::from(preview.height));
        let width = (f64::from(preview.width) * scale).round() as u32;
        let height = (f64::from(preview.height) * scale).round() as u32;
        (width.clamp(1, max_width), height.clamp(1, max_height))
    }

    fn sample_preview_pixel(
        preview: &imq::preview::PreviewImage,
        x: u32,
        y: u32,
        display_width: u32,
        display_height: u32,
        app: &App,
    ) -> [u8; 3] {
        if app.preview_resolution == PreviewResolution::Actual {
            return preview.pixel(x, y);
        }
        sample_preview_pixel_scaled(preview, x, y, display_width, display_height, false)
    }

    fn sample_preview_pixel_scaled(
        preview: &imq::preview::PreviewImage,
        x: u32,
        y: u32,
        display_width: u32,
        display_height: u32,
        exact: bool,
    ) -> [u8; 3] {
        if exact {
            return preview.pixel(x, y);
        }
        let source_x = (u64::from(x) * u64::from(preview.width) / u64::from(display_width)) as u32;
        let source_y =
            (u64::from(y) * u64::from(preview.height) / u64::from(display_height)) as u32;
        preview.pixel(source_x, source_y)
    }

    fn compare_target_paths(
        reference_path: &Path,
        targets: &[PathBuf],
        metrics_csv: &str,
    ) -> Result<Vec<TargetComparison>> {
        let reference = image_crate::load_image_path(reference_path)
            .with_context(|| format!("failed to decode `{}`", reference_path.display()))?;
        let metrics = MetricSet::from_csv(metrics_csv)?;
        let workers = comparison_worker_count(targets.len());
        if workers <= 1 {
            return Ok(targets
                .iter()
                .map(|target| compare_one_target(&reference, &metrics, target))
                .collect());
        }
        let mut comparisons = Vec::with_capacity(targets.len());
        for chunk in targets.chunks(workers) {
            let mut chunk_results = thread::scope(|scope| {
                chunk
                    .iter()
                    .map(|target| scope.spawn(|| compare_one_target(&reference, &metrics, target)))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|handle| {
                        handle.join().unwrap_or_else(|_| TargetComparison {
                            path: PathBuf::from("<worker panic>"),
                            comparison: None,
                            error: Some("comparison worker panicked".to_string()),
                        })
                    })
                    .collect::<Vec<_>>()
            });
            comparisons.append(&mut chunk_results);
        }
        Ok(comparisons)
    }

    fn compare_one_target(
        reference: &imq::FrameOwned,
        metrics: &MetricSet,
        target: &Path,
    ) -> TargetComparison {
        match compare_loaded_target(reference, metrics, target) {
            Ok(comparison) => TargetComparison {
                path: target.to_path_buf(),
                comparison: Some(comparison),
                error: None,
            },
            Err(err) => TargetComparison {
                path: target.to_path_buf(),
                comparison: None,
                error: Some(format!("{err:#}")),
            },
        }
    }

    fn compare_loaded_target(
        reference: &imq::FrameOwned,
        metrics: &MetricSet,
        target: &Path,
    ) -> Result<Comparison> {
        let distorted = image_crate::load_image_path(target)
            .with_context(|| format!("failed to decode `{}`", target.display()))?;
        let outputs = metrics.compare(&reference.as_view(), &distorted.as_view())?;
        Ok(Comparison {
            dimensions: reference.dimensions(),
            metrics: outputs,
        })
    }

    fn comparison_worker_count(target_count: usize) -> usize {
        thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(target_count.max(1))
    }

    fn initial_cwd(
        initial_dir: Option<&Path>,
        reference: Option<&Path>,
        distorted: Option<&Path>,
    ) -> Result<PathBuf> {
        if let Some(path) = initial_dir {
            return Ok(path.canonicalize().unwrap_or_else(|_| path.to_path_buf()));
        }
        let candidate = reference.or(distorted);
        if let Some(path) = candidate.and_then(Path::parent)
            && !path.as_os_str().is_empty()
        {
            return Ok(path.canonicalize().unwrap_or_else(|_| path.to_path_buf()));
        }
        std::env::current_dir().context("failed to get current directory")
    }

    fn read_entries(
        cwd: &Path,
        show_dirs: bool,
        show_other: bool,
        extension_filter: Option<&str>,
        name_filter: &str,
        sort_mode: SortMode,
    ) -> Result<Vec<FileEntry>> {
        let mut entries = Vec::new();
        let mut parent_added = false;
        if show_dirs && let Some(parent) = cwd.parent() {
            entries.push(FileEntry {
                path: parent.to_path_buf(),
                name: "..".to_string(),
                is_dir: true,
                is_image: false,
                is_video: false,
                is_other: false,
                extension: None,
                modified: None,
            });
            parent_added = true;
        }
        for item in
            fs::read_dir(cwd).with_context(|| format!("failed to read {}", cwd.display()))?
        {
            let item = item?;
            let file_type = item.file_type()?;
            let is_dir = file_type.is_dir();
            let path = item.path();
            let extension = path_extension(&path);
            let is_image = is_supported_image(&path);
            let is_video = imq::preview::is_video_path(&path);
            let is_other = !is_dir && !is_image && !is_video;
            if is_dir && !show_dirs {
                continue;
            }
            if is_other && !show_other {
                continue;
            }
            if !matches_extension(extension_filter, extension.as_deref(), is_dir) {
                continue;
            }
            let name = item.file_name().to_string_lossy().into_owned();
            if !matches_name_filter(&name, name_filter) {
                continue;
            }
            entries.push(FileEntry {
                path,
                name,
                is_dir,
                is_image,
                is_video,
                is_other,
                extension,
                modified: item
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .ok(),
            });
        }
        let sortable_start = usize::from(parent_added);
        sort_entries(&mut entries[sortable_start..], sort_mode);
        Ok(entries)
    }

    fn sort_entries(entries: &mut [FileEntry], sort_mode: SortMode) {
        entries.sort_by(|a, b| {
            b.is_dir.cmp(&a.is_dir).then_with(|| match sort_mode {
                SortMode::Name => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                SortMode::Kind => entry_kind(a)
                    .cmp(&entry_kind(b))
                    .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())),
                SortMode::Extension => a
                    .extension
                    .cmp(&b.extension)
                    .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())),
                SortMode::Modified => b
                    .modified
                    .cmp(&a.modified)
                    .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())),
            })
        });
    }

    fn matches_extension(filter: Option<&str>, extension: Option<&str>, is_dir: bool) -> bool {
        is_dir
            || match filter {
                Some(filter) => extension == Some(filter),
                None => true,
            }
    }

    fn matches_name_filter(name: &str, filter: &str) -> bool {
        filter.is_empty()
            || name
                .to_ascii_lowercase()
                .contains(&filter.to_ascii_lowercase())
    }

    fn path_extension(path: &Path) -> Option<String> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
    }

    fn entry_kind(entry: &FileEntry) -> u8 {
        match (entry.is_dir, entry.is_image, entry.is_video, entry.is_other) {
            (true, _, _, _) => 0,
            (_, true, _, _) => 1,
            (_, _, true, _) => 2,
            _ => 3,
        }
    }

    fn entry_prefix(entry: &FileEntry) -> &'static str {
        if entry.is_dir {
            "dir  "
        } else if entry.is_image {
            "img  "
        } else if entry.is_video {
            "vid  "
        } else {
            "file "
        }
    }

    fn panel_block<'a>(title: impl Into<ratatui::text::Line<'a>>) -> Block<'a> {
        Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Blue))
    }

    fn is_supported_image(path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                matches!(
                    ext.to_ascii_lowercase().as_str(),
                    "avif"
                        | "bmp"
                        | "dds"
                        | "exr"
                        | "ff"
                        | "gif"
                        | "hdr"
                        | "ico"
                        | "jpg"
                        | "jpeg"
                        | "png"
                        | "pnm"
                        | "qoi"
                        | "tga"
                        | "tif"
                        | "tiff"
                        | "webp"
                )
            })
            .unwrap_or(false)
    }

    fn display_path(path: Option<&Path>) -> String {
        path.map(|p| p.display().to_string())
            .unwrap_or_else(|| "<not selected>".to_string())
    }

    fn short_path(path: &Path) -> String {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| path.display().to_string())
    }

    fn on_off(value: bool) -> &'static str {
        if value { "on" } else { "off" }
    }

    fn entry_role(entry: &FileEntry, app: &App) -> Option<&'static str> {
        if app.reference.as_ref() == Some(&entry.path) {
            Some("R")
        } else if app.targets.iter().any(|target| target == &entry.path) {
            Some("*")
        } else {
            None
        }
    }

    fn label_style(slot: Slot, active_slot: Slot) -> Style {
        let color = match slot {
            Slot::Reference => Color::LightGreen,
            Slot::Distorted => Color::LightMagenta,
        };
        if slot == active_slot {
            Style::default().fg(color).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(color)
        }
    }

    fn direction_style(direction: imq::metrics::Direction) -> Style {
        match direction {
            imq::metrics::Direction::HigherIsBetter => Style::default().fg(Color::LightGreen),
            imq::metrics::Direction::LowerIsBetter => Style::default().fg(Color::LightYellow),
            imq::metrics::Direction::Neutral => Style::default().fg(Color::Gray),
        }
    }

    fn metric_value(comparison: &Comparison, name: &str) -> String {
        comparison
            .metrics
            .iter()
            .find(|metric| metric.name == name)
            .map(format_metric_score)
            .unwrap_or_else(|| "--".to_string())
    }

    fn format_metric_score(metric: &MetricOutput) -> String {
        if metric.score.is_infinite() {
            "inf".to_string()
        } else if metric.score.is_nan() {
            "nan".to_string()
        } else {
            format!("{:.5}", metric.score)
        }
    }

    fn metric_row_style(metric: &MetricOutput) -> Style {
        direction_style(metric.direction)
    }

    fn metric_bar(metric: &MetricOutput) -> String {
        let normalized = match metric.direction {
            imq::metrics::Direction::HigherIsBetter => {
                if metric.score.is_infinite() {
                    1.0
                } else if metric.score.is_finite() {
                    (metric.score.abs() / (metric.score.abs() + 1.0)).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
            imq::metrics::Direction::LowerIsBetter => {
                if metric.score.is_finite() {
                    (1.0 / (1.0 + metric.score.abs())).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
            imq::metrics::Direction::Neutral => 0.5,
        };
        let filled = (normalized * 10.0).round() as usize;
        format!("[{}{}]", "#".repeat(filled), ".".repeat(10 - filled))
    }

    fn fit_label(fit: imq::preview::FitMode) -> &'static str {
        match fit {
            imq::preview::FitMode::Contain => "contain",
            imq::preview::FitMode::Cover => "cover",
            imq::preview::FitMode::Stretch => "stretch",
        }
    }
}
