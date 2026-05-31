//! Command-line and optional TUI frontend for `imq`.

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use imq::adapters::image_crate;
use imq::metrics::MetricSet;
use imq::report::{ComparisonReport, VideoReport};
use imq::{Dimensions, MetricOutput};
use serde::Serialize;
use std::io::Read;
use std::path::{Path, PathBuf};

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
    Compare(CompareCmd),
    /// Compare two still images decoded by the image crate.
    #[command(alias = "i")]
    Image(ImageCmd),
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
    /// Preview images or video thumbnails in the terminal.
    #[command(alias = "p")]
    Preview(PreviewCmd),
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
    /// Comma-separated metrics: psnr,ssim,mse,rmse,mae,maxae; optional domains: psnr:color,mse:all.
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
    output: OutputArgs,
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
    /// Print JSON instead of text.
    #[arg(short, long)]
    json: bool,
    #[command(flatten)]
    output: OutputArgs,
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

#[derive(Debug, Args, Clone, Copy)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum StdinImageFormatArg {
    /// Decode stdin as an encoded image such as PNG/JPEG/WebP.
    Encoded,
    /// Interpret stdin as raw packed pixel bytes.
    Raw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum RawPixelFormatArg {
    Rgb8,
    Rgba8,
    Bgr8,
    Bgra8,
    Luma8,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PreviewDisplayArg {
    Auto,
    Kitty,
    Sixel,
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
        Command::Compare(cmd) => run_compare(cmd),
        Command::Image(cmd) => run_image(cmd),
        Command::Stats(cmd) => run_stats(cmd),
        #[cfg(feature = "ffmpeg")]
        Command::Video(cmd) => run_video(cmd),
        #[cfg(feature = "ffmpeg")]
        Command::ExtractFrame(cmd) => run_extract_frame(cmd),
        #[cfg(feature = "ffmpeg")]
        Command::Probe(cmd) => run_probe(cmd),
        Command::Formats(cmd) => run_formats(cmd),
        Command::Preview(cmd) => run_preview(cmd),
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

    let reference_is_video = is_video_path(&cmd.reference);
    let distorted_is_video = is_video_path(distorted);
    match (reference_is_video, distorted_is_video) {
        (false, false) => {
            let (report, stats) = compare_image_paths(
                &cmd.reference,
                distorted,
                &cmd.metrics,
                cmd.stdin,
                cmd.stats.then_some(cmd.histogram_bins),
            )?;
            write_sqlite_report(cmd.output.sqlite.as_deref(), SqlReport::Image(&report))?;
            if let Some(stats) = &stats {
                write_sqlite_stats(cmd.output.sqlite.as_deref(), &stats.reference)?;
                write_sqlite_stats(cmd.output.sqlite.as_deref(), &stats.distorted)?;
            }
            if let Some(stats) = stats {
                emit_image_comparison_stats_report(
                    &ImageComparisonStatsReport {
                        comparison: report,
                        stats,
                    },
                    output_format(cmd.json, cmd.output.format),
                    &cmd.output,
                )
            } else {
                emit_comparison_report(
                    &report,
                    output_format(cmd.json, cmd.output.format),
                    &cmd.output,
                )
            }
        }
        (true, true) => {
            if cmd.stats {
                bail!("--stats currently applies to still images; omit it for video comparison")
            }
            #[cfg(feature = "ffmpeg")]
            {
                run_video(VideoCmd {
                    reference: cmd.reference,
                    distorted: distorted.clone(),
                    metrics: cmd.metrics,
                    every: cmd.every,
                    max_frames: cmd.max_frames,
                    width: cmd.width,
                    height: cmd.height,
                    ffmpeg: cmd.ffmpeg,
                    ffprobe: cmd.ffprobe,
                    stream: cmd.stream,
                    json: cmd.json,
                    output: cmd.output,
                })
            }
            #[cfg(not(feature = "ffmpeg"))]
            {
                bail!("video support is disabled; rebuild with `--features ffmpeg`")
            }
        }
        _ => bail!("reference and distorted must both be images or both be videos"),
    }
}

fn run_image(cmd: ImageCmd) -> Result<()> {
    let (report, stats) = compare_image_paths(
        &cmd.reference,
        &cmd.distorted,
        &cmd.metrics,
        cmd.stdin,
        cmd.stats.then_some(cmd.histogram_bins),
    )?;
    write_sqlite_report(cmd.output.sqlite.as_deref(), SqlReport::Image(&report))?;
    if let Some(stats) = stats {
        write_sqlite_stats(cmd.output.sqlite.as_deref(), &stats.reference)?;
        write_sqlite_stats(cmd.output.sqlite.as_deref(), &stats.distorted)?;
        emit_image_comparison_stats_report(
            &ImageComparisonStatsReport {
                comparison: report,
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
    Ok(())
}

fn compare_image_paths(
    reference_path: &Path,
    distorted_path: &Path,
    metrics_csv: &str,
    stdin: StdinImageArgs,
    histogram_bins: Option<usize>,
) -> Result<(ComparisonReport, Option<ImageComparisonStats>)> {
    reject_double_stdin(reference_path, Some(distorted_path))?;
    let reference = load_image_input(reference_path, stdin).with_context(|| {
        format!(
            "failed to decode reference image `{}`",
            image_input_label(reference_path)
        )
    })?;
    let distorted = load_image_input(distorted_path, stdin).with_context(|| {
        format!(
            "failed to decode distorted image `{}`",
            image_input_label(distorted_path)
        )
    })?;
    let metrics = MetricSet::from_csv(metrics_csv)?;
    let outputs = metrics.compare(&reference.as_view(), &distorted.as_view())?;
    let report = ComparisonReport::new(
        reference.dimensions(),
        reference.format(),
        distorted.format(),
        outputs,
    )
    .with_labels(
        image_input_label(reference_path),
        image_input_label(distorted_path),
    );
    let stats = histogram_bins
        .map(|histogram_bins| {
            Ok::<_, anyhow::Error>(ImageComparisonStats {
                reference: image_stats_for_frame(
                    image_input_label(reference_path),
                    &reference,
                    histogram_bins,
                )?,
                distorted: image_stats_for_frame(
                    image_input_label(distorted_path),
                    &distorted,
                    histogram_bins,
                )?,
            })
        })
        .transpose()?;
    Ok((report, stats))
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
    let image = load_image_input(input, stdin)
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

fn load_image_input(path: &Path, stdin: StdinImageArgs) -> Result<imq::FrameOwned> {
    if !is_stdin_path(path) {
        return Ok(image_crate::load_image_path(path)?);
    }

    let mut bytes = Vec::new();
    std::io::stdin()
        .read_to_end(&mut bytes)
        .with_context(|| "failed to read image bytes from stdin")?;
    match stdin.stdin_format {
        StdinImageFormatArg::Encoded => Ok(image_crate::decode_image_bytes(&bytes)?),
        StdinImageFormatArg::Raw => {
            let width = stdin
                .raw_width
                .ok_or_else(|| anyhow::anyhow!("--raw-width is required for raw stdin"))?;
            let height = stdin
                .raw_height
                .ok_or_else(|| anyhow::anyhow!("--raw-height is required for raw stdin"))?;
            Ok(imq::FrameOwned::packed_tight(
                bytes,
                width,
                height,
                stdin.raw_pixel_format.into(),
            )?)
        }
    }
}

fn reject_double_stdin(first: &Path, second: Option<&Path>) -> Result<()> {
    if is_stdin_path(first) && second.is_some_and(is_stdin_path) {
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

impl From<RawPixelFormatArg> for imq::PixelFormat {
    fn from(value: RawPixelFormatArg) -> Self {
        match value {
            RawPixelFormatArg::Rgb8 => Self::Rgb8,
            RawPixelFormatArg::Rgba8 => Self::Rgba8,
            RawPixelFormatArg::Bgr8 => Self::Bgr8,
            RawPixelFormatArg::Bgra8 => Self::Bgra8,
            RawPixelFormatArg::Luma8 => Self::Luma8,
        }
    }
}

#[cfg(feature = "ffmpeg")]
fn run_video(cmd: VideoCmd) -> Result<()> {
    let metrics = MetricSet::from_csv(&cmd.metrics)?;
    let mut ffmpeg = imq::video::FfmpegOptions {
        ffmpeg: cmd.ffmpeg,
        ffprobe: cmd.ffprobe,
        stream_index: cmd.stream,
        scale: parse_optional_scale(cmd.width, cmd.height)?,
        input_args: Vec::new(),
    };
    if ffmpeg.scale.is_none() {
        ffmpeg.scale = None;
    }
    let compare = imq::video::VideoCompareOptions {
        every: cmd.every.max(1),
        max_frames: cmd.max_frames,
    };
    let report =
        imq::video::compare_videos(&cmd.reference, &cmd.distorted, &ffmpeg, &compare, &metrics)
            .with_context(|| "video comparison failed")?;

    write_sqlite_report(cmd.output.sqlite.as_deref(), SqlReport::Video(&report))?;
    emit_video_report(
        &report,
        output_format(cmd.json, cmd.output.format),
        &cmd.output,
    )?;
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

#[derive(Debug, Serialize)]
struct FormatsReport {
    formats: Vec<String>,
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

struct MetricCsvContext {
    report_kind: &'static str,
    reference: String,
    distorted: String,
    width: u32,
    height: u32,
    scope: &'static str,
    frame_index: Option<u64>,
    pts_seconds: Option<f64>,
}

enum SqlReport<'a> {
    Image(&'a ComparisonReport),
    Video(&'a VideoReport),
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
        OutputFormatArg::Csv => csv_string(comparison_csv_rows(report))?,
    };
    write_output(output.output.as_deref(), &content)
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
            let mut csv = csv_string(comparison_csv_rows(&report.comparison))?;
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

fn render_metrics(metrics: &[MetricOutput]) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "{:<18} {:>16}  {:<20}  direction\n",
        "metric", "score", "unit"
    ));
    output.push_str(&format!(
        "{:-<18} {:-<16}  {:-<20}  {:-<12}\n",
        "", "", "", ""
    ));
    metrics.iter().for_each(|metric| {
        output.push_str(&format!(
            "{:<18} {:>16.8}  {:<20}  {:?}\n",
            metric.name, metric.score, metric.unit, metric.direction
        ));
    });
    output
}

fn comparison_csv_rows(report: &ComparisonReport) -> Vec<MetricCsvRow> {
    metric_csv_rows(
        &MetricCsvContext {
            report_kind: "image",
            reference: report.reference.clone().unwrap_or_default(),
            distorted: report.distorted.clone().unwrap_or_default(),
            width: report.dimensions.width,
            height: report.dimensions.height,
            scope: "image",
            frame_index: None,
            pts_seconds: None,
        },
        &report.metrics,
    )
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
    let mean_rows = metric_csv_rows(
        &MetricCsvContext {
            report_kind: "video",
            reference: report.reference.clone(),
            distorted: report.distorted.clone(),
            width: report.dimensions.width,
            height: report.dimensions.height,
            scope: "mean",
            frame_index: None,
            pts_seconds: None,
        },
        &report.mean_metrics,
    );
    let frame_rows = report.frames.iter().flat_map(|frame| {
        metric_csv_rows(
            &MetricCsvContext {
                report_kind: "video",
                reference: report.reference.clone(),
                distorted: report.distorted.clone(),
                width: report.dimensions.width,
                height: report.dimensions.height,
                scope: "frame",
                frame_index: Some(frame.frame_index),
                pts_seconds: frame.pts_seconds,
            },
            &frame.metrics,
        )
    });
    mean_rows.into_iter().chain(frame_rows).collect()
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
            metric: metric.name.clone(),
            score: metric.score,
            unit: metric.unit.clone(),
            direction: format!("{:?}", metric.direction),
            details_json: serde_json::to_string(&metric.details).unwrap_or_default(),
        })
        .collect()
}

fn write_sqlite_report(path: Option<&Path>, report: SqlReport<'_>) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let conn = open_sqlite(path)?;
    init_sqlite(&conn)?;
    match report {
        SqlReport::Image(report) => {
            let payload = serde_json::to_string(report)?;
            let report_id = insert_sqlite_report(
                &conn,
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
            insert_sqlite_metrics(&conn, report_id, "image", None, None, &report.metrics)?;
        }
        SqlReport::Video(report) => {
            let payload = serde_json::to_string(report)?;
            let report_id = insert_sqlite_report(
                &conn,
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
            insert_sqlite_metrics(&conn, report_id, "mean", None, None, &report.mean_metrics)?;
            for frame in &report.frames {
                insert_sqlite_metrics(
                    &conn,
                    report_id,
                    "frame",
                    Some(sql_u64(frame.frame_index)),
                    frame.pts_seconds,
                    &frame.metrics,
                )?;
            }
        }
    }
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
            report.info.avg_frame_rate,
            report.info.nb_frames.map(sql_u64),
            report.info.codec_name.as_deref(),
            report.info.duration_seconds,
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
                pts_seconds,
                metric.name.as_str(),
                metric.score,
                metric.unit.as_str(),
                format!("{:?}", metric.direction),
                serde_json::to_string(&metric.details)?,
            ],
        )?;
    }
    Ok(())
}

fn sql_u64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

#[cfg(feature = "preview")]
fn run_preview(cmd: PreviewCmd) -> Result<()> {
    use imq::preview::{
        DecodeMode, DisplayMode, FitMode, PreviewOptions, preview_path, render_preview,
    };

    if cmd.inputs.is_empty() {
        bail!("at least one input path is required");
    }
    let (columns, rows) = preview_layout(cmd.inputs.len(), cmd.rows, cmd.cols);
    let display = match cmd.display {
        PreviewDisplayArg::Auto => DisplayMode::Auto,
        PreviewDisplayArg::Kitty => DisplayMode::Kitty,
        PreviewDisplayArg::Sixel => DisplayMode::Sixel,
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
        let preview = preview_path(input, &options)
            .with_context(|| format!("failed to preview `{}`", input.display()))?;
        previews.push(preview);
    }

    for (index, input) in cmd.inputs.iter().enumerate() {
        eprintln!("[{}] {}", index + 1, input.display());
    }
    let image = montage_previews(&previews, Some(rows), Some(columns));
    print!("{}", render_preview(&image, display));
    Ok(())
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
                PreviewDisplayArg::Kitty | PreviewDisplayArg::Sixel => true,
                PreviewDisplayArg::Auto => {
                    let capabilities = imq::preview::terminal_capabilities();
                    capabilities.kitty || capabilities.sixel
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

fn is_video_path(path: &Path) -> bool {
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
    use ratatui_image::picker::Picker;
    use ratatui_image::protocol::StatefulProtocol;
    use ratatui_image::{Resize, StatefulImage};
    use std::collections::{HashMap, VecDeque};
    use std::fs;
    use std::io;
    use std::path::{Path, PathBuf};
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
            self.comparisons = self
                .targets
                .iter()
                .map(
                    |target| match compare_paths(reference_path, target, &self.metrics_csv) {
                        Ok(comparison) => TargetComparison {
                            path: target.clone(),
                            comparison: Some(comparison),
                            error: None,
                        },
                        Err(err) => TargetComparison {
                            path: target.clone(),
                            comparison: None,
                            error: Some(format!("{err:#}")),
                        },
                    },
                )
                .collect();
            self.status = format!("Compared {} target(s)", self.targets.len());
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
                self.preview_protocol = None;
                return;
            };
            if entry.is_dir || entry.is_other {
                self.preview = None;
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
                    self.set_preview(preview);
                }
                Err(err) => {
                    self.preview = None;
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
        let preview_picker = Picker::from_query_stdio().unwrap_or_else(|_| Picker::halfblocks());
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
                        if let Some(entry) = app.entries.get(app.selected).cloned() {
                            if entry.is_image {
                                app.toggle_target(entry.path);
                            }
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
            let empty =
                Paragraph::new("Select an image or video file").block(panel_block(" preview "));
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

    fn compare_paths(
        reference_path: &Path,
        distorted_path: &Path,
        metrics_csv: &str,
    ) -> Result<Comparison> {
        let reference = image_crate::load_image_path(reference_path)
            .with_context(|| format!("failed to decode `{}`", reference_path.display()))?;
        let distorted = image_crate::load_image_path(distorted_path)
            .with_context(|| format!("failed to decode `{}`", distorted_path.display()))?;
        let metrics = MetricSet::from_csv(metrics_csv)?;
        let outputs = metrics.compare(&reference.as_view(), &distorted.as_view())?;
        Ok(Comparison {
            dimensions: reference.dimensions(),
            metrics: outputs,
        })
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
        if let Some(path) = candidate.and_then(Path::parent) {
            if !path.as_os_str().is_empty() {
                return Ok(path.canonicalize().unwrap_or_else(|_| path.to_path_buf()));
            }
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
        if show_dirs {
            if let Some(parent) = cwd.parent() {
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
