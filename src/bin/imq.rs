//! Command-line and optional TUI frontend for `imq`.

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use imq::adapters::image_crate;
use imq::metrics::MetricSet;
use imq::report::ComparisonReport;
use imq::{Dimensions, MetricOutput};
use std::path::PathBuf;

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
    /// Compare two still images decoded by the image crate.
    Image(ImageCmd),
    /// Compare two videos by piping RGBA frames from ffmpeg stdout.
    #[cfg(feature = "ffmpeg")]
    Video(VideoCmd),
    /// Decode one video frame with ffmpeg and save it as PNG.
    #[cfg(feature = "ffmpeg")]
    ExtractFrame(ExtractFrameCmd),
    /// Probe a video stream with ffprobe.
    #[cfg(feature = "ffmpeg")]
    Probe(ProbeCmd),
    /// Show image formats available through the image adapter.
    Formats,
    /// Interactive terminal comparison view.
    Tui(TuiCmd),
}

#[derive(Debug, Args)]
struct ImageCmd {
    /// Reference/original image.
    reference: PathBuf,
    /// Distorted/test image.
    distorted: PathBuf,
    /// Comma-separated metrics: psnr,ssim,mse,rmse,mae,maxae; optional domains: psnr:color,mse:all.
    #[arg(long, default_value = "psnr,ssim,mse,mae,maxae")]
    metrics: String,
    /// Print JSON instead of a text table.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct TuiCmd {
    /// Reference/original image.
    reference: PathBuf,
    /// Distorted/test image.
    distorted: PathBuf,
    /// Comma-separated metrics.
    #[arg(long, default_value = "psnr,ssim,mse,mae,maxae")]
    metrics: String,
}

#[cfg(feature = "ffmpeg")]
#[derive(Debug, Args)]
struct VideoCmd {
    /// Reference/original video.
    reference: PathBuf,
    /// Distorted/test video.
    distorted: PathBuf,
    /// Comma-separated metrics.
    #[arg(long, default_value = "psnr,ssim,mse,mae,maxae")]
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
    #[arg(long)]
    json: bool,
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
    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .without_time()
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Image(cmd) => run_image(cmd),
        #[cfg(feature = "ffmpeg")]
        Command::Video(cmd) => run_video(cmd),
        #[cfg(feature = "ffmpeg")]
        Command::ExtractFrame(cmd) => run_extract_frame(cmd),
        #[cfg(feature = "ffmpeg")]
        Command::Probe(cmd) => run_probe(cmd),
        Command::Formats => run_formats(),
        Command::Tui(cmd) => run_tui(cmd),
    }
}

fn run_image(cmd: ImageCmd) -> Result<()> {
    let reference = image_crate::load_image_path(&cmd.reference).with_context(|| {
        format!(
            "failed to decode reference image `{}`",
            cmd.reference.display()
        )
    })?;
    let distorted = image_crate::load_image_path(&cmd.distorted).with_context(|| {
        format!(
            "failed to decode distorted image `{}`",
            cmd.distorted.display()
        )
    })?;
    let metrics = MetricSet::from_csv(&cmd.metrics)?;
    let outputs = metrics.compare(&reference.as_view(), &distorted.as_view())?;
    let report = ComparisonReport::new(
        reference.dimensions(),
        reference.format(),
        distorted.format(),
        outputs,
    )
    .with_labels(
        cmd.reference.display().to_string(),
        cmd.distorted.display().to_string(),
    );

    if cmd.json {
        println!("{}", report.to_json_pretty()?);
    } else {
        println!("reference : {}", cmd.reference.display());
        println!("distorted : {}", cmd.distorted.display());
        println!(
            "size      : {}x{}",
            report.dimensions.width, report.dimensions.height
        );
        print_metrics(&report.metrics);
    }
    Ok(())
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

    if cmd.json {
        println!("{}", report.to_json_pretty()?);
    } else {
        println!("reference       : {}", report.reference);
        println!("distorted       : {}", report.distorted);
        println!(
            "size            : {}x{}",
            report.dimensions.width, report.dimensions.height
        );
        println!("compared frames : {}", report.compared_frames);
        println!("\nmean metrics");
        print_metrics(&report.mean_metrics);
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
    if cmd.json {
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        println!("file      : {}", cmd.input.display());
        println!("size      : {}x{}", info.width, info.height);
        println!(
            "codec     : {}",
            info.codec_name.as_deref().unwrap_or("unknown")
        );
        println!(
            "fps       : {}",
            info.avg_frame_rate
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "unknown".to_string())
        );
        println!(
            "frames    : {}",
            info.nb_frames
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
        println!(
            "duration  : {}",
            info.duration_seconds
                .map(|v| format!("{v:.3}s"))
                .unwrap_or_else(|| "unknown".to_string())
        );
    }
    Ok(())
}

fn run_formats() -> Result<()> {
    println!("image crate adapter format hint:");
    for f in image_crate::enabled_format_hint() {
        println!("- {f}");
    }
    Ok(())
}

fn run_tui(cmd: TuiCmd) -> Result<()> {
    #[cfg(feature = "tui")]
    {
        let reference = image_crate::load_image_path(&cmd.reference)?;
        let distorted = image_crate::load_image_path(&cmd.distorted)?;
        let metrics = MetricSet::from_csv(&cmd.metrics)?;
        let outputs = metrics.compare(&reference.as_view(), &distorted.as_view())?;
        tui_app::run(
            cmd.reference.display().to_string(),
            cmd.distorted.display().to_string(),
            reference.dimensions(),
            outputs,
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

fn print_metrics(metrics: &[MetricOutput]) {
    println!(
        "{:<18} {:>16}  {:<20}  direction",
        "metric", "score", "unit"
    );
    println!("{:-<18} {:-<16}  {:-<20}  {:-<12}", "", "", "", "");
    for metric in metrics {
        println!(
            "{:<18} {:>16.8}  {:<20}  {:?}",
            metric.name, metric.score, metric.unit, metric.direction
        );
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
    use ratatui::layout::{Constraint, Direction as LayoutDirection, Layout};
    use ratatui::style::{Modifier, Style};
    use ratatui::text::Line;
    use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};
    use std::io;

    pub fn run(
        reference: String,
        distorted: String,
        dimensions: Dimensions,
        metrics: Vec<MetricOutput>,
    ) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = loop {
            terminal.draw(|frame| {
                let chunks = Layout::default()
                    .direction(LayoutDirection::Vertical)
                    .constraints([
                        Constraint::Length(5),
                        Constraint::Min(5),
                        Constraint::Length(2),
                    ])
                    .split(frame.area());

                let header = Paragraph::new(vec![
                    Line::from(format!("reference: {reference}")),
                    Line::from(format!("distorted: {distorted}")),
                    Line::from(format!("size: {}x{}", dimensions.width, dimensions.height)),
                ])
                .block(
                    Block::default()
                        .title("imq comparison")
                        .borders(Borders::ALL),
                );
                frame.render_widget(header, chunks[0]);

                let rows = metrics.iter().map(|m| {
                    Row::new(vec![
                        Cell::from(m.name.clone()),
                        Cell::from(format!("{:.8}", m.score)),
                        Cell::from(m.unit.clone()),
                        Cell::from(format!("{:?}", m.direction)),
                    ])
                });
                let table = Table::new(
                    rows,
                    [
                        Constraint::Length(18),
                        Constraint::Length(18),
                        Constraint::Length(22),
                        Constraint::Min(12),
                    ],
                )
                .header(
                    Row::new(vec!["metric", "score", "unit", "direction"])
                        .style(Style::default().add_modifier(Modifier::BOLD)),
                )
                .block(Block::default().title("metrics").borders(Borders::ALL));
                frame.render_widget(table, chunks[1]);

                frame.render_widget(Paragraph::new("q / Esc: quit"), chunks[2]);
            })?;

            if let Event::Key(key) = event::read()? {
                if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
                    break Ok::<(), anyhow::Error>(());
                }
            }
        };

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        result
    }
}
