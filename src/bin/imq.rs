//! Command-line and optional TUI frontend for `imq`.

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
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
    /// Preview images or video thumbnails in the terminal.
    Preview(PreviewCmd),
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
    reference: Option<PathBuf>,
    /// Distorted/test image.
    distorted: Option<PathBuf>,
    /// Comma-separated metrics.
    #[arg(long, default_value = "psnr,ssim,mse,mae,maxae")]
    metrics: String,
}

#[derive(Debug, Args)]
struct PreviewCmd {
    /// Image or video files to preview.
    inputs: Vec<PathBuf>,
    /// Terminal display mode.
    #[arg(long, value_enum, default_value_t = PreviewDisplayArg::Auto)]
    display: PreviewDisplayArg,
    /// Video decode policy.
    #[arg(long, value_enum, default_value_t = PreviewDecodeArg::Auto)]
    decode: PreviewDecodeArg,
    /// Maximum cell width for each input.
    #[arg(long, default_value_t = 48)]
    width: u32,
    /// Maximum cell height for each input.
    #[arg(long, default_value_t = 24)]
    height: u32,
    /// Number of montage rows.
    #[arg(long)]
    rows: Option<usize>,
    /// Number of montage columns.
    #[arg(long)]
    cols: Option<usize>,
    /// ffmpeg executable for video thumbnails.
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: PathBuf,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PreviewDisplayArg {
    Auto,
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
        Command::Preview(cmd) => run_preview(cmd),
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

#[cfg(feature = "preview")]
fn run_preview(cmd: PreviewCmd) -> Result<()> {
    use imq::preview::{DecodeMode, DisplayMode, PreviewOptions, preview_path, render_preview};

    if cmd.inputs.is_empty() {
        bail!("at least one input path is required");
    }
    let options = PreviewOptions {
        width: cmd.width,
        height: cmd.height,
        ffmpeg: cmd.ffmpeg,
        decode: match cmd.decode {
            PreviewDecodeArg::Auto => DecodeMode::Auto,
            PreviewDecodeArg::Hardware => DecodeMode::Hardware,
            PreviewDecodeArg::Cpu => DecodeMode::Cpu,
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
    let image = montage_previews(&previews, cmd.rows, cmd.cols);
    let display = match cmd.display {
        PreviewDisplayArg::Auto => DisplayMode::Auto,
        PreviewDisplayArg::Sixel => DisplayMode::Sixel,
        PreviewDisplayArg::Ansi => DisplayMode::Ansi,
        PreviewDisplayArg::None => DisplayMode::None,
    };
    print!("{}", render_preview(&image, display));
    Ok(())
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
        pixels,
        source: "montage".to_string(),
    }
}

fn run_tui(cmd: TuiCmd) -> Result<()> {
    #[cfg(feature = "tui")]
    {
        tui_app::run(cmd.reference, cmd.distorted, cmd.metrics)
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
    use ratatui::layout::{Constraint, Direction as LayoutDirection, Layout, Rect};
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{
        Block, Borders, Cell, List, ListItem, ListState, Paragraph, Row, Table,
    };
    use std::fs;
    use std::io;
    use std::path::{Path, PathBuf};

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
        is_video: bool,
    }

    #[derive(Debug, Clone)]
    struct Comparison {
        dimensions: Dimensions,
        metrics: Vec<MetricOutput>,
    }

    struct App {
        reference: Option<PathBuf>,
        distorted: Option<PathBuf>,
        metrics_csv: String,
        comparison: Option<Comparison>,
        preview: Option<imq::preview::PreviewImage>,
        cwd: PathBuf,
        entries: Vec<FileEntry>,
        selected: usize,
        active_slot: Slot,
        status: String,
        list_state: ListState,
    }

    impl App {
        fn new(
            reference: Option<PathBuf>,
            distorted: Option<PathBuf>,
            metrics_csv: String,
        ) -> Result<Self> {
            let cwd = initial_cwd(reference.as_deref(), distorted.as_deref())?;
            let mut app = Self {
                reference,
                distorted,
                metrics_csv,
                comparison: None,
                preview: None,
                cwd,
                entries: Vec::new(),
                selected: 0,
                active_slot: Slot::Reference,
                status: String::new(),
                list_state: ListState::default(),
            };
            app.refresh_entries();
            if let Some(path) = app.reference.clone().or_else(|| app.distorted.clone()) {
                app.select_path(&path);
            }
            app.update_preview();
            app.compare_selected();
            Ok(app)
        }

        fn refresh_entries(&mut self) {
            match read_entries(&self.cwd) {
                Ok(entries) => {
                    self.entries = entries;
                    self.selected = self.selected.min(self.entries.len().saturating_sub(1));
                    self.status = format!("Browsing {}", self.cwd.display());
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
            if entry.is_video {
                self.status =
                    "Videos can be previewed, but image comparison needs still images".to_string();
                return;
            }
            match self.active_slot {
                Slot::Reference => self.reference = Some(entry.path),
                Slot::Distorted => self.distorted = Some(entry.path),
            }
            self.status = format!("Set {}", self.active_slot.label());
            self.active_slot = self.active_slot.toggle();
            self.compare_selected();
        }

        fn set_slot(&mut self, slot: Slot) {
            let Some(entry) = self.entries.get(self.selected).cloned() else {
                return;
            };
            if entry.is_dir || entry.is_video {
                self.status = "Select a still image file for comparison".to_string();
                return;
            }
            match slot {
                Slot::Reference => self.reference = Some(entry.path),
                Slot::Distorted => self.distorted = Some(entry.path),
            }
            self.active_slot = slot.toggle();
            self.status = format!("Set {}", slot.label());
            self.compare_selected();
        }

        fn compare_selected(&mut self) {
            let (Some(reference_path), Some(distorted_path)) = (&self.reference, &self.distorted)
            else {
                self.comparison = None;
                if self.reference.is_none() || self.distorted.is_none() {
                    self.status = "Select reference and distorted images".to_string();
                }
                return;
            };

            match compare_paths(reference_path, distorted_path, &self.metrics_csv) {
                Ok(comparison) => {
                    self.status = format!(
                        "Compared {}x{}",
                        comparison.dimensions.width, comparison.dimensions.height
                    );
                    self.comparison = Some(comparison);
                }
                Err(err) => {
                    self.comparison = None;
                    self.status = format!("Comparison failed: {err:#}");
                }
            }
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
                return;
            };
            if entry.is_dir {
                self.preview = None;
                return;
            }
            let options = imq::preview::PreviewOptions {
                width: 34,
                height: 14,
                ..Default::default()
            };
            match imq::preview::preview_path(&entry.path, &options) {
                Ok(preview) => self.preview = Some(preview),
                Err(err) => {
                    self.preview = None;
                    self.status = format!("Preview failed: {err:#}");
                }
            }
        }
    }

    pub fn run(
        reference: Option<PathBuf>,
        distorted: Option<PathBuf>,
        metrics_csv: String,
    ) -> Result<()> {
        let mut app = App::new(reference, distorted, metrics_csv)?;
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
                    .constraints([Constraint::Length(17), Constraint::Min(5)])
                    .split(body[1]);
                render_preview_panel(frame, right[0], &app);
                render_metrics(frame, right[1], &app);

                let footer = Paragraph::new(Line::from(vec![
                    Span::styled("Up/Down", Style::default().fg(Color::Cyan)),
                    Span::raw(" move  "),
                    Span::styled("Enter", Style::default().fg(Color::Green)),
                    Span::raw(" open/set  "),
                    Span::styled("Tab", Style::default().fg(Color::Yellow)),
                    Span::raw(" target  "),
                    Span::styled("r/d", Style::default().fg(Color::Yellow)),
                    Span::raw(" set  "),
                    Span::styled("q/Esc", Style::default().fg(Color::Red)),
                    Span::raw(" quit"),
                ]));
                frame.render_widget(footer, chunks[2]);
            })?;

            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break Ok::<(), anyhow::Error>(()),
                    KeyCode::Up => app.move_selection(-1),
                    KeyCode::Down => app.move_selection(1),
                    KeyCode::PageUp => app.page_selection(-1),
                    KeyCode::PageDown => app.page_selection(1),
                    KeyCode::Enter => app.open_or_select(),
                    KeyCode::Tab => {
                        app.active_slot = app.active_slot.toggle();
                        app.status = format!("Target: {}", app.active_slot.label());
                    }
                    KeyCode::Char('r') => app.set_slot(Slot::Reference),
                    KeyCode::Char('d') => app.set_slot(Slot::Distorted),
                    KeyCode::Char('c') => app.compare_selected(),
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
            .comparison
            .as_ref()
            .map(|c| format!("{}x{}", c.dimensions.width, c.dimensions.height))
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
                Span::styled("distorted ", label_style(Slot::Distorted, app.active_slot)),
                Span::raw(display_path(app.distorted.as_deref())),
            ]),
            Line::from(vec![
                Span::styled("target ", active),
                Span::raw(app.active_slot.label()),
                Span::raw("    "),
                Span::styled("size ", Style::default().fg(Color::Cyan)),
                Span::raw(dims),
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
            let style = if entry.is_dir {
                Style::default().fg(Color::LightBlue)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(Line::from(vec![
                Span::styled(entry_prefix(entry), style),
                Span::styled(entry.name.as_str(), style),
            ]))
        });
        let list = List::new(items)
            .block(
                Block::default()
                    .title(format!(" browser: {} ", app.cwd.display()))
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
        let Some(comparison) = &app.comparison else {
            let empty = Paragraph::new(vec![
                Line::from(Span::styled(
                    "No comparison yet",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from("Select a reference and distorted image from the browser."),
            ])
            .block(
                Block::default()
                    .title(" metrics ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Blue)),
            );
            frame.render_widget(empty, area);
            return;
        };

        let rows = comparison.metrics.iter().map(|metric| {
            Row::new(vec![
                Cell::from(metric.name.clone()),
                Cell::from(format!("{:.8}", metric.score)),
                Cell::from(compact_unit(&metric.unit)),
                Cell::from(direction_label(metric.direction)),
            ])
            .style(direction_style(metric.direction))
        });
        let table = Table::new(
            rows,
            [
                Constraint::Length(10),
                Constraint::Length(14),
                Constraint::Length(10),
                Constraint::Min(8),
            ],
        )
        .header(
            Row::new(vec!["metric", "score", "unit", "dir"]).style(
                Style::default()
                    .fg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD),
            ),
        )
        .block(
            Block::default()
                .title(" metrics ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue)),
        );
        frame.render_widget(table, area);
    }

    fn render_preview_panel(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
        let Some(preview) = &app.preview else {
            let empty =
                Paragraph::new("Select an image or video file").block(panel_block(" preview "));
            frame.render_widget(empty, area);
            return;
        };
        let inner = panel_block(format!(" preview: {} ", preview.source));
        let content_area = inner.inner(area);
        frame.render_widget(inner, area);
        let max_rows = u32::from(content_area.height).min(preview.height);
        let max_cols = u32::from(content_area.width / 2).min(preview.width);
        for y in 0..max_rows {
            let spans = (0..max_cols).map(|x| {
                let [r, g, b] = preview.pixel(x, y);
                Span::styled("██", Style::default().fg(Color::Rgb(r, g, b)))
            });
            frame.render_widget(
                Paragraph::new(Line::from(spans.collect::<Vec<_>>())),
                Rect {
                    x: content_area.x,
                    y: content_area.y + y as u16,
                    width: content_area.width,
                    height: 1,
                },
            );
        }
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

    fn initial_cwd(reference: Option<&Path>, distorted: Option<&Path>) -> Result<PathBuf> {
        let candidate = reference.or(distorted);
        if let Some(path) = candidate.and_then(Path::parent) {
            if !path.as_os_str().is_empty() {
                return Ok(path.canonicalize().unwrap_or_else(|_| path.to_path_buf()));
            }
        }
        std::env::current_dir().context("failed to get current directory")
    }

    fn read_entries(cwd: &Path) -> Result<Vec<FileEntry>> {
        let mut entries = Vec::new();
        if let Some(parent) = cwd.parent() {
            entries.push(FileEntry {
                path: parent.to_path_buf(),
                name: "..".to_string(),
                is_dir: true,
                is_video: false,
            });
        }
        for item in
            fs::read_dir(cwd).with_context(|| format!("failed to read {}", cwd.display()))?
        {
            let item = item?;
            let file_type = item.file_type()?;
            let is_dir = file_type.is_dir();
            let is_video = imq::preview::is_video_path(&item.path());
            if !is_dir && !is_supported_image(&item.path()) && !is_video {
                continue;
            }
            let name = item.file_name().to_string_lossy().into_owned();
            entries.push(FileEntry {
                path: item.path(),
                name,
                is_dir,
                is_video,
            });
        }
        let sortable_start = usize::from(cwd.parent().is_some());
        entries[sortable_start..].sort_by(|a, b| {
            b.is_dir
                .cmp(&a.is_dir)
                .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
        });
        Ok(entries)
    }

    fn entry_prefix(entry: &FileEntry) -> &'static str {
        if entry.is_dir {
            "dir  "
        } else if entry.is_video {
            "vid  "
        } else {
            "img  "
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

    fn direction_label(direction: imq::metrics::Direction) -> &'static str {
        match direction {
            imq::metrics::Direction::HigherIsBetter => "higher",
            imq::metrics::Direction::LowerIsBetter => "lower",
            imq::metrics::Direction::Neutral => "neutral",
        }
    }

    fn compact_unit(unit: &str) -> String {
        match unit {
            "normalized_code^2" => "norm^2".to_string(),
            "normalized_code" => "norm".to_string(),
            other => other.to_string(),
        }
    }
}
