//! Supplemental CLI for advanced quality workflows.

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use imq::adapters::image_crate;
use imq::diff::{DiffImageMode, DiffImageOptions, diff_image};
use imq::external::ExternalMetricCommand;
use imq::gate::{MetricThresholdRule, evaluate_thresholds};
use imq::metrics::{Direction, MetricSet};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "imq-quality")]
#[command(
    about = "Advanced imq quality helpers: diff images, CI gates, and external metric wrappers"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate a diff image or heatmap PNG.
    Diff(DiffCmd),
    /// Compare two still images and evaluate CI threshold rules.
    Gate(GateCmd),
    /// Run an external full-reference metric command.
    External(ExternalCmd),
}

#[derive(Debug, Args)]
struct DiffCmd {
    /// Reference/original still image.
    reference: PathBuf,
    /// Distorted/test still image.
    distorted: PathBuf,
    /// Output PNG path.
    output: PathBuf,
    /// Visual encoding mode.
    #[arg(long, value_enum, default_value_t = DiffModeArg::Heatmap)]
    mode: DiffModeArg,
    /// Difference scale multiplier.
    #[arg(long, default_value_t = 4.0)]
    scale: f64,
    /// Include alpha differences when supported.
    #[arg(long)]
    include_alpha: bool,
}

#[derive(Debug, Args)]
struct GateCmd {
    /// Reference/original still image.
    reference: PathBuf,
    /// Distorted/test still image.
    distorted: PathBuf,
    /// Comma-separated metric list.
    #[arg(short, long, default_value = "psnr,ssim,wssim,ms-ssim,mse,mae,maxae")]
    metrics: String,
    /// Threshold rule such as psnr>=40 or mae<=0.01. Repeat for multiple gates.
    #[arg(long = "rule", required = true)]
    rules: Vec<String>,
    /// Print JSON instead of text.
    #[arg(short, long)]
    json: bool,
}

#[derive(Debug, Args)]
struct ExternalCmd {
    /// Reference/original path passed to the external command.
    reference: PathBuf,
    /// Distorted/test path passed to the external command.
    distorted: PathBuf,
    /// Program to execute.
    #[arg(long)]
    program: String,
    /// Argument template. Use {reference} and {distorted}; repeat for multiple args.
    #[arg(long = "arg", required = true)]
    args: Vec<String>,
    /// Metric name for the parsed first-float stdout value.
    #[arg(long, default_value = "external")]
    metric_name: String,
    /// Metric unit.
    #[arg(long, default_value = "unitless")]
    unit: String,
    /// Metric direction.
    #[arg(long, value_enum, default_value_t = DirectionArg::Higher)]
    direction: DirectionArg,
    /// Print JSON instead of text.
    #[arg(short, long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DiffModeArg {
    AbsoluteRgb,
    Heatmap,
    SignedLuma,
}

impl From<DiffModeArg> for DiffImageMode {
    fn from(value: DiffModeArg) -> Self {
        match value {
            DiffModeArg::AbsoluteRgb => Self::AbsoluteRgb,
            DiffModeArg::Heatmap => Self::Heatmap,
            DiffModeArg::SignedLuma => Self::SignedLuma,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DirectionArg {
    Higher,
    Lower,
    Neutral,
}

impl From<DirectionArg> for Direction {
    fn from(value: DirectionArg) -> Self {
        match value {
            DirectionArg::Higher => Self::HigherIsBetter,
            DirectionArg::Lower => Self::LowerIsBetter,
            DirectionArg::Neutral => Self::Neutral,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Diff(command) => run_diff(command),
        Command::Gate(command) => run_gate(command),
        Command::External(command) => run_external(command),
    }
}

fn run_diff(command: DiffCmd) -> Result<()> {
    let reference = image_crate::load_image_path(&command.reference)
        .with_context(|| format!("failed to load {}", command.reference.display()))?;
    let distorted = image_crate::load_image_path(&command.distorted)
        .with_context(|| format!("failed to load {}", command.distorted.display()))?;
    let image = diff_image(
        &reference.as_view(),
        &distorted.as_view(),
        DiffImageOptions {
            mode: command.mode.into(),
            scale: command.scale,
            include_alpha: command.include_alpha,
        },
    )?;
    image
        .save_png(&command.output)
        .with_context(|| format!("failed to save {}", command.output.display()))?;
    println!("wrote {}", command.output.display());
    Ok(())
}

fn run_gate(command: GateCmd) -> Result<()> {
    let reference = image_crate::load_image_path(&command.reference)
        .with_context(|| format!("failed to load {}", command.reference.display()))?;
    let distorted = image_crate::load_image_path(&command.distorted)
        .with_context(|| format!("failed to load {}", command.distorted.display()))?;
    let metric_set = MetricSet::from_csv(&command.metrics)?;
    let outputs = metric_set.compare(&reference.as_view(), &distorted.as_view())?;
    let rules = command
        .rules
        .iter()
        .map(|rule| MetricThresholdRule::parse(rule))
        .collect::<imq::Result<Vec<_>>>()?;
    let evaluation = evaluate_thresholds(&outputs, &rules);

    if command.json {
        println!("{}", serde_json::to_string_pretty(&evaluation)?);
    } else {
        for check in &evaluation.checks {
            let status = if check.passed { "PASS" } else { "FAIL" };
            println!("{status}: {}", check.message);
        }
    }

    if evaluation.passed {
        Ok(())
    } else {
        bail!("quality gate failed")
    }
}

fn run_external(command: ExternalCmd) -> Result<()> {
    let external = ExternalMetricCommand::first_float(
        command.program,
        command.args,
        command.metric_name,
        command.unit,
        command.direction.into(),
    );
    let run = external.run(&command.reference, &command.distorted)?;
    if command.json {
        println!("{}", serde_json::to_string_pretty(&run)?);
    } else if let Some(metric) = &run.metric {
        println!("{} = {} {}", metric.name, metric.score, metric.unit);
    } else {
        print!("{}", run.stdout);
    }
    Ok(())
}
