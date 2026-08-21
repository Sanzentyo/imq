//! Supplemental CLI for advanced quality workflows.

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use imq::adapters::image_crate;
use imq::diff::{DiffImageMode, DiffImageOptions, diff_image};
use imq::external::ExternalMetricCommand;
use imq::gate::{
    BaselineThresholdRule, MetricThresholdRule, evaluate_baseline_thresholds, evaluate_thresholds,
};
use imq::metrics::{Direction, MetricSet};
use imq::report::QualityGateReport;
use std::fs;
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
    /// Previous/baseline candidate compared against the same reference.
    #[arg(long)]
    baseline: Option<PathBuf>,
    /// Comma-separated metric list.
    #[arg(short, long, default_value = "psnr,ssim,wssim,ms-ssim,mse,mae,maxae")]
    metrics: String,
    /// Threshold rule such as psnr>=40, mae:color.red_mae<=0.01, or psnr>=baseline-0.5.
    /// Repeat for multiple gates. Baseline expressions require --baseline.
    #[arg(long = "rule", required = true)]
    rules: Vec<String>,
    /// Print JSON instead of text.
    #[arg(short, long)]
    json: bool,
    /// Write a full structured JSON report including metrics and gate decisions.
    #[arg(long)]
    report: Option<PathBuf>,
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
    let (absolute_rules, baseline_rules) = parse_gate_rules(&command.rules)?;
    if !baseline_rules.is_empty() && command.baseline.is_none() {
        bail!("baseline-relative rules require --baseline PATH")
    }
    let evaluation = evaluate_thresholds(&outputs, &absolute_rules);

    let (baseline_label, baseline_format, baseline_outputs, baseline_evaluation) =
        if let Some(path) = command
            .baseline
            .as_ref()
            .filter(|_| !baseline_rules.is_empty())
        {
            let baseline = image_crate::load_image_path(path)
                .with_context(|| format!("failed to load baseline {}", path.display()))?;
            let baseline_outputs = metric_set.compare(&reference.as_view(), &baseline.as_view())?;
            let baseline_evaluation =
                evaluate_baseline_thresholds(&outputs, &baseline_outputs, &baseline_rules);
            (
                Some(path.display().to_string()),
                Some(baseline.format()),
                Some(baseline_outputs),
                Some(baseline_evaluation),
            )
        } else {
            (None, None, None, None)
        };

    let mut report = QualityGateReport::new(
        command.reference.display().to_string(),
        command.distorted.display().to_string(),
        reference.dimensions(),
        reference.format(),
        distorted.format(),
        outputs,
        evaluation,
    );
    if let (Some(label), Some(format), Some(metrics), Some(evaluation)) = (
        baseline_label,
        baseline_format,
        baseline_outputs,
        baseline_evaluation,
    ) {
        report = report.with_baseline(label, format, metrics, evaluation);
    }

    if let Some(path) = &command.report {
        fs::write(path, report.to_json_pretty()?)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }

    if command.json {
        if report.baseline_thresholds.is_some() {
            println!("{}", report.to_json_pretty()?);
        } else {
            // Preserve the original JSON shape for absolute-only invocations.
            println!("{}", serde_json::to_string_pretty(&report.thresholds)?);
        }
    } else {
        for check in &report.thresholds.checks {
            let status = if check.passed { "PASS" } else { "FAIL" };
            println!("{status}: {}", check.message);
        }
        if let Some(evaluation) = &report.baseline_thresholds {
            for check in &evaluation.checks {
                let status = if check.passed { "PASS" } else { "FAIL" };
                println!("{status}: {}", check.message);
            }
        }
        if let Some(path) = &command.report {
            println!("wrote {}", path.display());
        }
    }

    if report.passed {
        Ok(())
    } else {
        bail!("quality gate failed")
    }
}

fn parse_gate_rules(
    rules: &[String],
) -> Result<(Vec<MetricThresholdRule>, Vec<BaselineThresholdRule>)> {
    let mut absolute = Vec::new();
    let mut baseline = Vec::new();
    for rule in rules {
        if rule_rhs(rule).is_some_and(|value| value.trim_start().starts_with("baseline")) {
            baseline.push(BaselineThresholdRule::parse(rule)?);
        } else {
            absolute.push(MetricThresholdRule::parse(rule)?);
        }
    }
    Ok((absolute, baseline))
}

fn rule_rhs(rule: &str) -> Option<&str> {
    [">=", "<=", ">", "<"]
        .into_iter()
        .find_map(|operator| rule.split_once(operator).map(|(_, rhs)| rhs))
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn gate_cli_accepts_baseline_rules_and_report_path() {
        let cli = Cli::try_parse_from([
            "imq-quality",
            "gate",
            "reference.png",
            "candidate.png",
            "--baseline",
            "previous.png",
            "--rule",
            "psnr>=baseline-0.5",
            "--report",
            "gate.json",
        ])
        .unwrap();
        let Command::Gate(gate) = cli.command else {
            panic!("expected gate command");
        };
        assert_eq!(gate.baseline, Some(PathBuf::from("previous.png")));
        assert_eq!(gate.report, Some(PathBuf::from("gate.json")));
    }

    #[test]
    fn partitions_absolute_and_baseline_rules() {
        let rules = vec![
            "ssim>=0.99".to_string(),
            "mae:color.red_mae<=baseline*1.05".to_string(),
        ];
        let (absolute, baseline) = parse_gate_rules(&rules).unwrap();
        assert_eq!(absolute.len(), 1);
        assert_eq!(baseline.len(), 1);
        assert_eq!(baseline[0].metric, "mae:color.red_mae");
    }

    #[test]
    fn gate_writes_full_relative_report() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "imq-quality-report-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let reference = directory.join("reference.png");
        let candidate = directory.join("candidate.png");
        let baseline = directory.join("baseline.png");
        let report = directory.join("report.json");
        RgbImage::from_pixel(1, 1, Rgb([0, 0, 0]))
            .save(&reference)
            .unwrap();
        RgbImage::from_pixel(1, 1, Rgb([11, 11, 11]))
            .save(&candidate)
            .unwrap();
        RgbImage::from_pixel(1, 1, Rgb([10, 10, 10]))
            .save(&baseline)
            .unwrap();

        run_gate(GateCmd {
            reference,
            distorted: candidate,
            baseline: Some(baseline),
            metrics: "mae:color".to_string(),
            rules: vec!["mae:color.abs_error_p99<=baseline*1.2".to_string()],
            json: false,
            report: Some(report.clone()),
        })
        .unwrap();

        let value: serde_json::Value = serde_json::from_slice(&fs::read(&report).unwrap()).unwrap();
        assert_eq!(value["passed"], true);
        assert_eq!(value["dimensions"]["width"], 1);
        assert_eq!(value["candidate_format"]["pixel_format"], "Rgba8");
        assert_eq!(value["candidate_metrics"][0]["name"], "mae:color");
        assert_eq!(value["baseline_thresholds"]["checks"][0]["passed"], true);
        fs::remove_dir_all(directory).unwrap();
    }
}
