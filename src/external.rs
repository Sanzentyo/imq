//! External metric command wrapper.
//!
//! Some metrics, such as VMAF or model-heavy perceptual scores, are better kept
//! outside the Sans-I/O core. This module provides a small, explicit bridge that
//! replaces `{reference}` and `{distorted}` placeholders, runs a command, and
//! converts its stdout into an [`crate::metrics::MetricOutput`].

use crate::metrics::{Direction, MetricOutput};
use crate::{Error, Result};
use std::path::Path;
use std::process::Command;

/// How stdout from an external metric command should be interpreted.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExternalMetricParser {
    /// Parse the first finite floating-point token in stdout.
    FirstFloat,
    /// Keep stdout/stderr only; no numeric metric output is produced.
    RawOnly,
}

/// Command specification for an external full-reference metric.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExternalMetricCommand {
    /// Program path or executable name.
    pub program: String,
    /// Arguments. `{reference}` and `{distorted}` are replaced before execution.
    pub args: Vec<String>,
    /// Metric name to emit when parsing succeeds.
    pub metric_name: String,
    /// Metric unit.
    pub unit: String,
    /// Metric direction.
    pub direction: Direction,
    /// Stdout parser.
    pub parser: ExternalMetricParser,
}

impl ExternalMetricCommand {
    /// Creates a command that parses the first float from stdout.
    pub fn first_float(
        program: impl Into<String>,
        args: Vec<String>,
        metric_name: impl Into<String>,
        unit: impl Into<String>,
        direction: Direction,
    ) -> Self {
        Self {
            program: program.into(),
            args,
            metric_name: metric_name.into(),
            unit: unit.into(),
            direction,
            parser: ExternalMetricParser::FirstFloat,
        }
    }

    /// Builds concrete command arguments for a reference/distorted pair.
    pub fn expanded_args(&self, reference: &Path, distorted: &Path) -> Vec<String> {
        let reference = reference.to_string_lossy();
        let distorted = distorted.to_string_lossy();
        self.args
            .iter()
            .map(|arg| {
                arg.replace("{reference}", &reference)
                    .replace("{distorted}", &distorted)
            })
            .collect()
    }

    /// Runs the external command.
    pub fn run(
        &self,
        reference: impl AsRef<Path>,
        distorted: impl AsRef<Path>,
    ) -> Result<ExternalMetricRun> {
        let reference = reference.as_ref();
        let distorted = distorted.as_ref();
        let args = self.expanded_args(reference, distorted);
        let output = Command::new(&self.program).args(&args).output()?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let status_code = output.status.code();
        if !output.status.success() {
            return Err(Error::ProcessFailed {
                program: self.program.clone(),
                status: output.status.to_string(),
                stderr,
            });
        }
        let metric = match self.parser {
            ExternalMetricParser::FirstFloat => Some(MetricOutput::new(
                self.metric_name.clone(),
                parse_first_float(&stdout)?,
                self.unit.clone(),
                self.direction,
            )),
            ExternalMetricParser::RawOnly => None,
        };
        Ok(ExternalMetricRun {
            program: self.program.clone(),
            args,
            status_code,
            stdout,
            stderr,
            metric,
        })
    }
}

/// Captured result of an external metric command.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExternalMetricRun {
    /// Program that was executed.
    pub program: String,
    /// Expanded arguments.
    pub args: Vec<String>,
    /// Process status code when available.
    pub status_code: Option<i32>,
    /// Captured stdout.
    pub stdout: String,
    /// Captured stderr.
    pub stderr: String,
    /// Parsed metric output when requested.
    pub metric: Option<MetricOutput>,
}

fn parse_first_float(stdout: &str) -> Result<f64> {
    for token in
        stdout.split(|ch: char| !(ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E')))
    {
        if token.is_empty() || token == "+" || token == "-" || token == "." {
            continue;
        }
        if let Ok(value) = token.parse::<f64>() {
            if value.is_finite() {
                return Ok(value);
            }
        }
    }
    Err(Error::unsupported(
        "external metric stdout did not contain a finite float",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_first_float() {
        assert_eq!(parse_first_float("score: 97.5\n").unwrap(), 97.5);
    }

    #[test]
    fn expands_placeholders() {
        let command = ExternalMetricCommand::first_float(
            "tool",
            vec!["--ref".into(), "{reference}".into(), "{distorted}".into()],
            "vmaf",
            "unitless",
            Direction::HigherIsBetter,
        );
        let args = command.expanded_args(Path::new("a.mp4"), Path::new("b.mp4"));
        assert_eq!(args, vec!["--ref", "a.mp4", "b.mp4"]);
    }
}
