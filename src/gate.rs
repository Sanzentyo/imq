//! Threshold gates for CI and regression tests.
//!
//! This module turns metric rows into pass/fail decisions, allowing callers or
//! small CLIs to fail a build when PSNR/SSIM drops below a floor or an error
//! metric rises above a ceiling.

use crate::metrics::MetricOutput;
use crate::{Error, Result};

/// Comparison operator used by a metric threshold rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GateOperator {
    /// Pass when the metric score is greater than or equal to the threshold.
    GreaterOrEqual,
    /// Pass when the metric score is greater than the threshold.
    GreaterThan,
    /// Pass when the metric score is less than or equal to the threshold.
    LessOrEqual,
    /// Pass when the metric score is less than the threshold.
    LessThan,
}

impl GateOperator {
    /// Evaluates this operator against a metric score and threshold.
    pub fn evaluate(self, score: f64, threshold: f64) -> bool {
        match self {
            Self::GreaterOrEqual => score >= threshold,
            Self::GreaterThan => score > threshold,
            Self::LessOrEqual => score <= threshold,
            Self::LessThan => score < threshold,
        }
    }

    /// Returns the textual operator used by [`MetricThresholdRule::parse`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GreaterOrEqual => ">=",
            Self::GreaterThan => ">",
            Self::LessOrEqual => "<=",
            Self::LessThan => "<",
        }
    }
}

/// One metric threshold rule, such as `psnr>=40` or `mae<0.01`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetricThresholdRule {
    /// Metric name to match exactly.
    pub metric: String,
    /// Comparison operator.
    pub operator: GateOperator,
    /// Numeric threshold value.
    pub threshold: f64,
}

impl MetricThresholdRule {
    /// Creates a new rule.
    pub fn new(metric: impl Into<String>, operator: GateOperator, threshold: f64) -> Self {
        Self {
            metric: metric.into(),
            operator,
            threshold,
        }
    }

    /// Parses `metric>=value`, `metric>value`, `metric<=value`, or `metric<value`.
    pub fn parse(input: &str) -> Result<Self> {
        for (token, operator) in [
            (">=", GateOperator::GreaterOrEqual),
            ("<=", GateOperator::LessOrEqual),
            (">", GateOperator::GreaterThan),
            ("<", GateOperator::LessThan),
        ] {
            if let Some((metric, threshold)) = input.split_once(token) {
                let metric = metric.trim();
                if metric.is_empty() {
                    return Err(Error::unsupported("threshold rule metric name is empty"));
                }
                let threshold = threshold.trim().parse::<f64>().map_err(|_| {
                    Error::unsupported(format!("invalid threshold value in `{input}`"))
                })?;
                if !threshold.is_finite() {
                    return Err(Error::unsupported("threshold value must be finite"));
                }
                return Ok(Self::new(metric, operator, threshold));
            }
        }
        Err(Error::unsupported(format!(
            "invalid threshold rule `{input}`; expected metric>=value, metric>value, metric<=value, or metric<value"
        )))
    }
}

/// Result for a single metric threshold check.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetricThresholdCheck {
    /// Rule that was evaluated.
    pub rule: MetricThresholdRule,
    /// Metric score, if the metric was present.
    pub score: Option<f64>,
    /// Whether this check passed.
    pub passed: bool,
    /// Human-readable reason for the decision.
    pub message: String,
}

/// Aggregate pass/fail result for a group of rules.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GateEvaluation {
    /// Whether all checks passed.
    pub passed: bool,
    /// Per-rule decisions.
    pub checks: Vec<MetricThresholdCheck>,
}

impl GateEvaluation {
    /// Returns only failing checks.
    pub fn failures(&self) -> impl Iterator<Item = &MetricThresholdCheck> {
        self.checks.iter().filter(|check| !check.passed)
    }
}

/// Evaluates metric outputs against threshold rules.
pub fn evaluate_thresholds(
    metrics: &[MetricOutput],
    rules: &[MetricThresholdRule],
) -> GateEvaluation {
    let checks = rules
        .iter()
        .cloned()
        .map(|rule| evaluate_one(metrics, rule))
        .collect::<Vec<_>>();
    let passed = checks.iter().all(|check| check.passed);
    GateEvaluation { passed, checks }
}

fn evaluate_one(metrics: &[MetricOutput], rule: MetricThresholdRule) -> MetricThresholdCheck {
    match metrics.iter().find(|metric| metric.name == rule.metric) {
        Some(metric) => {
            let passed = rule.operator.evaluate(metric.score, rule.threshold);
            let message = if passed {
                format!(
                    "{}={} satisfies {}{}",
                    rule.metric,
                    metric.score,
                    rule.operator.as_str(),
                    rule.threshold
                )
            } else {
                format!(
                    "{}={} violates {}{}",
                    rule.metric,
                    metric.score,
                    rule.operator.as_str(),
                    rule.threshold
                )
            };
            MetricThresholdCheck {
                rule,
                score: Some(metric.score),
                passed,
                message,
            }
        }
        None => MetricThresholdCheck {
            message: format!("metric `{}` was not produced", rule.metric),
            rule,
            score: None,
            passed: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Direction;

    #[test]
    fn parses_and_evaluates_rules() {
        let rule = MetricThresholdRule::parse("psnr>=40").unwrap();
        let metrics = vec![MetricOutput::new(
            "psnr",
            41.0,
            "dB",
            Direction::HigherIsBetter,
        )];
        let evaluation = evaluate_thresholds(&metrics, &[rule]);
        assert!(evaluation.passed);
    }

    #[test]
    fn missing_metric_fails() {
        let rule = MetricThresholdRule::parse("ssim>=0.99").unwrap();
        let evaluation = evaluate_thresholds(&[], &[rule]);
        assert!(!evaluation.passed);
        assert_eq!(evaluation.failures().count(), 1);
    }
}
