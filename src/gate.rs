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

/// One metric threshold rule, such as `psnr>=40`, `mae<0.01`, or
/// `mae:color.red_mae<=0.02` for a metric detail value.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetricThresholdRule {
    /// Metric name or `metric.detail` selector to match.
    pub metric: String,
    /// Comparison operator.
    pub operator: GateOperator,
    /// Numeric threshold value.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
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
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
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

/// A threshold rule whose right-hand side is derived from a baseline result.
///
/// Supported expressions are `metric>=baseline`, `metric>=baseline-0.5`,
/// `metric<=baseline*1.05`, and `metric<=baseline/2` (with any gate operator).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BaselineThresholdRule {
    /// Metric name or `metric.detail` selector to evaluate.
    pub metric: String,
    /// Comparison operator applied to the candidate and derived threshold.
    pub operator: GateOperator,
    /// Multiplier applied to the baseline value.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub baseline_scale: f64,
    /// Offset added after multiplying the baseline value.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub offset: f64,
}

impl BaselineThresholdRule {
    /// Creates a baseline-relative rule.
    pub fn new(
        metric: impl Into<String>,
        operator: GateOperator,
        baseline_scale: f64,
        offset: f64,
    ) -> Self {
        Self {
            metric: metric.into(),
            operator,
            baseline_scale,
            offset,
        }
    }

    /// Parses a rule with a `baseline` expression on its right-hand side.
    pub fn parse(input: &str) -> Result<Self> {
        let (metric, operator, expression) = split_rule(input)?;
        let (baseline_scale, offset) = parse_baseline_expression(expression, input)?;
        Ok(Self::new(metric, operator, baseline_scale, offset))
    }

    /// Computes the concrete threshold for one baseline value.
    pub fn threshold(&self, baseline: f64) -> f64 {
        baseline * self.baseline_scale + self.offset
    }
}

/// Result for one baseline-relative threshold check.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BaselineThresholdCheck {
    /// Rule that was evaluated.
    pub rule: BaselineThresholdRule,
    /// Candidate metric or detail value, when found.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub candidate_score: Option<f64>,
    /// Baseline metric or detail value, when found.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub baseline_score: Option<f64>,
    /// Concrete threshold derived from the baseline, when available.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub threshold: Option<f64>,
    /// Whether this check passed.
    pub passed: bool,
    /// Human-readable reason for the decision.
    pub message: String,
}

/// Aggregate result for baseline-relative threshold rules.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BaselineGateEvaluation {
    /// Whether all checks passed.
    pub passed: bool,
    /// Per-rule decisions.
    pub checks: Vec<BaselineThresholdCheck>,
}

impl BaselineGateEvaluation {
    /// Returns only failing checks.
    pub fn failures(&self) -> impl Iterator<Item = &BaselineThresholdCheck> {
        self.checks.iter().filter(|check| !check.passed)
    }
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

/// Evaluates candidate metrics against thresholds derived from baseline metrics.
pub fn evaluate_baseline_thresholds(
    candidate_metrics: &[MetricOutput],
    baseline_metrics: &[MetricOutput],
    rules: &[BaselineThresholdRule],
) -> BaselineGateEvaluation {
    let checks = rules
        .iter()
        .cloned()
        .map(|rule| evaluate_baseline_one(candidate_metrics, baseline_metrics, rule))
        .collect::<Vec<_>>();
    let passed = checks.iter().all(|check| check.passed);
    BaselineGateEvaluation { passed, checks }
}

fn evaluate_one(metrics: &[MetricOutput], rule: MetricThresholdRule) -> MetricThresholdCheck {
    match metric_value(metrics, &rule.metric) {
        MetricValueLookup::Found(score) => {
            let passed = rule.operator.evaluate(score, rule.threshold);
            let message = if passed {
                format!(
                    "{}={} satisfies {}{}",
                    rule.metric,
                    score,
                    rule.operator.as_str(),
                    rule.threshold
                )
            } else {
                format!(
                    "{}={} violates {}{}",
                    rule.metric,
                    score,
                    rule.operator.as_str(),
                    rule.threshold
                )
            };
            MetricThresholdCheck {
                rule,
                score: Some(score),
                passed,
                message,
            }
        }
        MetricValueLookup::MissingMetric => MetricThresholdCheck {
            message: format!("metric `{}` was not produced", rule.metric),
            rule,
            score: None,
            passed: false,
        },
        MetricValueLookup::MissingDetail => MetricThresholdCheck {
            message: format!("metric detail `{}` was not produced", rule.metric),
            rule,
            score: None,
            passed: false,
        },
    }
}

fn evaluate_baseline_one(
    candidate_metrics: &[MetricOutput],
    baseline_metrics: &[MetricOutput],
    rule: BaselineThresholdRule,
) -> BaselineThresholdCheck {
    let candidate = metric_value(candidate_metrics, &rule.metric);
    let baseline = metric_value(baseline_metrics, &rule.metric);
    match (candidate, baseline) {
        (MetricValueLookup::Found(candidate), MetricValueLookup::Found(baseline)) => {
            let threshold = rule.threshold(baseline);
            let passed = rule.operator.evaluate(candidate, threshold);
            let message = if passed {
                format!(
                    "{}={} satisfies {}{} (baseline={})",
                    rule.metric,
                    candidate,
                    rule.operator.as_str(),
                    threshold,
                    baseline
                )
            } else {
                format!(
                    "{}={} violates {}{} (baseline={})",
                    rule.metric,
                    candidate,
                    rule.operator.as_str(),
                    threshold,
                    baseline
                )
            };
            BaselineThresholdCheck {
                rule,
                candidate_score: Some(candidate),
                baseline_score: Some(baseline),
                threshold: Some(threshold),
                passed,
                message,
            }
        }
        (candidate, baseline) => {
            let missing = match (candidate, baseline) {
                (MetricValueLookup::MissingDetail, _) => "candidate metric detail",
                (MetricValueLookup::MissingMetric, _) => "candidate metric",
                (_, MetricValueLookup::MissingDetail) => "baseline metric detail",
                (_, MetricValueLookup::MissingMetric) => "baseline metric",
                _ => unreachable!("found/found handled above"),
            };
            BaselineThresholdCheck {
                message: format!("{missing} `{}` was not produced", rule.metric),
                rule,
                candidate_score: candidate.found(),
                baseline_score: baseline.found(),
                threshold: None,
                passed: false,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MetricValueLookup {
    Found(f64),
    MissingMetric,
    MissingDetail,
}

impl MetricValueLookup {
    fn found(self) -> Option<f64> {
        match self {
            Self::Found(value) => Some(value),
            Self::MissingMetric | Self::MissingDetail => None,
        }
    }
}

fn metric_value(metrics: &[MetricOutput], selector: &str) -> MetricValueLookup {
    if let Some(metric) = metrics.iter().find(|metric| metric.name == selector) {
        return MetricValueLookup::Found(metric.score);
    }
    let Some((metric_name, detail_name)) = selector.rsplit_once('.') else {
        return MetricValueLookup::MissingMetric;
    };
    let Some(metric) = metrics.iter().find(|metric| metric.name == metric_name) else {
        return MetricValueLookup::MissingMetric;
    };
    metric
        .details
        .get(detail_name)
        .copied()
        .map(MetricValueLookup::Found)
        .unwrap_or(MetricValueLookup::MissingDetail)
}

fn split_rule(input: &str) -> Result<(&str, GateOperator, &str)> {
    for (token, operator) in [
        (">=", GateOperator::GreaterOrEqual),
        ("<=", GateOperator::LessOrEqual),
        (">", GateOperator::GreaterThan),
        ("<", GateOperator::LessThan),
    ] {
        if let Some((metric, value)) = input.split_once(token) {
            let metric = metric.trim();
            if metric.is_empty() {
                return Err(Error::unsupported("threshold rule metric name is empty"));
            }
            return Ok((metric, operator, value.trim()));
        }
    }
    Err(Error::unsupported(format!(
        "invalid threshold rule `{input}`; expected metric>=value, metric>value, metric<=value, or metric<value"
    )))
}

fn parse_baseline_expression(expression: &str, input: &str) -> Result<(f64, f64)> {
    let expression = expression.trim();
    let Some(suffix) = expression.strip_prefix("baseline") else {
        return Err(Error::unsupported(format!(
            "invalid baseline threshold in `{input}`; right-hand side must start with `baseline`"
        )));
    };
    let suffix = suffix.trim();
    let (scale, offset) = if suffix.is_empty() {
        (1.0, 0.0)
    } else if let Some(value) = suffix.strip_prefix('*') {
        (parse_finite(value, input)?, 0.0)
    } else if let Some(value) = suffix.strip_prefix('/') {
        let divisor = parse_finite(value, input)?;
        if divisor == 0.0 {
            return Err(Error::unsupported("baseline divisor must not be zero"));
        }
        (1.0 / divisor, 0.0)
    } else if suffix.starts_with('+') || suffix.starts_with('-') {
        (1.0, parse_finite(suffix, input)?)
    } else {
        return Err(Error::unsupported(format!(
            "invalid baseline expression `{expression}` in `{input}`"
        )));
    };
    if !scale.is_finite() {
        return Err(Error::unsupported("baseline scale must be finite"));
    }
    Ok((scale, offset))
}

fn parse_finite(value: &str, input: &str) -> Result<f64> {
    let value = value.trim().parse::<f64>().map_err(|_| {
        Error::unsupported(format!("invalid baseline threshold value in `{input}`"))
    })?;
    if !value.is_finite() {
        return Err(Error::unsupported(
            "baseline threshold values must be finite",
        ));
    }
    Ok(value)
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

    #[test]
    fn evaluates_metric_detail_selectors() {
        let metric = MetricOutput::new("mae:color", 0.1, "normalized", Direction::LowerIsBetter)
            .with_detail("red_mae", 0.02);
        let rule = MetricThresholdRule::parse("mae:color.red_mae<=0.03").unwrap();
        let evaluation = evaluate_thresholds(&[metric], &[rule]);
        assert!(evaluation.passed);
        assert_eq!(evaluation.checks[0].score, Some(0.02));
    }

    #[test]
    fn baseline_rules_support_offsets_and_scales() {
        let candidate = vec![
            MetricOutput::new("psnr", 39.6, "dB", Direction::HigherIsBetter),
            MetricOutput::new("mae", 0.011, "normalized", Direction::LowerIsBetter),
        ];
        let baseline = vec![
            MetricOutput::new("psnr", 40.0, "dB", Direction::HigherIsBetter),
            MetricOutput::new("mae", 0.01, "normalized", Direction::LowerIsBetter),
        ];
        let rules = [
            BaselineThresholdRule::parse("psnr>=baseline-0.5").unwrap(),
            BaselineThresholdRule::parse("mae<=baseline*1.1").unwrap(),
        ];
        let evaluation = evaluate_baseline_thresholds(&candidate, &baseline, &rules);
        assert!(evaluation.passed);
        assert_eq!(evaluation.checks[0].threshold, Some(39.5));
        assert!((evaluation.checks[1].threshold.unwrap() - 0.011).abs() < 1e-12);
    }

    #[test]
    fn missing_baseline_detail_fails_structurally() {
        let candidate = MetricOutput::new("mae", 0.01, "normalized", Direction::LowerIsBetter)
            .with_detail("red_mae", 0.01);
        let baseline = MetricOutput::new("mae", 0.01, "normalized", Direction::LowerIsBetter);
        let rule = BaselineThresholdRule::parse("mae.red_mae<=baseline").unwrap();
        let evaluation = evaluate_baseline_thresholds(&[candidate], &[baseline], &[rule]);
        assert!(!evaluation.passed);
        assert!(
            evaluation.checks[0]
                .message
                .contains("baseline metric detail")
        );
    }

    #[test]
    fn equal_infinite_psnr_passes_baseline_floor() {
        let candidate = MetricOutput::new("psnr", f64::INFINITY, "dB", Direction::HigherIsBetter);
        let baseline = candidate.clone();
        let rule = BaselineThresholdRule::parse("psnr>=baseline").unwrap();
        let evaluation = evaluate_baseline_thresholds(&[candidate], &[baseline], &[rule]);
        assert!(evaluation.passed);
    }
}
