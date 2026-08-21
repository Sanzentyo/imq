//! Multi-candidate comparison, ranking, and regression-suite orchestration.
//!
//! A single metric row answers whether one candidate resembles a reference.
//! This module handles the next layer needed by codec, renderer, and model
//! evaluations: compare several candidates under one metric policy, preserve
//! per-candidate failures, rank every directional metric, calculate a
//! deterministic consensus rank, and identify the Pareto frontier.

use crate::frame::{Dimensions, FormatSpec, FrameView, Validated};
use crate::gate::{
    BaselineGateEvaluation, BaselineThresholdRule, GateEvaluation, MetricThresholdRule,
    evaluate_baseline_thresholds, evaluate_thresholds,
};
use crate::metrics::{Direction, MetricOutput, MetricSet};
use crate::{Error, Result};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::thread;

/// One named candidate frame in a comparison suite.
#[derive(Debug, Clone)]
pub struct ComparisonCandidate<'a> {
    /// Stable label used in reports and rankings.
    pub label: String,
    /// Validated candidate frame.
    pub frame: FrameView<'a, Validated>,
}

impl<'a> ComparisonCandidate<'a> {
    /// Creates a named candidate.
    pub fn new(label: impl Into<String>, frame: FrameView<'a, Validated>) -> Self {
        Self {
            label: label.into(),
            frame,
        }
    }
}

/// How a suite handles a metric error for one candidate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum CandidateFailurePolicy {
    /// Keep comparing and record the candidate error in the suite report.
    #[default]
    Continue,
    /// Return the first candidate error in input order.
    FailFast,
}

/// Numeric tolerance used when assigning tied ranks and Pareto dominance.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScoreTolerance {
    /// Absolute score difference considered equivalent.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub absolute: f64,
    /// Relative difference, scaled by the larger score magnitude.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub relative: f64,
}

impl ScoreTolerance {
    /// Creates a validated absolute/relative tolerance.
    pub fn new(absolute: f64, relative: f64) -> Result<Self> {
        if !absolute.is_finite() || absolute < 0.0 {
            return Err(Error::unsupported(
                "score absolute tolerance must be finite and non-negative",
            ));
        }
        if !relative.is_finite() || relative < 0.0 {
            return Err(Error::unsupported(
                "score relative tolerance must be finite and non-negative",
            ));
        }
        Ok(Self { absolute, relative })
    }

    fn equivalent(self, left: f64, right: f64) -> bool {
        if left == right || (left.is_nan() && right.is_nan()) {
            return true;
        }
        if !left.is_finite() || !right.is_finite() {
            return false;
        }
        let allowed = self.absolute + self.relative * left.abs().max(right.abs());
        (left - right).abs() <= allowed
    }
}

impl Default for ScoreTolerance {
    fn default() -> Self {
        Self {
            absolute: 1e-12,
            relative: 1e-9,
        }
    }
}

/// Settings for a multi-candidate comparison suite.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComparisonSuiteOptions {
    /// Optional metric that controls the top-level candidate rank.
    ///
    /// When omitted, the top-level rank is the mean of all directional metric
    /// ranks. The name must match a produced [`MetricOutput::name`].
    pub primary_metric: Option<String>,
    /// Optional candidate whose metric scores are subtracted from every result.
    pub baseline_candidate: Option<String>,
    /// Non-negative weights used by the consensus mean rank. Missing metrics
    /// use weight 1 and weight 0 excludes a metric from consensus ranking.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::map"))]
    pub metric_weights: BTreeMap<String, f64>,
    /// CI/regression rules evaluated independently for every candidate.
    pub rules: Vec<MetricThresholdRule>,
    /// Regression rules evaluated against `baseline_candidate` metric values.
    pub baseline_rules: Vec<BaselineThresholdRule>,
    /// Candidate error handling policy.
    pub failure_policy: CandidateFailurePolicy,
    /// Maximum worker threads. `1` is deterministic sequential execution.
    pub max_threads: usize,
    /// Tolerance for ties and Pareto comparisons.
    pub score_tolerance: ScoreTolerance,
}

impl Default for ComparisonSuiteOptions {
    fn default() -> Self {
        Self {
            primary_metric: None,
            baseline_candidate: None,
            metric_weights: BTreeMap::new(),
            rules: Vec::new(),
            baseline_rules: Vec::new(),
            failure_policy: CandidateFailurePolicy::Continue,
            max_threads: thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            score_tolerance: ScoreTolerance::default(),
        }
    }
}

/// One candidate's comparison result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CandidateComparison {
    /// Candidate label.
    pub label: String,
    /// Candidate format.
    pub format: FormatSpec,
    /// Metric rows. Empty when comparison failed.
    pub metrics: Vec<MetricOutput>,
    /// Per-candidate threshold result when rules were configured.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub gate: Option<GateEvaluation>,
    /// Baseline-relative threshold result when those rules were configured.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub baseline_gate: Option<BaselineGateEvaluation>,
    /// Candidate comparison failure, retained when failure policy is `continue`.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub error: Option<String>,
    /// Rank by metric name. Rank 1 is best and ties share a rank.
    pub metric_ranks: BTreeMap<String, usize>,
    /// Mean rank across directional metrics.
    #[cfg_attr(
        feature = "serde",
        serde(
            with = "crate::serde_f64::option",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub mean_rank: Option<f64>,
    /// Top-level rank driven by the primary metric or mean rank.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub overall_rank: Option<usize>,
    /// Score minus the selected baseline candidate's score, by metric.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::map"))]
    pub baseline_deltas: BTreeMap<String, f64>,
    /// Whether no other successful candidate dominates this one across all
    /// shared directional metrics.
    pub pareto_optimal: bool,
}

impl CandidateComparison {
    /// Returns whether metric computation succeeded and all configured rules passed.
    pub fn passed(&self) -> bool {
        self.error.is_none()
            && self
                .gate
                .as_ref()
                .map(|evaluation| evaluation.passed)
                .unwrap_or(true)
            && self
                .baseline_gate
                .as_ref()
                .map(|evaluation| evaluation.passed)
                .unwrap_or(true)
    }
}

/// One entry in a per-metric ranking.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RankedCandidate {
    /// Candidate label.
    pub label: String,
    /// One-based competition rank (`1, 1, 3`).
    pub rank: usize,
    /// Metric score.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub score: f64,
}

/// Ordered candidates for one metric.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetricRanking {
    /// Metric name.
    pub metric: String,
    /// Quality direction used for ordering.
    pub direction: Direction,
    /// Best-to-worst successful candidates.
    pub candidates: Vec<RankedCandidate>,
}

/// Complete multi-candidate comparison report.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComparisonSuiteReport {
    /// Optional reference label.
    pub reference: Option<String>,
    /// Shared reference dimensions.
    pub dimensions: Dimensions,
    /// Reference format.
    pub reference_format: FormatSpec,
    /// Name of the selected primary metric.
    pub primary_metric: Option<String>,
    /// Name of the selected baseline candidate.
    pub baseline_candidate: Option<String>,
    /// Candidate reports in caller-provided order.
    pub candidates: Vec<CandidateComparison>,
    /// Per-metric rankings in metric-output order.
    pub rankings: Vec<MetricRanking>,
    /// Whether every candidate comparison and threshold gate passed.
    pub passed: bool,
}

impl ComparisonSuiteReport {
    /// Returns the candidate with overall rank 1, if any comparison succeeded.
    pub fn winner(&self) -> Option<&CandidateComparison> {
        self.candidates
            .iter()
            .find(|candidate| candidate.overall_rank == Some(1))
    }

    /// Serializes as pretty JSON.
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    pub fn to_json_pretty(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Compares and ranks several candidates against one reference frame.
///
/// Candidate results remain in input order even when `max_threads` enables
/// parallel work. Ranking is deterministic and does not mix metric units: each
/// metric is ranked independently, then ordinal ranks are averaged.
pub fn compare_candidate_suite(
    reference_label: Option<String>,
    reference: &FrameView<'_, Validated>,
    candidates: &[ComparisonCandidate<'_>],
    metrics: &MetricSet,
    options: &ComparisonSuiteOptions,
) -> Result<ComparisonSuiteReport> {
    if candidates.is_empty() {
        return Err(Error::unsupported(
            "comparison suite requires at least one candidate",
        ));
    }
    if options.max_threads == 0 {
        return Err(Error::unsupported(
            "comparison suite max_threads must be at least one",
        ));
    }
    ScoreTolerance::new(
        options.score_tolerance.absolute,
        options.score_tolerance.relative,
    )?;
    ensure_unique_labels(candidates)?;

    let raw_results = compare_all(reference, candidates, metrics, options.max_threads);
    if options.failure_policy == CandidateFailurePolicy::FailFast
        && let Some((candidate, error)) = candidates
            .iter()
            .zip(&raw_results)
            .find_map(|(candidate, result)| result.as_ref().err().map(|error| (candidate, error)))
    {
        return Err(Error::unsupported(format!(
            "candidate `{}` comparison failed: {error}",
            candidate.label
        )));
    }

    let mut reports = candidates
        .iter()
        .zip(raw_results)
        .map(|(candidate, result)| match result {
            Ok(metric_outputs) => CandidateComparison {
                label: candidate.label.clone(),
                format: candidate.frame.format(),
                gate: (!options.rules.is_empty())
                    .then(|| evaluate_thresholds(&metric_outputs, &options.rules)),
                baseline_gate: None,
                metrics: metric_outputs,
                error: None,
                metric_ranks: BTreeMap::new(),
                mean_rank: None,
                overall_rank: None,
                baseline_deltas: BTreeMap::new(),
                pareto_optimal: false,
            },
            Err(error) => CandidateComparison {
                label: candidate.label.clone(),
                format: candidate.frame.format(),
                metrics: Vec::new(),
                gate: None,
                baseline_gate: None,
                error: Some(error.to_string()),
                metric_ranks: BTreeMap::new(),
                mean_rank: None,
                overall_rank: None,
                baseline_deltas: BTreeMap::new(),
                pareto_optimal: false,
            },
        })
        .collect::<Vec<_>>();

    let rankings = build_rankings(&mut reports, options.score_tolerance);
    validate_requested_metric(options.primary_metric.as_deref(), &rankings, "primary")?;
    validate_metric_weights(&options.metric_weights, &rankings)?;
    apply_mean_and_overall_ranks(
        &mut reports,
        options.primary_metric.as_deref(),
        &options.metric_weights,
        options.score_tolerance,
    );
    apply_baseline_deltas(&mut reports, options.baseline_candidate.as_deref())?;
    apply_baseline_gates(
        &mut reports,
        options.baseline_candidate.as_deref(),
        &options.baseline_rules,
    )?;
    apply_pareto_frontier(&mut reports, &rankings, options.score_tolerance);
    let passed = reports.iter().all(CandidateComparison::passed);

    Ok(ComparisonSuiteReport {
        reference: reference_label,
        dimensions: reference.dimensions(),
        reference_format: reference.format(),
        primary_metric: options.primary_metric.clone(),
        baseline_candidate: options.baseline_candidate.clone(),
        candidates: reports,
        rankings,
        passed,
    })
}

fn ensure_unique_labels(candidates: &[ComparisonCandidate<'_>]) -> Result<()> {
    let mut labels = BTreeSet::new();
    for candidate in candidates {
        if candidate.label.trim().is_empty() {
            return Err(Error::unsupported("candidate label cannot be empty"));
        }
        if !labels.insert(candidate.label.as_str()) {
            return Err(Error::unsupported(format!(
                "duplicate comparison candidate label `{}`",
                candidate.label
            )));
        }
    }
    Ok(())
}

fn compare_all(
    reference: &FrameView<'_, Validated>,
    candidates: &[ComparisonCandidate<'_>],
    metrics: &MetricSet,
    max_threads: usize,
) -> Vec<Result<Vec<MetricOutput>>> {
    if max_threads == 1 || candidates.len() == 1 {
        return candidates
            .iter()
            .map(|candidate| metrics.compare(reference, &candidate.frame))
            .collect();
    }

    let mut results = Vec::with_capacity(candidates.len());
    for chunk in candidates.chunks(max_threads.min(candidates.len())) {
        let chunk_results = thread::scope(|scope| {
            let handles = chunk
                .iter()
                .map(|candidate| scope.spawn(move || metrics.compare(reference, &candidate.frame)))
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| {
                    handle.join().unwrap_or_else(|payload| {
                        Err(Error::unsupported(format!(
                            "comparison worker panicked: {}",
                            panic_message(payload)
                        )))
                    })
                })
                .collect::<Vec<_>>()
        });
        results.extend(chunk_results);
    }
    results
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic payload")
        .to_string()
}

fn build_rankings(
    reports: &mut [CandidateComparison],
    tolerance: ScoreTolerance,
) -> Vec<MetricRanking> {
    let mut metric_order = Vec::<(String, Direction)>::new();
    for report in reports.iter().filter(|report| report.error.is_none()) {
        for metric in &report.metrics {
            if metric.direction != Direction::Neutral
                && !metric_order.iter().any(|(name, _)| name == &metric.name)
            {
                metric_order.push((metric.name.clone(), metric.direction));
            }
        }
    }

    metric_order
        .into_iter()
        .filter_map(|(metric_name, direction)| {
            let mut values = reports
                .iter()
                .filter_map(|report| {
                    report
                        .metrics
                        .iter()
                        .find(|metric| metric.name == metric_name)
                        .map(|metric| (report.label.clone(), metric.score))
                })
                .collect::<Vec<_>>();
            if values.is_empty() {
                return None;
            }
            values.sort_by(|left, right| {
                quality_order(left.1, right.1, direction).then_with(|| left.0.cmp(&right.0))
            });
            let mut ranked = Vec::with_capacity(values.len());
            for (index, (label, score)) in values.into_iter().enumerate() {
                let rank = ranked
                    .last()
                    .filter(|previous: &&RankedCandidate| {
                        tolerance.equivalent(previous.score, score)
                    })
                    .map(|previous| previous.rank)
                    .unwrap_or(index + 1);
                if let Some(report) = reports.iter_mut().find(|report| report.label == label) {
                    report.metric_ranks.insert(metric_name.clone(), rank);
                }
                ranked.push(RankedCandidate { label, rank, score });
            }
            Some(MetricRanking {
                metric: metric_name,
                direction,
                candidates: ranked,
            })
        })
        .collect()
}

fn quality_order(left: f64, right: f64, direction: Direction) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => match direction {
            Direction::HigherIsBetter => right.total_cmp(&left),
            Direction::LowerIsBetter => left.total_cmp(&right),
            Direction::Neutral => Ordering::Equal,
        },
    }
}

fn validate_requested_metric(
    requested: Option<&str>,
    rankings: &[MetricRanking],
    kind: &str,
) -> Result<()> {
    if let Some(requested) = requested
        && !rankings.iter().any(|ranking| ranking.metric == requested)
    {
        return Err(Error::unsupported(format!(
            "{kind} metric `{requested}` was not produced as a directional metric"
        )));
    }
    Ok(())
}

fn validate_metric_weights(
    weights: &BTreeMap<String, f64>,
    rankings: &[MetricRanking],
) -> Result<()> {
    for (metric, weight) in weights {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(Error::unsupported(format!(
                "metric weight for `{metric}` must be finite and non-negative"
            )));
        }
        if !rankings.iter().any(|ranking| ranking.metric == *metric) {
            return Err(Error::unsupported(format!(
                "weighted metric `{metric}` was not produced as a directional metric"
            )));
        }
    }
    if !rankings.is_empty()
        && rankings
            .iter()
            .all(|ranking| weights.get(&ranking.metric).copied() == Some(0.0))
    {
        return Err(Error::unsupported(
            "at least one directional metric must have a positive consensus weight",
        ));
    }
    Ok(())
}

fn apply_mean_and_overall_ranks(
    reports: &mut [CandidateComparison],
    primary_metric: Option<&str>,
    metric_weights: &BTreeMap<String, f64>,
    tolerance: ScoreTolerance,
) {
    for report in reports.iter_mut().filter(|report| report.error.is_none()) {
        let (weighted_sum, weight_sum) = report.metric_ranks.iter().fold(
            (0.0, 0.0),
            |(weighted_sum, weight_sum), (metric, rank)| {
                let weight = metric_weights.get(metric).copied().unwrap_or(1.0);
                (weighted_sum + *rank as f64 * weight, weight_sum + weight)
            },
        );
        if weight_sum > 0.0 {
            report.mean_rank = Some(weighted_sum / weight_sum);
        }
    }

    let mut values = reports
        .iter()
        .filter_map(|report| {
            let value = primary_metric
                .and_then(|metric| report.metric_ranks.get(metric).copied())
                .map(|rank| rank as f64)
                .or(report.mean_rank)?;
            Some((report.label.clone(), value))
        })
        .collect::<Vec<_>>();
    values.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut previous: Option<(f64, usize)> = None;
    for (index, (label, value)) in values.into_iter().enumerate() {
        let rank = previous
            .filter(|(previous_value, _)| tolerance.equivalent(*previous_value, value))
            .map(|(_, rank)| rank)
            .unwrap_or(index + 1);
        if let Some(report) = reports.iter_mut().find(|report| report.label == label) {
            report.overall_rank = Some(rank);
        }
        previous = Some((value, rank));
    }
}

fn apply_baseline_deltas(
    reports: &mut [CandidateComparison],
    baseline_label: Option<&str>,
) -> Result<()> {
    let Some(baseline_label) = baseline_label else {
        return Ok(());
    };
    let baseline = reports
        .iter()
        .find(|report| report.label == baseline_label)
        .ok_or_else(|| {
            Error::unsupported(format!(
                "baseline candidate `{baseline_label}` does not exist"
            ))
        })?;
    if let Some(error) = &baseline.error {
        return Err(Error::unsupported(format!(
            "baseline candidate `{baseline_label}` failed comparison: {error}"
        )));
    }
    let baseline_scores = baseline
        .metrics
        .iter()
        .map(|metric| (metric.name.clone(), metric.score))
        .collect::<BTreeMap<_, _>>();
    for report in reports.iter_mut().filter(|report| report.error.is_none()) {
        report.baseline_deltas = report
            .metrics
            .iter()
            .filter_map(|metric| {
                baseline_scores
                    .get(&metric.name)
                    .map(|baseline| (metric.name.clone(), metric.score - baseline))
            })
            .collect();
    }
    Ok(())
}

fn apply_baseline_gates(
    reports: &mut [CandidateComparison],
    baseline_label: Option<&str>,
    rules: &[BaselineThresholdRule],
) -> Result<()> {
    if rules.is_empty() {
        return Ok(());
    }
    let baseline_label = baseline_label.ok_or_else(|| {
        Error::unsupported("baseline-relative rules require a baseline candidate")
    })?;
    let baseline_metrics = reports
        .iter()
        .find(|report| report.label == baseline_label)
        .ok_or_else(|| {
            Error::unsupported(format!(
                "baseline candidate `{baseline_label}` does not exist"
            ))
        })?
        .metrics
        .clone();
    for report in reports.iter_mut().filter(|report| report.error.is_none()) {
        report.baseline_gate = Some(evaluate_baseline_thresholds(
            &report.metrics,
            &baseline_metrics,
            rules,
        ));
    }
    Ok(())
}

fn apply_pareto_frontier(
    reports: &mut [CandidateComparison],
    rankings: &[MetricRanking],
    tolerance: ScoreTolerance,
) {
    let directions = rankings
        .iter()
        .map(|ranking| (ranking.metric.as_str(), ranking.direction))
        .collect::<Vec<_>>();
    let successful = reports
        .iter()
        .enumerate()
        .filter(|(_, report)| report.error.is_none())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    for &candidate_index in &successful {
        let dominated = successful.iter().copied().any(|other_index| {
            other_index != candidate_index
                && dominates(
                    &reports[other_index],
                    &reports[candidate_index],
                    &directions,
                    tolerance,
                )
        });
        reports[candidate_index].pareto_optimal = !dominated;
    }
}

fn dominates(
    left: &CandidateComparison,
    right: &CandidateComparison,
    directions: &[(&str, Direction)],
    tolerance: ScoreTolerance,
) -> bool {
    let mut strictly_better = false;
    for &(name, direction) in directions {
        let Some(left_score) = metric_score(left, name) else {
            return false;
        };
        let Some(right_score) = metric_score(right, name) else {
            return false;
        };
        match compare_quality(left_score, right_score, direction, tolerance) {
            QualityComparison::Better => strictly_better = true,
            QualityComparison::Equivalent => {}
            QualityComparison::Worse => return false,
        }
    }
    strictly_better
}

fn metric_score(report: &CandidateComparison, name: &str) -> Option<f64> {
    report
        .metrics
        .iter()
        .find(|metric| metric.name == name)
        .map(|metric| metric.score)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QualityComparison {
    Better,
    Equivalent,
    Worse,
}

fn compare_quality(
    left: f64,
    right: f64,
    direction: Direction,
    tolerance: ScoreTolerance,
) -> QualityComparison {
    if tolerance.equivalent(left, right) {
        return QualityComparison::Equivalent;
    }
    if left.is_nan() {
        return QualityComparison::Worse;
    }
    if right.is_nan() {
        return QualityComparison::Better;
    }
    let left_is_better = match direction {
        Direction::HigherIsBetter => left > right,
        Direction::LowerIsBetter => left < right,
        Direction::Neutral => return QualityComparison::Equivalent,
    };
    if left_is_better {
        QualityComparison::Better
    } else {
        QualityComparison::Worse
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FrameOwned, PixelFormat};

    fn luma(values: &[u8], width: u32, height: u32) -> FrameOwned {
        FrameOwned::packed_tight(values.to_vec(), width, height, PixelFormat::Luma8).unwrap()
    }

    #[test]
    fn ranks_candidates_and_marks_pareto_frontier() {
        let reference = luma(&[0, 64, 128, 255], 2, 2);
        let exact = luma(&[0, 64, 128, 255], 2, 2);
        let near = luma(&[0, 65, 128, 254], 2, 2);
        let far = luma(&[20, 80, 150, 220], 2, 2);
        let candidates = vec![
            ComparisonCandidate::new("far", far.as_view()),
            ComparisonCandidate::new("exact", exact.as_view()),
            ComparisonCandidate::new("near", near.as_view()),
        ];
        let metrics = MetricSet::from_csv("psnr,mse,mae").unwrap();
        let options = ComparisonSuiteOptions {
            primary_metric: Some("psnr".to_string()),
            baseline_candidate: Some("near".to_string()),
            max_threads: 3,
            ..ComparisonSuiteOptions::default()
        };

        let report = compare_candidate_suite(
            Some("reference".to_string()),
            &reference.as_view(),
            &candidates,
            &metrics,
            &options,
        )
        .unwrap();

        assert_eq!(
            report.winner().map(|candidate| candidate.label.as_str()),
            Some("exact")
        );
        assert_eq!(report.candidates[0].overall_rank, Some(3));
        assert_eq!(report.candidates[1].overall_rank, Some(1));
        assert_eq!(report.candidates[2].overall_rank, Some(2));
        assert!(report.candidates[1].pareto_optimal);
        assert!(!report.candidates[0].pareto_optimal);
        assert_eq!(report.candidates[2].baseline_deltas["mse"], 0.0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn exact_suite_json_roundtrips_infinite_scores_and_nan_deltas() {
        let reference = luma(&[0, 64, 128, 255], 2, 2);
        let exact_a = luma(&[0, 64, 128, 255], 2, 2);
        let exact_b = luma(&[0, 64, 128, 255], 2, 2);
        let candidates = vec![
            ComparisonCandidate::new("baseline", exact_a.as_view()),
            ComparisonCandidate::new("candidate", exact_b.as_view()),
        ];
        let report = compare_candidate_suite(
            Some("reference".into()),
            &reference.as_view(),
            &candidates,
            &MetricSet::from_csv("psnr,mse").unwrap(),
            &ComparisonSuiteOptions {
                baseline_candidate: Some("baseline".into()),
                ..ComparisonSuiteOptions::default()
            },
        )
        .unwrap();

        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"Infinity\""));
        assert!(json.contains("\"NaN\""));
        let decoded: ComparisonSuiteReport = serde_json::from_str(&json).unwrap();
        assert!(decoded.rankings[0].candidates[0].score.is_infinite());
        assert!(decoded.candidates[1].baseline_deltas["psnr"].is_nan());
    }

    #[test]
    fn continue_policy_keeps_dimension_failure() {
        let reference = luma(&[0, 1, 2, 3], 2, 2);
        let valid = luma(&[0, 1, 2, 3], 2, 2);
        let invalid = luma(&[0, 1], 2, 1);
        let candidates = vec![
            ComparisonCandidate::new("valid", valid.as_view()),
            ComparisonCandidate::new("invalid", invalid.as_view()),
        ];
        let report = compare_candidate_suite(
            None,
            &reference.as_view(),
            &candidates,
            &MetricSet::from_csv("mse").unwrap(),
            &ComparisonSuiteOptions::default(),
        )
        .unwrap();

        assert!(!report.passed);
        assert!(report.candidates[0].error.is_none());
        assert!(report.candidates[1].error.is_some());
    }

    #[test]
    fn gates_each_candidate() {
        let reference = luma(&[0, 0, 0, 0], 2, 2);
        let exact = luma(&[0, 0, 0, 0], 2, 2);
        let changed = luma(&[255, 255, 255, 255], 2, 2);
        let candidates = vec![
            ComparisonCandidate::new("exact", exact.as_view()),
            ComparisonCandidate::new("changed", changed.as_view()),
        ];
        let options = ComparisonSuiteOptions {
            rules: vec![MetricThresholdRule::parse("mse<=0").unwrap()],
            ..ComparisonSuiteOptions::default()
        };
        let report = compare_candidate_suite(
            None,
            &reference.as_view(),
            &candidates,
            &MetricSet::from_csv("mse").unwrap(),
            &options,
        )
        .unwrap();

        assert!(report.candidates[0].passed());
        assert!(!report.candidates[1].passed());
        assert!(!report.passed);
    }

    #[test]
    fn gates_relative_to_baseline_candidate() {
        let reference = luma(&[0, 0, 0, 0], 2, 2);
        let baseline = luma(&[10, 10, 10, 10], 2, 2);
        let better = luma(&[5, 5, 5, 5], 2, 2);
        let worse = luma(&[20, 20, 20, 20], 2, 2);
        let candidates = vec![
            ComparisonCandidate::new("baseline", baseline.as_view()),
            ComparisonCandidate::new("better", better.as_view()),
            ComparisonCandidate::new("worse", worse.as_view()),
        ];
        let options = ComparisonSuiteOptions {
            baseline_candidate: Some("baseline".to_string()),
            baseline_rules: vec![BaselineThresholdRule::parse("mse<=baseline*1.1").unwrap()],
            ..ComparisonSuiteOptions::default()
        };
        let report = compare_candidate_suite(
            None,
            &reference.as_view(),
            &candidates,
            &MetricSet::from_csv("mse").unwrap(),
            &options,
        )
        .unwrap();

        assert!(report.candidates[0].passed());
        assert!(report.candidates[1].passed());
        assert!(!report.candidates[2].passed());
    }

    #[test]
    fn rejects_duplicate_labels_and_zero_threads() {
        let frame = luma(&[0], 1, 1);
        let duplicate = vec![
            ComparisonCandidate::new("same", frame.as_view()),
            ComparisonCandidate::new("same", frame.as_view()),
        ];
        assert!(
            compare_candidate_suite(
                None,
                &frame.as_view(),
                &duplicate,
                &MetricSet::defaults(),
                &ComparisonSuiteOptions::default(),
            )
            .is_err()
        );

        let options = ComparisonSuiteOptions {
            max_threads: 0,
            ..ComparisonSuiteOptions::default()
        };
        assert!(
            compare_candidate_suite(
                None,
                &frame.as_view(),
                &[ComparisonCandidate::new("candidate", frame.as_view())],
                &MetricSet::defaults(),
                &options,
            )
            .is_err()
        );

        let options = ComparisonSuiteOptions {
            metric_weights: BTreeMap::from([("mse".to_string(), -1.0)]),
            ..ComparisonSuiteOptions::default()
        };
        assert!(
            compare_candidate_suite(
                None,
                &frame.as_view(),
                &[ComparisonCandidate::new("candidate", frame.as_view())],
                &MetricSet::from_csv("mse").unwrap(),
                &options,
            )
            .is_err()
        );
    }
}
