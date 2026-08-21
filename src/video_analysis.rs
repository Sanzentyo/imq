//! Video aggregation and timestamp-aware frame-pairing helpers.
//!
//! Existing video comparison can report per-frame metrics. This module adds a
//! reusable aggregation layer and a timestamp pairer for variable-frame-rate or
//! dropped-frame workflows.

use crate::metrics::Direction;
use crate::report::{FrameReport, VideoReport};

/// Aggregation settings for per-frame video metrics.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VideoAggregationPolicy {
    /// Percentiles to compute, expressed in the inclusive range 0..=100.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::vec"))]
    pub percentiles: Vec<f64>,
    /// Whether to record the worst frame according to the metric direction.
    pub include_worst_frame: bool,
}

impl Default for VideoAggregationPolicy {
    fn default() -> Self {
        Self {
            percentiles: vec![1.0, 5.0, 50.0, 95.0, 99.0],
            include_worst_frame: true,
        }
    }
}

/// One percentile value for a metric series.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PercentileValue {
    /// Percentile in the inclusive range 0..=100.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub percentile: f64,
    /// Interpolated score at the percentile.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub score: f64,
}

/// Aggregate statistics for one metric across a video comparison.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetricSeriesAggregate {
    /// Metric name.
    pub name: String,
    /// Metric unit.
    pub unit: String,
    /// Metric direction.
    pub direction: Direction,
    /// Number of finite frame-level samples used for arithmetic statistics.
    pub samples: usize,
    /// Number of frame-level metric values before non-finite values are filtered.
    pub total_samples: usize,
    /// Number of `NaN` frame-level values.
    pub nan_samples: usize,
    /// Number of positive-infinity frame-level values.
    pub positive_infinite_samples: usize,
    /// Number of negative-infinity frame-level values.
    pub negative_infinite_samples: usize,
    /// Arithmetic mean score.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub mean: f64,
    /// Minimum score.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub min: f64,
    /// Maximum score.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub max: f64,
    /// Median score.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub median: f64,
    /// Population standard deviation of finite frame-level scores.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub standard_deviation: f64,
    /// Mean weighted by the presentation-time interval represented by each sample.
    ///
    /// This is `None` when fewer than two usable presentation timestamps are
    /// available. It is useful for VFR material where a simple frame mean can
    /// overweight short-lived high-frame-rate sections.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub time_weighted_mean: Option<f64>,
    /// Requested percentile scores.
    pub percentiles: Vec<PercentileValue>,
    /// Worst frame index according to metric direction.
    pub worst_frame_index: Option<u64>,
    /// Worst frame timestamp when available.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub worst_pts_seconds: Option<f64>,
    /// Worst score according to metric direction.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub worst_score: Option<f64>,
}

/// Aggregates all metrics present in a [`VideoReport`].
pub fn aggregate_video_report(
    report: &VideoReport,
    policy: &VideoAggregationPolicy,
) -> Vec<MetricSeriesAggregate> {
    let mut names = Vec::<String>::new();
    for frame in &report.frames {
        for metric in &frame.metrics {
            if !names.iter().any(|name| name == &metric.name) {
                names.push(metric.name.clone());
            }
        }
    }
    names
        .into_iter()
        .filter_map(|name| aggregate_metric_series(&report.frames, &name, policy))
        .collect()
}

fn aggregate_metric_series(
    frames: &[FrameReport],
    name: &str,
    policy: &VideoAggregationPolicy,
) -> Option<MetricSeriesAggregate> {
    let mut samples = Vec::<(u64, Option<f64>, f64, String, Direction)>::new();
    let mut total_samples = 0usize;
    let mut nan_samples = 0usize;
    let mut positive_infinite_samples = 0usize;
    let mut negative_infinite_samples = 0usize;
    let mut observed = Vec::<(u64, Option<f64>, f64, String, Direction)>::new();
    for frame in frames {
        if let Some(metric) = frame.metrics.iter().find(|metric| metric.name == name) {
            total_samples += 1;
            observed.push((
                frame.frame_index,
                frame.pts_seconds,
                metric.score,
                metric.unit.clone(),
                metric.direction,
            ));
            if metric.score.is_finite() {
                samples.push((
                    frame.frame_index,
                    frame.pts_seconds,
                    metric.score,
                    metric.unit.clone(),
                    metric.direction,
                ));
            } else if metric.score.is_nan() {
                nan_samples += 1;
            } else if metric.score.is_sign_positive() {
                positive_infinite_samples += 1;
            } else {
                negative_infinite_samples += 1;
            }
        }
    }
    if samples.is_empty() {
        let first = observed.first()?;
        let mut ordered_non_nan = observed
            .iter()
            .filter(|sample| !sample.2.is_nan())
            .map(|sample| sample.2)
            .collect::<Vec<_>>();
        ordered_non_nan.sort_by(f64::total_cmp);
        let mean = match (
            positive_infinite_samples,
            negative_infinite_samples,
            nan_samples,
        ) {
            (positive, 0, 0) if positive > 0 => f64::INFINITY,
            (0, negative, 0) if negative > 0 => f64::NEG_INFINITY,
            _ => f64::NAN,
        };
        let min = ordered_non_nan.first().copied().unwrap_or(f64::NAN);
        let max = ordered_non_nan.last().copied().unwrap_or(f64::NAN);
        let median = percentile_sorted(&ordered_non_nan, 50.0);
        let percentiles = policy
            .percentiles
            .iter()
            .copied()
            .map(|percentile| PercentileValue {
                percentile,
                score: percentile_sorted(&ordered_non_nan, percentile),
            })
            .collect();
        let worst = if policy.include_worst_frame {
            observed
                .iter()
                .filter(|sample| !sample.2.is_nan())
                .min_by(|a, b| compare_worst(a.2, b.2, first.4))
        } else {
            None
        };
        return Some(MetricSeriesAggregate {
            name: name.to_string(),
            unit: first.3.clone(),
            direction: first.4,
            samples: 0,
            total_samples,
            nan_samples,
            positive_infinite_samples,
            negative_infinite_samples,
            mean,
            min,
            max,
            median,
            standard_deviation: f64::NAN,
            time_weighted_mean: None,
            percentiles,
            worst_frame_index: worst.map(|sample| sample.0),
            worst_pts_seconds: worst.and_then(|sample| sample.1),
            worst_score: worst.map(|sample| sample.2),
        });
    }

    let unit = samples[0].3.clone();
    let direction = samples[0].4;
    let mut sorted_scores = samples.iter().map(|sample| sample.2).collect::<Vec<_>>();
    sorted_scores.sort_by(|a, b| a.total_cmp(b));
    let mean = sorted_scores.iter().sum::<f64>() / sorted_scores.len() as f64;
    let variance = sorted_scores
        .iter()
        .map(|score| {
            let delta = score - mean;
            delta * delta
        })
        .sum::<f64>()
        / sorted_scores.len() as f64;
    let min = *sorted_scores.first().expect("non-empty scores");
    let max = *sorted_scores.last().expect("non-empty scores");
    let median = percentile_sorted(&sorted_scores, 50.0);
    let percentiles = policy
        .percentiles
        .iter()
        .copied()
        .map(|percentile| PercentileValue {
            percentile,
            score: percentile_sorted(&sorted_scores, percentile),
        })
        .collect::<Vec<_>>();

    let worst = if policy.include_worst_frame {
        samples
            .iter()
            .min_by(|a, b| compare_worst(a.2, b.2, direction))
    } else {
        None
    };

    Some(MetricSeriesAggregate {
        name: name.to_string(),
        unit,
        direction,
        samples: sorted_scores.len(),
        total_samples,
        nan_samples,
        positive_infinite_samples,
        negative_infinite_samples,
        mean,
        min,
        max,
        median,
        standard_deviation: variance.sqrt(),
        time_weighted_mean: time_weighted_mean(&samples),
        percentiles,
        worst_frame_index: worst.map(|sample| sample.0),
        worst_pts_seconds: worst.and_then(|sample| sample.1),
        worst_score: worst.map(|sample| sample.2),
    })
}

fn time_weighted_mean(samples: &[(u64, Option<f64>, f64, String, Direction)]) -> Option<f64> {
    let mut timestamped = samples
        .iter()
        .filter_map(|sample| {
            sample
                .1
                .filter(|pts| pts.is_finite())
                .map(|pts| (pts, sample.2))
        })
        .collect::<Vec<_>>();
    timestamped.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut grouped = Vec::<(f64, f64, usize)>::new();
    for (pts, score) in timestamped {
        if let Some(group) = grouped.last_mut().filter(|group| group.0 == pts) {
            group.1 += score;
            group.2 += 1;
        } else {
            grouped.push((pts, score, 1));
        }
    }
    if grouped.len() < 2 {
        return None;
    }

    let mut intervals = grouped
        .windows(2)
        .filter_map(|window| {
            let duration = window[1].0 - window[0].0;
            (duration.is_finite() && duration > 0.0).then_some(duration)
        })
        .collect::<Vec<_>>();
    if intervals.is_empty() {
        return None;
    }
    intervals.sort_by(f64::total_cmp);
    let terminal_duration = percentile_sorted(&intervals, 50.0);
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;
    for (index, sample) in grouped.iter().enumerate() {
        let duration = grouped
            .get(index + 1)
            .map(|next| next.0 - sample.0)
            .filter(|duration| duration.is_finite() && *duration > 0.0)
            .unwrap_or(terminal_duration);
        weighted_sum += (sample.1 / sample.2 as f64) * duration;
        total_weight += duration;
    }
    (total_weight > 0.0).then_some(weighted_sum / total_weight)
}

fn compare_worst(a: f64, b: f64, direction: Direction) -> std::cmp::Ordering {
    match direction {
        Direction::HigherIsBetter => a.total_cmp(&b),
        Direction::LowerIsBetter => b.total_cmp(&a),
        Direction::Neutral => a.abs().total_cmp(&b.abs()).reverse(),
    }
}

fn percentile_sorted(sorted: &[f64], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let p = percentile.clamp(0.0, 100.0) / 100.0;
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let t = pos - lo as f64;
        sorted[lo] * (1.0 - t) + sorted[hi] * t
    }
}

/// Lightweight timestamped frame descriptor used by the pairer.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimestampedFrame {
    /// Decode-order or caller-defined frame index.
    pub index: u64,
    /// Presentation timestamp in seconds.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub pts_seconds: f64,
}

/// Affine transform applied to distorted timestamps before pairing.
///
/// The aligned timeline is `distorted * scale + offset_seconds`. A scale other
/// than one accounts for clock drift as well as a constant start-time offset.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimestampTransform {
    /// Multiplicative clock-rate correction.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub scale: f64,
    /// Constant offset added after scaling, in seconds.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub offset_seconds: f64,
}

impl TimestampTransform {
    /// Identity transform.
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        offset_seconds: 0.0,
    };

    /// Applies the transform to one distorted timestamp.
    pub fn apply(self, distorted_pts_seconds: f64) -> f64 {
        distorted_pts_seconds.mul_add(self.scale, self.offset_seconds)
    }
}

impl Default for TimestampTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Options for timestamp-aware frame pairing.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimestampPairingOptions {
    /// Maximum allowed absolute timestamp difference in seconds.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub max_delta_seconds: f64,
    /// Whether one distorted frame may be paired with multiple reference frames.
    pub allow_reuse_distorted: bool,
}

impl Default for TimestampPairingOptions {
    fn default() -> Self {
        Self {
            max_delta_seconds: 0.5 / 30.0,
            allow_reuse_distorted: false,
        }
    }
}

/// One timestamp-based frame pair.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimestampFramePair {
    /// Reference frame index.
    pub reference_index: u64,
    /// Distorted frame index.
    pub distorted_index: u64,
    /// Reference timestamp in seconds.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub reference_pts_seconds: f64,
    /// Distorted timestamp in seconds.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub distorted_pts_seconds: f64,
    /// Absolute timestamp delta in seconds after applying the alignment transform.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub delta_seconds: f64,
}

/// Diagnostics for one timestamp-alignment pass.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimestampAlignmentReport {
    /// Transform applied to distorted timestamps.
    pub transform: TimestampTransform,
    /// Accepted frame pairs in reference presentation order.
    pub pairs: Vec<TimestampFramePair>,
    /// Reference frame indices for which no match was accepted.
    pub unmatched_reference_indices: Vec<u64>,
    /// Distorted frame indices that were not used by any pair.
    pub unmatched_distorted_indices: Vec<u64>,
    /// Mean signed residual (`reference - aligned distorted`) in seconds.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub mean_signed_delta_seconds: Option<f64>,
    /// Mean absolute residual in seconds.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub mean_absolute_delta_seconds: Option<f64>,
    /// Root-mean-square residual in seconds.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub rms_delta_seconds: Option<f64>,
    /// Largest absolute residual in seconds.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub max_delta_seconds: Option<f64>,
}

/// Pairs frames by nearest presentation timestamp.
pub fn pair_frames_by_timestamp(
    reference: &[TimestampedFrame],
    distorted: &[TimestampedFrame],
    options: TimestampPairingOptions,
) -> Vec<TimestampFramePair> {
    pair_frames_by_timestamp_with_transform(
        reference,
        distorted,
        options,
        TimestampTransform::IDENTITY,
    )
}

/// Pairs frames after applying an affine correction to distorted timestamps.
pub fn pair_frames_by_timestamp_with_transform(
    reference: &[TimestampedFrame],
    distorted: &[TimestampedFrame],
    options: TimestampPairingOptions,
    transform: TimestampTransform,
) -> Vec<TimestampFramePair> {
    align_frames_by_timestamp(reference, distorted, options, transform).pairs
}

/// Pairs frames and returns unmatched-frame and residual diagnostics.
pub fn align_frames_by_timestamp(
    reference: &[TimestampedFrame],
    distorted: &[TimestampedFrame],
    options: TimestampPairingOptions,
    transform: TimestampTransform,
) -> TimestampAlignmentReport {
    let max_delta = if options.max_delta_seconds.is_finite() {
        options.max_delta_seconds.max(0.0)
    } else {
        0.0
    };
    let mut references = reference
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, frame)| frame.pts_seconds.is_finite())
        .collect::<Vec<_>>();
    let mut distorteds = distorted
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(position, frame)| {
            let aligned = transform.apply(frame.pts_seconds);
            (frame.pts_seconds.is_finite() && aligned.is_finite())
                .then_some((position, frame, aligned))
        })
        .collect::<Vec<_>>();
    references.sort_by(|a, b| {
        a.1.pts_seconds
            .total_cmp(&b.1.pts_seconds)
            .then_with(|| a.1.index.cmp(&b.1.index))
    });
    distorteds.sort_by(|a, b| a.2.total_cmp(&b.2).then_with(|| a.1.index.cmp(&b.1.index)));

    let mut matched_reference = vec![false; reference.len()];
    let mut matched_distorted = vec![false; distorted.len()];
    let mut pairs = Vec::new();

    if options.allow_reuse_distorted {
        for (reference_position, reference_frame) in references.iter().copied() {
            let insertion =
                distorteds.partition_point(|candidate| candidate.2 < reference_frame.pts_seconds);
            let best = [
                insertion.checked_sub(1),
                (insertion < distorteds.len()).then_some(insertion),
            ]
            .into_iter()
            .flatten()
            .min_by(|a, b| {
                let delta_a = (reference_frame.pts_seconds - distorteds[*a].2).abs();
                let delta_b = (reference_frame.pts_seconds - distorteds[*b].2).abs();
                delta_a
                    .total_cmp(&delta_b)
                    .then_with(|| distorteds[*a].1.index.cmp(&distorteds[*b].1.index))
            });
            if let Some(best) = best {
                let (distorted_position, distorted_frame, aligned_pts) = distorteds[best];
                let delta = (reference_frame.pts_seconds - aligned_pts).abs();
                if delta <= max_delta {
                    matched_reference[reference_position] = true;
                    matched_distorted[distorted_position] = true;
                    pairs.push(timestamp_pair(reference_frame, distorted_frame, delta));
                }
            }
        }
    } else {
        let mut distorted_cursor = 0usize;
        for (reference_position, reference_frame) in references.iter().copied() {
            while distorted_cursor < distorteds.len()
                && distorteds[distorted_cursor].2 < reference_frame.pts_seconds - max_delta
            {
                distorted_cursor += 1;
            }
            if distorted_cursor >= distorteds.len() {
                continue;
            }

            let mut best = distorted_cursor;
            if let Some(next) = distorteds.get(distorted_cursor + 1) {
                let current_delta =
                    (reference_frame.pts_seconds - distorteds[distorted_cursor].2).abs();
                let next_delta = (reference_frame.pts_seconds - next.2).abs();
                if next_delta < current_delta {
                    best = distorted_cursor + 1;
                }
            }
            let (distorted_position, distorted_frame, aligned_pts) = distorteds[best];
            let delta = (reference_frame.pts_seconds - aligned_pts).abs();
            if delta <= max_delta {
                matched_reference[reference_position] = true;
                matched_distorted[distorted_position] = true;
                pairs.push(timestamp_pair(reference_frame, distorted_frame, delta));
                distorted_cursor = best + 1;
            }
        }
    }

    let residuals = pairs
        .iter()
        .map(|pair| pair.reference_pts_seconds - transform.apply(pair.distorted_pts_seconds))
        .collect::<Vec<_>>();
    let residual_count = residuals.len() as f64;
    let mean_signed_delta_seconds =
        (!residuals.is_empty()).then(|| residuals.iter().sum::<f64>() / residual_count);
    let mean_absolute_delta_seconds = (!residuals.is_empty())
        .then(|| residuals.iter().map(|delta| delta.abs()).sum::<f64>() / residual_count);
    let rms_delta_seconds = (!residuals.is_empty()).then(|| {
        (residuals.iter().map(|delta| delta * delta).sum::<f64>() / residual_count).sqrt()
    });
    let max_delta_seconds = residuals
        .iter()
        .map(|delta| delta.abs())
        .max_by(f64::total_cmp);

    TimestampAlignmentReport {
        transform,
        pairs,
        unmatched_reference_indices: reference
            .iter()
            .enumerate()
            .filter_map(|(position, frame)| (!matched_reference[position]).then_some(frame.index))
            .collect(),
        unmatched_distorted_indices: distorted
            .iter()
            .enumerate()
            .filter_map(|(position, frame)| (!matched_distorted[position]).then_some(frame.index))
            .collect(),
        mean_signed_delta_seconds,
        mean_absolute_delta_seconds,
        rms_delta_seconds,
        max_delta_seconds,
    }
}

fn timestamp_pair(
    reference: TimestampedFrame,
    distorted: TimestampedFrame,
    delta_seconds: f64,
) -> TimestampFramePair {
    TimestampFramePair {
        reference_index: reference.index,
        distorted_index: distorted.index,
        reference_pts_seconds: reference.pts_seconds,
        distorted_pts_seconds: distorted.pts_seconds,
        delta_seconds,
    }
}

/// Estimates an affine distorted-to-reference timestamp transform.
///
/// The estimator aligns presentation-time quantiles before fitting a least
/// squares line. Quantile alignment is resilient to unequal frame counts caused
/// by dropped or duplicated frames and avoids assuming matching decode indices.
pub fn estimate_timestamp_transform(
    reference: &[TimestampedFrame],
    distorted: &[TimestampedFrame],
) -> Option<TimestampTransform> {
    let mut reference_pts = reference
        .iter()
        .map(|frame| frame.pts_seconds)
        .filter(|pts| pts.is_finite())
        .collect::<Vec<_>>();
    let mut distorted_pts = distorted
        .iter()
        .map(|frame| frame.pts_seconds)
        .filter(|pts| pts.is_finite())
        .collect::<Vec<_>>();
    reference_pts.sort_by(f64::total_cmp);
    distorted_pts.sort_by(f64::total_cmp);
    if reference_pts.is_empty() || distorted_pts.is_empty() {
        return None;
    }
    if reference_pts.len() == 1 || distorted_pts.len() == 1 {
        return Some(TimestampTransform {
            scale: 1.0,
            offset_seconds: reference_pts[0] - distorted_pts[0],
        });
    }

    let sample_count = reference_pts.len().min(distorted_pts.len()).min(129);
    let mut samples = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let quantile = index as f64 / (sample_count - 1) as f64;
        samples.push((
            percentile_sorted(&distorted_pts, quantile * 100.0),
            percentile_sorted(&reference_pts, quantile * 100.0),
        ));
    }
    let mean_x = samples.iter().map(|sample| sample.0).sum::<f64>() / sample_count as f64;
    let mean_y = samples.iter().map(|sample| sample.1).sum::<f64>() / sample_count as f64;
    let covariance = samples
        .iter()
        .map(|sample| (sample.0 - mean_x) * (sample.1 - mean_y))
        .sum::<f64>();
    let variance_x = samples
        .iter()
        .map(|sample| {
            let delta = sample.0 - mean_x;
            delta * delta
        })
        .sum::<f64>();
    let scale = if variance_x > f64::EPSILON {
        covariance / variance_x
    } else {
        1.0
    };
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    let offset_seconds = mean_y - scale * mean_x;
    Some(TimestampTransform {
        scale,
        offset_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{Direction, MetricOutput};

    #[test]
    fn aggregates_worst_frame_for_higher_is_better() {
        let report = VideoReport {
            reference: "ref".into(),
            distorted: "dist".into(),
            dimensions: crate::frame::Dimensions::new(1, 1).unwrap(),
            compared_frames: 2,
            frames: vec![
                FrameReport {
                    frame_index: 0,
                    pts_seconds: Some(0.0),
                    metrics: vec![MetricOutput::new(
                        "psnr",
                        40.0,
                        "dB",
                        Direction::HigherIsBetter,
                    )],
                },
                FrameReport {
                    frame_index: 1,
                    pts_seconds: Some(1.0),
                    metrics: vec![MetricOutput::new(
                        "psnr",
                        30.0,
                        "dB",
                        Direction::HigherIsBetter,
                    )],
                },
            ],
            mean_metrics: Vec::new(),
        };
        let aggregate = aggregate_video_report(&report, &VideoAggregationPolicy::default());
        assert_eq!(aggregate[0].worst_frame_index, Some(1));
    }

    #[test]
    fn pairs_nearest_timestamps_without_reuse() {
        let reference = [TimestampedFrame {
            index: 0,
            pts_seconds: 0.0,
        }];
        let distorted = [TimestampedFrame {
            index: 10,
            pts_seconds: 0.01,
        }];
        let pairs =
            pair_frames_by_timestamp(&reference, &distorted, TimestampPairingOptions::default());
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].distorted_index, 10);
    }

    #[test]
    fn aggregates_non_finite_counts_and_vfr_weighted_mean() {
        let metric =
            |score| MetricOutput::new("mse", score, "normalized", Direction::LowerIsBetter);
        let report = VideoReport {
            reference: "ref".into(),
            distorted: "dist".into(),
            dimensions: crate::frame::Dimensions::new(1, 1).unwrap(),
            compared_frames: 4,
            frames: vec![
                FrameReport {
                    frame_index: 0,
                    pts_seconds: Some(0.0),
                    metrics: vec![metric(10.0)],
                },
                FrameReport {
                    frame_index: 1,
                    pts_seconds: Some(1.0),
                    metrics: vec![metric(20.0)],
                },
                FrameReport {
                    frame_index: 2,
                    pts_seconds: Some(3.0),
                    metrics: vec![metric(30.0)],
                },
                FrameReport {
                    frame_index: 3,
                    pts_seconds: Some(4.0),
                    metrics: vec![metric(f64::NAN)],
                },
            ],
            mean_metrics: Vec::new(),
        };

        let aggregate = aggregate_video_report(&report, &VideoAggregationPolicy::default());
        assert_eq!(aggregate[0].samples, 3);
        assert_eq!(aggregate[0].total_samples, 4);
        assert_eq!(aggregate[0].nan_samples, 1);
        assert_eq!(aggregate[0].positive_infinite_samples, 0);
        assert!((aggregate[0].time_weighted_mean.unwrap() - 21.111_111).abs() < 1e-5);
    }

    #[test]
    fn aggregates_all_perfect_psnr_without_dropping_the_series() {
        let report = VideoReport {
            reference: "ref".into(),
            distorted: "dist".into(),
            dimensions: crate::frame::Dimensions::new(1, 1).unwrap(),
            compared_frames: 2,
            frames: vec![
                FrameReport {
                    frame_index: 0,
                    pts_seconds: Some(0.0),
                    metrics: vec![MetricOutput::new(
                        "psnr",
                        f64::INFINITY,
                        "dB",
                        Direction::HigherIsBetter,
                    )],
                },
                FrameReport {
                    frame_index: 1,
                    pts_seconds: Some(1.0),
                    metrics: vec![MetricOutput::new(
                        "psnr",
                        f64::INFINITY,
                        "dB",
                        Direction::HigherIsBetter,
                    )],
                },
            ],
            mean_metrics: Vec::new(),
        };

        let aggregate = aggregate_video_report(&report, &VideoAggregationPolicy::default());
        assert_eq!(aggregate.len(), 1);
        assert_eq!(aggregate[0].samples, 0);
        assert_eq!(aggregate[0].positive_infinite_samples, 2);
        assert_eq!(aggregate[0].mean, f64::INFINITY);
        assert_eq!(aggregate[0].worst_frame_index, Some(0));
        #[cfg(feature = "serde")]
        {
            let json = serde_json::to_string(&aggregate[0]).unwrap();
            assert!(json.contains("\"mean\":\"Infinity\""));
            let decoded: MetricSeriesAggregate = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.mean, f64::INFINITY);
        }
    }

    #[test]
    fn time_weighted_mean_shares_duration_between_duplicate_pts() {
        let samples = vec![
            (0, Some(0.0), 0.0, "unit".to_string(), Direction::Neutral),
            (1, Some(0.0), 100.0, "unit".to_string(), Direction::Neutral),
            (2, Some(1.0), 0.0, "unit".to_string(), Direction::Neutral),
        ];
        assert_eq!(time_weighted_mean(&samples), Some(25.0));
    }

    #[test]
    fn affine_timestamp_transform_recovers_offset_and_drift() {
        let reference = [0.0, 2.0, 4.0]
            .into_iter()
            .enumerate()
            .map(|(index, pts_seconds)| TimestampedFrame {
                index: index as u64,
                pts_seconds,
            })
            .collect::<Vec<_>>();
        let distorted = [1.0, 2.0, 3.0]
            .into_iter()
            .enumerate()
            .map(|(index, pts_seconds)| TimestampedFrame {
                index: index as u64,
                pts_seconds,
            })
            .collect::<Vec<_>>();

        let transform = estimate_timestamp_transform(&reference, &distorted).unwrap();
        assert!((transform.scale - 2.0).abs() < 1e-12);
        assert!((transform.offset_seconds + 2.0).abs() < 1e-12);
        let report = align_frames_by_timestamp(
            &reference,
            &distorted,
            TimestampPairingOptions {
                max_delta_seconds: 1e-9,
                allow_reuse_distorted: false,
            },
            transform,
        );
        assert_eq!(report.pairs.len(), 3);
        assert_eq!(report.max_delta_seconds, Some(0.0));
        assert!(report.unmatched_reference_indices.is_empty());
        assert!(report.unmatched_distorted_indices.is_empty());
    }

    #[test]
    fn timestamp_alignment_reports_invalid_and_unmatched_frames() {
        let reference = [
            TimestampedFrame {
                index: 0,
                pts_seconds: 0.0,
            },
            TimestampedFrame {
                index: 1,
                pts_seconds: f64::NAN,
            },
            TimestampedFrame {
                index: 2,
                pts_seconds: 2.0,
            },
        ];
        let distorted = [TimestampedFrame {
            index: 10,
            pts_seconds: 0.001,
        }];

        let report = align_frames_by_timestamp(
            &reference,
            &distorted,
            TimestampPairingOptions {
                max_delta_seconds: 0.01,
                allow_reuse_distorted: false,
            },
            TimestampTransform::IDENTITY,
        );
        assert_eq!(report.pairs.len(), 1);
        assert_eq!(report.unmatched_reference_indices, [1, 2]);
        assert!(report.unmatched_distorted_indices.is_empty());
    }
}
