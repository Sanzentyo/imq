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
    pub percentile: f64,
    /// Interpolated score at the percentile.
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
    /// Number of frame-level samples.
    pub samples: usize,
    /// Arithmetic mean score.
    pub mean: f64,
    /// Minimum score.
    pub min: f64,
    /// Maximum score.
    pub max: f64,
    /// Median score.
    pub median: f64,
    /// Requested percentile scores.
    pub percentiles: Vec<PercentileValue>,
    /// Worst frame index according to metric direction.
    pub worst_frame_index: Option<u64>,
    /// Worst frame timestamp when available.
    pub worst_pts_seconds: Option<f64>,
    /// Worst score according to metric direction.
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
    for frame in frames {
        if let Some(metric) = frame.metrics.iter().find(|metric| metric.name == name) {
            if metric.score.is_finite() {
                samples.push((
                    frame.frame_index,
                    frame.pts_seconds,
                    metric.score,
                    metric.unit.clone(),
                    metric.direction,
                ));
            }
        }
    }
    if samples.is_empty() {
        return None;
    }

    let unit = samples[0].3.clone();
    let direction = samples[0].4;
    let mut sorted_scores = samples.iter().map(|sample| sample.2).collect::<Vec<_>>();
    sorted_scores.sort_by(|a, b| a.total_cmp(b));
    let mean = sorted_scores.iter().sum::<f64>() / sorted_scores.len() as f64;
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
        mean,
        min,
        max,
        median,
        percentiles,
        worst_frame_index: worst.map(|sample| sample.0),
        worst_pts_seconds: worst.and_then(|sample| sample.1),
        worst_score: worst.map(|sample| sample.2),
    })
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
    pub pts_seconds: f64,
}

/// Options for timestamp-aware frame pairing.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimestampPairingOptions {
    /// Maximum allowed absolute timestamp difference in seconds.
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
    pub reference_pts_seconds: f64,
    /// Distorted timestamp in seconds.
    pub distorted_pts_seconds: f64,
    /// Absolute timestamp delta in seconds.
    pub delta_seconds: f64,
}

/// Pairs frames by nearest presentation timestamp.
pub fn pair_frames_by_timestamp(
    reference: &[TimestampedFrame],
    distorted: &[TimestampedFrame],
    options: TimestampPairingOptions,
) -> Vec<TimestampFramePair> {
    let mut pairs = Vec::new();
    let mut used = vec![false; distorted.len()];
    for reference_frame in reference {
        let mut best_index = None;
        let mut best_delta = f64::INFINITY;
        for (index, distorted_frame) in distorted.iter().enumerate() {
            if used[index] && !options.allow_reuse_distorted {
                continue;
            }
            let delta = (reference_frame.pts_seconds - distorted_frame.pts_seconds).abs();
            if delta < best_delta {
                best_delta = delta;
                best_index = Some(index);
            }
        }
        if let Some(index) = best_index {
            if best_delta <= options.max_delta_seconds {
                used[index] = true;
                let distorted_frame = distorted[index];
                pairs.push(TimestampFramePair {
                    reference_index: reference_frame.index,
                    distorted_index: distorted_frame.index,
                    reference_pts_seconds: reference_frame.pts_seconds,
                    distorted_pts_seconds: distorted_frame.pts_seconds,
                    delta_seconds: best_delta,
                });
            }
        }
    }
    pairs
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
}
