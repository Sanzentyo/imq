//! Report structs returned by image and video comparisons.

use crate::frame::{Dimensions, FormatSpec};
use crate::gate::{BaselineGateEvaluation, GateEvaluation};
use crate::metrics::{AlphaDiagnostics, MetricOutput};

/// Full-reference comparison report for one pair of images/frames.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComparisonReport {
    /// Optional caller-provided reference label/path.
    pub reference: Option<String>,
    /// Optional caller-provided distorted label/path.
    pub distorted: Option<String>,
    /// Structured reference input metadata.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub reference_input: Option<ComparisonInput>,
    /// Structured candidate/distorted input metadata.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub candidate_input: Option<ComparisonInput>,
    /// Frame dimensions.
    pub dimensions: Dimensions,
    /// Reference format.
    pub reference_format: FormatSpec,
    /// Distorted format.
    pub distorted_format: FormatSpec,
    /// Metric outputs.
    pub metrics: Vec<MetricOutput>,
    /// Alpha diagnostics, when computed.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub alpha: Option<AlphaDiagnostics>,
    /// Threshold gate result, when thresholds were requested.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub gate: Option<ComparisonGateReport>,
}

impl ComparisonReport {
    /// Creates a report.
    pub fn new(
        dimensions: Dimensions,
        reference_format: FormatSpec,
        distorted_format: FormatSpec,
        metrics: Vec<MetricOutput>,
    ) -> Self {
        Self {
            reference: None,
            distorted: None,
            reference_input: None,
            candidate_input: None,
            dimensions,
            reference_format,
            distorted_format,
            metrics,
            alpha: None,
            gate: None,
        }
    }

    /// Adds labels/paths.
    pub fn with_labels(
        mut self,
        reference: impl Into<String>,
        distorted: impl Into<String>,
    ) -> Self {
        self.reference = Some(reference.into());
        self.distorted = Some(distorted.into());
        self
    }

    /// Adds structured input metadata.
    pub fn with_inputs(mut self, reference: ComparisonInput, candidate: ComparisonInput) -> Self {
        self.reference_input = Some(reference);
        self.candidate_input = Some(candidate);
        self
    }

    /// Adds alpha diagnostics.
    pub fn with_alpha(mut self, alpha: AlphaDiagnostics) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Adds threshold gate details.
    pub fn with_gate(mut self, gate: ComparisonGateReport) -> Self {
        self.gate = Some(gate);
        self
    }

    /// Serializes as pretty JSON.
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    pub fn to_json_pretty(&self) -> crate::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Minimal machine-readable input metadata for comparison reports.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComparisonInput {
    /// Filesystem path when the input came from a path.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub path: Option<String>,
    /// `imqraw` bundle index when selected by index.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub index: Option<usize>,
    /// `imqraw` tag when selected by tag.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub tag: Option<String>,
    /// Record label when provided by the input.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub label: Option<String>,
}

impl ComparisonInput {
    /// Creates metadata for a path input.
    pub fn path(path: impl Into<String>) -> Self {
        Self {
            path: Some(path.into()),
            ..Self::default()
        }
    }

    /// Creates metadata for a labeled non-path input such as stdin.
    pub fn label(label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            ..Self::default()
        }
    }

    /// Creates metadata for an `imqraw` index selection.
    pub fn imqraw_index(index: usize, label: Option<String>) -> Self {
        Self {
            index: Some(index),
            label,
            ..Self::default()
        }
    }

    /// Creates metadata for an `imqraw` tag selection.
    pub fn imqraw_tag(tag: impl Into<String>, label: Option<String>) -> Self {
        Self {
            tag: Some(tag.into()),
            label,
            ..Self::default()
        }
    }
}

/// Thresholds applied to an image comparison.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComparisonThresholds {
    /// Metric key selected for PSNR gate checks.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub selected_metric: Option<String>,
    /// Minimum selected metric score.
    #[cfg_attr(
        feature = "serde",
        serde(
            with = "crate::serde_f64::option",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub fail_under: Option<f64>,
    /// Maximum selected metric channel delta in normalized units.
    #[cfg_attr(
        feature = "serde",
        serde(
            with = "crate::serde_f64::option",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub max_selected_channel_delta: Option<f64>,
    /// Maximum alpha delta in 8-bit code units.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub max_alpha_delta: Option<u8>,
    /// Maximum alpha mismatch count.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub max_alpha_mismatches: Option<u64>,
    /// Maximum alpha mismatch count above one 8-bit LSB.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub max_alpha_mismatches_beyond_one_lsb: Option<u64>,
}

/// Result of applying comparison thresholds.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComparisonGateReport {
    /// Thresholds that were applied.
    pub thresholds: ComparisonThresholds,
    /// Whether all thresholds passed.
    pub passed: bool,
    /// Human-readable failure reasons.
    pub failures: Vec<String>,
}

/// Structured report produced by the supplemental still-image quality gate.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QualityGateReport {
    /// Reference/original image label or path.
    pub reference: String,
    /// Candidate/distorted image label or path.
    pub candidate: String,
    /// Compared frame dimensions.
    pub dimensions: Dimensions,
    /// Reference format metadata used by metric conversion.
    pub reference_format: FormatSpec,
    /// Candidate format metadata used by metric conversion.
    pub candidate_format: FormatSpec,
    /// Optional baseline candidate label or path used by relative rules.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub baseline: Option<String>,
    /// Baseline format metadata, when a baseline was compared.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub baseline_format: Option<FormatSpec>,
    /// Candidate metric outputs, including distribution and channel details.
    pub candidate_metrics: Vec<MetricOutput>,
    /// Baseline metric outputs, when a baseline was compared.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub baseline_metrics: Option<Vec<MetricOutput>>,
    /// Absolute threshold evaluation.
    pub thresholds: GateEvaluation,
    /// Baseline-relative threshold evaluation, when requested.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub baseline_thresholds: Option<BaselineGateEvaluation>,
    /// Whether every absolute and baseline-relative check passed.
    pub passed: bool,
}

impl QualityGateReport {
    /// Creates a structured quality gate report and derives its aggregate status.
    pub fn new(
        reference: impl Into<String>,
        candidate: impl Into<String>,
        dimensions: Dimensions,
        reference_format: FormatSpec,
        candidate_format: FormatSpec,
        candidate_metrics: Vec<MetricOutput>,
        thresholds: GateEvaluation,
    ) -> Self {
        let passed = thresholds.passed;
        Self {
            reference: reference.into(),
            candidate: candidate.into(),
            dimensions,
            reference_format,
            candidate_format,
            baseline: None,
            baseline_format: None,
            candidate_metrics,
            baseline_metrics: None,
            thresholds,
            baseline_thresholds: None,
            passed,
        }
    }

    /// Adds baseline metrics and their threshold evaluation.
    pub fn with_baseline(
        mut self,
        baseline: impl Into<String>,
        format: FormatSpec,
        metrics: Vec<MetricOutput>,
        evaluation: BaselineGateEvaluation,
    ) -> Self {
        self.passed &= evaluation.passed;
        self.baseline = Some(baseline.into());
        self.baseline_format = Some(format);
        self.baseline_metrics = Some(metrics);
        self.baseline_thresholds = Some(evaluation);
        self
    }

    /// Serializes as pretty JSON.
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    pub fn to_json_pretty(&self) -> crate::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Report for one frame pair in a video comparison.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameReport {
    /// Zero-based compared frame index in decode order.
    pub frame_index: u64,
    /// Optional presentation timestamp in seconds when known.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub pts_seconds: Option<f64>,
    /// Per-frame metric results.
    pub metrics: Vec<MetricOutput>,
}

/// Video-level report.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VideoReport {
    /// Reference label/path.
    pub reference: String,
    /// Distorted label/path.
    pub distorted: String,
    /// Compared dimensions.
    pub dimensions: Dimensions,
    /// Number of frame pairs compared.
    pub compared_frames: u64,
    /// Per-frame reports.
    pub frames: Vec<FrameReport>,
    /// Mean metric scores over all frame pairs.
    pub mean_metrics: Vec<MetricOutput>,
}

impl VideoReport {
    /// Serializes as pretty JSON.
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    pub fn to_json_pretty(&self) -> crate::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Report for one explicitly ordered video frame pair.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FramePairReport {
    /// Zero-based pair index preserving user input order.
    pub pair_index: u64,
    /// Reference frame index.
    pub reference_frame_index: u64,
    /// Distorted/candidate frame index.
    pub distorted_frame_index: u64,
    /// Optional canonical/reporting frame index from 3-tuple syntax.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub label_frame_index: Option<u64>,
    /// Reference PTS when known.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub reference_pts_seconds: Option<f64>,
    /// Distorted/candidate PTS when known.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::option"))]
    pub distorted_pts_seconds: Option<f64>,
    /// Per-pair metric results.
    pub metrics: Vec<MetricOutput>,
}

/// Report for explicitly ordered video frame-pair comparisons.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VideoFramePairComparisonReport {
    /// Reference label/path/URI.
    pub reference: String,
    /// Distorted label/path/URI.
    pub distorted: String,
    /// Compared dimensions.
    pub dimensions: Dimensions,
    /// Ordered frame-pair reports.
    pub pairs: Vec<FramePairReport>,
    /// Mean metric scores over all pairs.
    pub mean_metrics: Vec<MetricOutput>,
}

impl VideoFramePairComparisonReport {
    /// Serializes as pretty JSON.
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    pub fn to_json_pretty(&self) -> crate::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}
