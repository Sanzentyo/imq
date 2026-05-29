//! Report structs returned by image and video comparisons.

use crate::frame::{Dimensions, FormatSpec};
use crate::metrics::MetricOutput;

/// Full-reference comparison report for one pair of images/frames.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComparisonReport {
    /// Optional caller-provided reference label/path.
    pub reference: Option<String>,
    /// Optional caller-provided distorted label/path.
    pub distorted: Option<String>,
    /// Frame dimensions.
    pub dimensions: Dimensions,
    /// Reference format.
    pub reference_format: FormatSpec,
    /// Distorted format.
    pub distorted_format: FormatSpec,
    /// Metric outputs.
    pub metrics: Vec<MetricOutput>,
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
            dimensions,
            reference_format,
            distorted_format,
            metrics,
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
