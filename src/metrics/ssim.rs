//! SSIM metric over luma samples.

use super::{
    Direction, Metric, MetricOutput, SampleDomain, ensure_same_dimensions, for_each_sample_pair,
};
use crate::Result;
use crate::frame::{FrameView, Validated};

/// Global-luma SSIM.
///
/// This is a deterministic, allocation-free global variant. For strict paper-style
/// windowed SSIM, wrap this crate's sample iterator or add a tiled implementation;
/// the public metric contract is intentionally small.
#[derive(Debug, Clone, Copy)]
pub struct Ssim {
    c1: f64,
    c2: f64,
}

impl Ssim {
    /// Creates SSIM with the common constants `(0.01^2, 0.03^2)` for normalized samples.
    pub fn new() -> Self {
        Self {
            c1: 0.01 * 0.01,
            c2: 0.03 * 0.03,
        }
    }

    /// Creates SSIM with explicit constants.
    pub fn with_constants(c1: f64, c2: f64) -> Self {
        Self { c1, c2 }
    }
}

impl Default for Ssim {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for Ssim {
    fn name(&self) -> &'static str {
        "ssim"
    }

    fn compare(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<MetricOutput> {
        let dims = ensure_same_dimensions(reference, distorted)?;
        let n = dims.pixels()?;

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for_each_sample_pair(reference, distorted, SampleDomain::Luma, |x, y| {
            sum_x += x;
            sum_y += y;
        })?;
        let mean_x = sum_x / n as f64;
        let mean_y = sum_y / n as f64;

        let mut var_x = 0.0;
        let mut var_y = 0.0;
        let mut cov_xy = 0.0;
        for_each_sample_pair(reference, distorted, SampleDomain::Luma, |x, y| {
            let dx = x - mean_x;
            let dy = y - mean_y;
            var_x += dx * dx;
            var_y += dy * dy;
            cov_xy += dx * dy;
        })?;

        let denom = (n.saturating_sub(1)).max(1) as f64;
        var_x /= denom;
        var_y /= denom;
        cov_xy /= denom;

        let numerator = (2.0 * mean_x * mean_y + self.c1) * (2.0 * cov_xy + self.c2);
        let denominator = (mean_x * mean_x + mean_y * mean_y + self.c1) * (var_x + var_y + self.c2);
        let score = numerator / denominator;

        Ok(
            MetricOutput::new("ssim", score, "unitless", Direction::HigherIsBetter)
                .with_detail("mean_reference", mean_x)
                .with_detail("mean_distorted", mean_y)
                .with_detail("variance_reference", var_x)
                .with_detail("variance_distorted", var_y)
                .with_detail("covariance", cov_xy)
                .with_detail("samples", n as f64),
        )
    }
}
