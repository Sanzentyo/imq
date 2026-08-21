//! Basic deterministic full-reference metrics.

use super::{Direction, Metric, MetricOutput, SampleDomain, aggregate_error};
use crate::Result;
use crate::frame::{FrameView, Validated};

/// Mean squared error over normalized samples.
#[derive(Debug, Clone, Copy)]
pub struct Mse {
    domain: SampleDomain,
}

impl Mse {
    /// Creates MSE for a domain.
    pub fn new(domain: SampleDomain) -> Self {
        Self { domain }
    }
}

impl Metric for Mse {
    fn name(&self) -> &'static str {
        "mse"
    }

    fn compare(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<MetricOutput> {
        let stats = aggregate_error(reference, distorted, self.domain)?;
        Ok(MetricOutput::new(
            "mse",
            stats.mse(),
            "normalized_code^2",
            Direction::LowerIsBetter,
        )
        .with_error_details(&stats, self.domain))
    }
}

/// Root mean squared error over normalized samples.
#[derive(Debug, Clone, Copy)]
pub struct Rmse {
    domain: SampleDomain,
}

impl Rmse {
    /// Creates RMSE for a domain.
    pub fn new(domain: SampleDomain) -> Self {
        Self { domain }
    }
}

impl Metric for Rmse {
    fn name(&self) -> &'static str {
        "rmse"
    }

    fn compare(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<MetricOutput> {
        let stats = aggregate_error(reference, distorted, self.domain)?;
        Ok(MetricOutput::new(
            "rmse",
            stats.rmse(),
            "normalized_code",
            Direction::LowerIsBetter,
        )
        .with_error_details(&stats, self.domain))
    }
}

/// Peak signal-to-noise ratio in decibels.
#[derive(Debug, Clone, Copy)]
pub struct Psnr {
    domain: SampleDomain,
    peak: f64,
}

impl Psnr {
    /// Creates PSNR with normalized peak value 1.0.
    pub fn new(domain: SampleDomain) -> Self {
        Self { domain, peak: 1.0 }
    }

    /// Creates PSNR with an explicit normalized peak.
    pub fn with_peak(domain: SampleDomain, peak: f64) -> Self {
        Self { domain, peak }
    }
}

impl Metric for Psnr {
    fn name(&self) -> &'static str {
        "psnr"
    }

    fn compare(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<MetricOutput> {
        let stats = aggregate_error(reference, distorted, self.domain)?;
        let mse = stats.mse();
        let psnr = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * ((self.peak * self.peak) / mse).log10()
        };
        Ok(
            MetricOutput::new("psnr", psnr, "dB", Direction::HigherIsBetter)
                .with_detail("mse", mse)
                .with_error_details(&stats, self.domain),
        )
    }
}

/// Mean absolute error over normalized samples.
#[derive(Debug, Clone, Copy)]
pub struct Mae {
    domain: SampleDomain,
}

impl Mae {
    /// Creates MAE for a domain.
    pub fn new(domain: SampleDomain) -> Self {
        Self { domain }
    }
}

impl Metric for Mae {
    fn name(&self) -> &'static str {
        "mae"
    }

    fn compare(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<MetricOutput> {
        let stats = aggregate_error(reference, distorted, self.domain)?;
        Ok(MetricOutput::new(
            "mae",
            stats.mae(),
            "normalized_code",
            Direction::LowerIsBetter,
        )
        .with_error_details(&stats, self.domain))
    }
}

/// Maximum absolute error over normalized samples.
#[derive(Debug, Clone, Copy)]
pub struct MaxAbsoluteError {
    domain: SampleDomain,
}

impl MaxAbsoluteError {
    /// Creates max absolute error for a domain.
    pub fn new(domain: SampleDomain) -> Self {
        Self { domain }
    }
}

impl Metric for MaxAbsoluteError {
    fn name(&self) -> &'static str {
        "maxae"
    }

    fn compare(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<MetricOutput> {
        let stats = aggregate_error(reference, distorted, self.domain)?;
        Ok(MetricOutput::new(
            "maxae",
            stats.max_abs,
            "normalized_code",
            Direction::LowerIsBetter,
        )
        .with_error_details(&stats, self.domain))
    }
}
