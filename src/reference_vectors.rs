//! Small deterministic reference vectors for metric regression tests.
//!
//! The vectors are intentionally tiny and codec-free. They are useful for
//! checking that core metric formulas and normalization behavior do not drift.

use crate::frame::{FrameOwned, PixelFormat};
use crate::metrics::MetricSet;
use crate::{Error, Result};

const IDENTICAL_REF: [u8; 4] = [0, 128, 255, 64];
const IDENTICAL_DIST: [u8; 4] = [0, 128, 255, 64];
const HALF_ERROR_REF: [u8; 4] = [0, 0, 255, 255];
const HALF_ERROR_DIST: [u8; 4] = [0, 255, 255, 0];

/// Expected scalar metrics for a reference vector.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReferenceMetricExpectations {
    /// Expected normalized MSE.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub mse: f64,
    /// Expected normalized MAE.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub mae: f64,
    /// Expected PSNR in dB; may be infinity for identical images.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub psnr: f64,
}

/// One luma8 reference/distorted fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceMetricVector {
    /// Stable vector name.
    pub name: &'static str,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Reference Luma8 samples.
    pub reference: &'static [u8],
    /// Distorted Luma8 samples.
    pub distorted: &'static [u8],
    /// Expected metric scores.
    pub expected: ReferenceMetricExpectations,
}

/// Returns built-in metric reference vectors.
pub fn reference_metric_vectors() -> Vec<ReferenceMetricVector> {
    vec![
        ReferenceMetricVector {
            name: "identical_luma4",
            width: 2,
            height: 2,
            reference: &IDENTICAL_REF,
            distorted: &IDENTICAL_DIST,
            expected: ReferenceMetricExpectations {
                mse: 0.0,
                mae: 0.0,
                psnr: f64::INFINITY,
            },
        },
        ReferenceMetricVector {
            name: "half_fullscale_luma4",
            width: 2,
            height: 2,
            reference: &HALF_ERROR_REF,
            distorted: &HALF_ERROR_DIST,
            expected: ReferenceMetricExpectations {
                mse: 0.5,
                mae: 0.5,
                psnr: 3.010_299_956_639_812,
            },
        },
    ]
}

/// Runs built-in reference vectors through the core metric pipeline.
pub fn assert_reference_vectors() -> Result<()> {
    for vector in reference_metric_vectors() {
        let reference = FrameOwned::packed_tight(
            vector.reference.to_vec(),
            vector.width,
            vector.height,
            PixelFormat::Luma8,
        )?;
        let distorted = FrameOwned::packed_tight(
            vector.distorted.to_vec(),
            vector.width,
            vector.height,
            PixelFormat::Luma8,
        )?;
        let outputs = MetricSet::from_csv("mse,mae,psnr")?
            .compare(&reference.as_view(), &distorted.as_view())?;
        assert_metric(&outputs, "mse", vector.expected.mse, vector.name)?;
        assert_metric(&outputs, "mae", vector.expected.mae, vector.name)?;
        assert_metric(&outputs, "psnr", vector.expected.psnr, vector.name)?;
    }
    Ok(())
}

fn assert_metric(
    outputs: &[crate::metrics::MetricOutput],
    name: &str,
    expected: f64,
    vector_name: &str,
) -> Result<()> {
    let actual = outputs
        .iter()
        .find(|output| output.name == name)
        .map(|output| output.score)
        .ok_or_else(|| {
            Error::unsupported(format!(
                "reference vector `{vector_name}` missing metric `{name}`"
            ))
        })?;
    let passed = if expected.is_infinite() {
        actual.is_infinite() && actual.is_sign_positive() == expected.is_sign_positive()
    } else {
        (actual - expected).abs() <= 1e-12
    };
    if passed {
        Ok(())
    } else {
        Err(Error::unsupported(format!(
            "reference vector `{vector_name}` metric `{name}` expected {expected}, got {actual}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_reference_vectors_match() {
        assert_reference_vectors().unwrap();
    }
}
