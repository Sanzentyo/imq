//! SSIM metrics over luma samples.

use super::{
    Direction, Metric, MetricOutput, SampleDomain, ensure_same_dimensions, for_each_sample_pair,
    read_luma,
};
use crate::frame::{FrameView, Validated};
use crate::{Error, Result};

/// Global-luma SSIM.
///
/// This is a deterministic, allocation-free global variant. Use [`WindowedSsim`]
/// when local structural changes should affect the score more strongly than a
/// single whole-image mean/variance/covariance aggregate can capture.
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

        let score = ssim_score(mean_x, mean_y, var_x, var_y, cov_xy, self.c1, self.c2);

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

/// Mean windowed luma SSIM over non-overlapping tiles.
///
/// The score is the sample-weighted mean of per-window SSIM values. Partial edge
/// windows are included and weighted by their sample count, so every luma sample
/// contributes exactly once. This metric is deterministic, allocation-free, and
/// intentionally CPU-only; it is not multi-scale SSIM.
#[derive(Debug, Clone, Copy)]
pub struct WindowedSsim {
    c1: f64,
    c2: f64,
    window_width: usize,
    window_height: usize,
}

impl WindowedSsim {
    /// Default non-overlapping SSIM window width in luma samples.
    pub const DEFAULT_WINDOW_WIDTH: usize = 8;
    /// Default non-overlapping SSIM window height in luma samples.
    pub const DEFAULT_WINDOW_HEIGHT: usize = 8;

    /// Creates windowed SSIM with 8x8 non-overlapping windows and common constants.
    pub fn new() -> Self {
        Self::with_window(Self::DEFAULT_WINDOW_WIDTH, Self::DEFAULT_WINDOW_HEIGHT)
    }

    /// Creates windowed SSIM with an explicit non-overlapping window size.
    ///
    /// Zero dimensions are clamped to one luma sample to keep construction
    /// infallible and avoid a later zero-step iterator panic.
    pub fn with_window(window_width: usize, window_height: usize) -> Self {
        Self::with_window_and_constants(window_width, window_height, 0.01 * 0.01, 0.03 * 0.03)
    }

    /// Creates windowed SSIM with explicit window size and constants.
    ///
    /// Zero dimensions are clamped to one luma sample.
    pub fn with_window_and_constants(
        window_width: usize,
        window_height: usize,
        c1: f64,
        c2: f64,
    ) -> Self {
        Self {
            c1,
            c2,
            window_width: window_width.max(1),
            window_height: window_height.max(1),
        }
    }

    /// Returns the configured non-overlapping window size.
    pub fn window(&self) -> (usize, usize) {
        (self.window_width, self.window_height)
    }
}

impl Default for WindowedSsim {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for WindowedSsim {
    fn name(&self) -> &'static str {
        "wssim"
    }

    fn compare(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<MetricOutput> {
        let dims = ensure_same_dimensions(reference, distorted)?;
        let (w, h) = dims.as_usize()?;

        let mut windows = 0usize;
        let mut total_samples = 0usize;
        let mut weighted_sum = 0.0;
        let mut unweighted_sum = 0.0;
        let mut min_score = f64::INFINITY;
        let mut max_score = f64::NEG_INFINITY;

        for y in (0..h).step_by(self.window_height) {
            for x in (0..w).step_by(self.window_width) {
                let stats = window_ssim(reference, distorted, x, y, self)?;
                windows += 1;
                total_samples += stats.count;
                weighted_sum += stats.score * stats.count as f64;
                unweighted_sum += stats.score;
                min_score = min_score.min(stats.score);
                max_score = max_score.max(stats.score);
            }
        }

        if total_samples == 0 {
            return Err(Error::unsupported(
                "windowed SSIM received zero comparable samples",
            ));
        }

        let score = weighted_sum / total_samples as f64;
        let unweighted_mean = unweighted_sum / windows as f64;

        Ok(
            MetricOutput::new("wssim", score, "unitless", Direction::HigherIsBetter)
                .with_detail("windows", windows as f64)
                .with_detail("samples", total_samples as f64)
                .with_detail("window_width", self.window_width as f64)
                .with_detail("window_height", self.window_height as f64)
                .with_detail("unweighted_mean", unweighted_mean)
                .with_detail("min_window_ssim", min_score)
                .with_detail("max_window_ssim", max_score),
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct WindowStats {
    count: usize,
    score: f64,
}

fn window_ssim(
    reference: &FrameView<'_, Validated>,
    distorted: &FrameView<'_, Validated>,
    x0: usize,
    y0: usize,
    metric: &WindowedSsim,
) -> Result<WindowStats> {
    let dims = reference.dimensions();
    let (frame_width, frame_height) = dims.as_usize()?;
    let x1 = x0.saturating_add(metric.window_width).min(frame_width);
    let y1 = y0.saturating_add(metric.window_height).min(frame_height);

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            sum_x += read_luma(reference, x, y)?;
            sum_y += read_luma(distorted, x, y)?;
            count += 1;
        }
    }

    if count == 0 {
        return Err(Error::unsupported("windowed SSIM received an empty window"));
    }

    let mean_x = sum_x / count as f64;
    let mean_y = sum_y / count as f64;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut cov_xy = 0.0;

    for y in y0..y1 {
        for x in x0..x1 {
            let dx = read_luma(reference, x, y)? - mean_x;
            let dy = read_luma(distorted, x, y)? - mean_y;
            var_x += dx * dx;
            var_y += dy * dy;
            cov_xy += dx * dy;
        }
    }

    let denom = (count.saturating_sub(1)).max(1) as f64;
    var_x /= denom;
    var_y /= denom;
    cov_xy /= denom;

    Ok(WindowStats {
        count,
        score: ssim_score(mean_x, mean_y, var_x, var_y, cov_xy, metric.c1, metric.c2),
    })
}

fn ssim_score(
    mean_x: f64,
    mean_y: f64,
    var_x: f64,
    var_y: f64,
    cov_xy: f64,
    c1: f64,
    c2: f64,
) -> f64 {
    let numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * cov_xy + c2);
    let denominator = (mean_x * mean_x + mean_y * mean_y + c1) * (var_x + var_y + c2);
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FrameOwned, PixelFormat};

    #[test]
    fn windowed_ssim_is_one_for_identical_frame() {
        let frame = FrameOwned::packed_tight(
            vec![0, 32, 64, 96, 128, 160, 192, 255],
            4,
            2,
            PixelFormat::Luma8,
        )
        .unwrap();
        let output = WindowedSsim::with_window(2, 2)
            .compare(&frame.as_view(), &frame.as_view())
            .unwrap();
        assert!((output.score - 1.0).abs() < 1e-12);
        assert_eq!(output.details.get("windows").copied(), Some(2.0));
        assert_eq!(output.details.get("samples").copied(), Some(8.0));
    }

    #[test]
    fn windowed_ssim_includes_partial_edge_windows() {
        let reference =
            FrameOwned::packed_tight(vec![0, 64, 128, 255, 32, 96], 3, 2, PixelFormat::Luma8)
                .unwrap();
        let distorted =
            FrameOwned::packed_tight(vec![0, 64, 120, 250, 40, 90], 3, 2, PixelFormat::Luma8)
                .unwrap();
        let output = WindowedSsim::with_window(2, 1)
            .compare(&reference.as_view(), &distorted.as_view())
            .unwrap();
        assert_eq!(output.details.get("windows").copied(), Some(4.0));
        assert_eq!(output.details.get("samples").copied(), Some(6.0));
        assert!(output.score.is_finite());
    }
}
