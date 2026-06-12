//! SSIM-family metrics over luma samples.
//!
//! [`Ssim`] keeps the original global-luma score. [`WindowedSsim`] computes
//! local box-window SSIM, and [`MsSsim`] computes a deterministic five-scale
//! luma MS-SSIM score without adding codec, I/O, or model dependencies.

use super::{
    Direction, Metric, MetricOutput, SampleDomain, ensure_same_dimensions, for_each_sample_pair,
};
use crate::frame::{FrameView, Validated};
use crate::{Error, Result};

const DEFAULT_WINDOW_WIDTH: usize = 8;
const DEFAULT_WINDOW_HEIGHT: usize = 8;
const DEFAULT_MS_SSIM_STRIDE: usize = 4;
const DEFAULT_MS_SSIM_WEIGHTS: [f64; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

/// Global-luma SSIM.
///
/// This is a deterministic whole-image SSIM variant. Use [`WindowedSsim`] when
/// local structural changes should be reflected more strongly, or [`MsSsim`] for
/// a deterministic multi-scale score.
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

        let (score, contrast_structure) =
            ssim_components(mean_x, mean_y, var_x, var_y, cov_xy, self.c1, self.c2);

        Ok(
            MetricOutput::new("ssim", score, "unitless", Direction::HigherIsBetter)
                .with_detail("contrast_structure", contrast_structure)
                .with_detail("mean_reference", mean_x)
                .with_detail("mean_distorted", mean_y)
                .with_detail("variance_reference", var_x)
                .with_detail("variance_distorted", var_y)
                .with_detail("covariance", cov_xy)
                .with_detail("samples", n as f64),
        )
    }
}

/// Mean local luma SSIM over deterministic box windows.
///
/// `new()` keeps the first implementation's non-overlapping 8x8 behavior. Use
/// [`WindowedSsim::with_sliding_window`] to request overlapping windows such as
/// an 8x8 window with a 4-pixel stride.
#[derive(Debug, Clone, Copy)]
pub struct WindowedSsim {
    c1: f64,
    c2: f64,
    window_width: usize,
    window_height: usize,
    stride_x: usize,
    stride_y: usize,
}

impl WindowedSsim {
    /// Default non-overlapping SSIM window width in luma samples.
    pub const DEFAULT_WINDOW_WIDTH: usize = DEFAULT_WINDOW_WIDTH;
    /// Default non-overlapping SSIM window height in luma samples.
    pub const DEFAULT_WINDOW_HEIGHT: usize = DEFAULT_WINDOW_HEIGHT;

    /// Creates windowed SSIM with 8x8 non-overlapping windows and common constants.
    pub fn new() -> Self {
        Self::with_window(Self::DEFAULT_WINDOW_WIDTH, Self::DEFAULT_WINDOW_HEIGHT)
    }

    /// Creates windowed SSIM with an explicit non-overlapping window size.
    ///
    /// Zero dimensions are clamped to one luma sample to preserve infallible
    /// construction and avoid zero-step iteration.
    pub fn with_window(window_width: usize, window_height: usize) -> Self {
        let window_width = window_width.max(1);
        let window_height = window_height.max(1);
        Self {
            c1: 0.01 * 0.01,
            c2: 0.03 * 0.03,
            window_width,
            window_height,
            stride_x: window_width,
            stride_y: window_height,
        }
    }

    /// Creates windowed SSIM with explicit window dimensions and strides.
    pub fn with_sliding_window(
        window_width: usize,
        window_height: usize,
        stride_x: usize,
        stride_y: usize,
    ) -> Result<Self> {
        if window_width == 0 || window_height == 0 {
            return Err(Error::unsupported(
                "SSIM window dimensions must be non-zero",
            ));
        }
        if stride_x == 0 || stride_y == 0 {
            return Err(Error::unsupported("SSIM window strides must be non-zero"));
        }
        Ok(Self {
            c1: 0.01 * 0.01,
            c2: 0.03 * 0.03,
            window_width,
            window_height,
            stride_x,
            stride_y,
        })
    }

    /// Returns the configured window dimensions.
    pub fn window(&self) -> (usize, usize) {
        (self.window_width, self.window_height)
    }

    /// Returns the configured window strides.
    pub fn stride(&self) -> (usize, usize) {
        (self.stride_x, self.stride_y)
    }

    /// Creates SSIM with explicit constants while preserving the window layout.
    pub fn with_constants(mut self, c1: f64, c2: f64) -> Self {
        self.c1 = c1;
        self.c2 = c2;
        self
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
        let (width, height, reference_luma, distorted_luma) =
            collect_luma_pairs(reference, distorted)?;
        let stats = windowed_luma_stats(
            &reference_luma,
            &distorted_luma,
            ImageSize { width, height },
            WindowLayout {
                width: self.window_width,
                height: self.window_height,
                stride_x: self.stride_x,
                stride_y: self.stride_y,
            },
            SsimConstants {
                c1: self.c1,
                c2: self.c2,
            },
        )?;

        Ok(MetricOutput::new(
            "wssim",
            stats.mean_ssim,
            "unitless",
            Direction::HigherIsBetter,
        )
        .with_detail("windows", stats.windows as f64)
        .with_detail("samples", (width * height) as f64)
        .with_detail("window_width", stats.window_width as f64)
        .with_detail("window_height", stats.window_height as f64)
        .with_detail("stride_x", self.stride_x as f64)
        .with_detail("stride_y", self.stride_y as f64)
        .with_detail("mean_contrast_structure", stats.mean_cs)
        .with_detail("unweighted_mean", stats.mean_ssim)
        .with_detail("min_window_ssim", stats.min_ssim)
        .with_detail("max_window_ssim", stats.max_ssim))
    }
}

/// Multi-scale SSIM over luma samples.
///
/// The implementation uses the standard five MS-SSIM weights, deterministic box
/// windows, and 2x2 box downsampling. If an image reaches 1x1 before five scales,
/// available weights are renormalized across the scales that were computed.
#[derive(Debug, Clone, Copy)]
pub struct MsSsim {
    c1: f64,
    c2: f64,
    window_width: usize,
    window_height: usize,
    stride_x: usize,
    stride_y: usize,
    weights: [f64; 5],
}

impl MsSsim {
    /// Creates luma MS-SSIM with 8x8 windows, 4-pixel stride, and standard five-scale weights.
    pub fn new() -> Self {
        Self {
            c1: 0.01 * 0.01,
            c2: 0.03 * 0.03,
            window_width: DEFAULT_WINDOW_WIDTH,
            window_height: DEFAULT_WINDOW_HEIGHT,
            stride_x: DEFAULT_MS_SSIM_STRIDE,
            stride_y: DEFAULT_MS_SSIM_STRIDE,
            weights: DEFAULT_MS_SSIM_WEIGHTS,
        }
    }

    /// Creates luma MS-SSIM with explicit local window dimensions and strides.
    pub fn with_window(
        window_width: usize,
        window_height: usize,
        stride_x: usize,
        stride_y: usize,
    ) -> Result<Self> {
        if window_width == 0 || window_height == 0 {
            return Err(Error::unsupported(
                "MS-SSIM window dimensions must be non-zero",
            ));
        }
        if stride_x == 0 || stride_y == 0 {
            return Err(Error::unsupported(
                "MS-SSIM window strides must be non-zero",
            ));
        }
        Ok(Self {
            window_width,
            window_height,
            stride_x,
            stride_y,
            ..Self::new()
        })
    }

    /// Creates luma MS-SSIM with explicit constants.
    pub fn with_constants(mut self, c1: f64, c2: f64) -> Self {
        self.c1 = c1;
        self.c2 = c2;
        self
    }

    /// Creates luma MS-SSIM with explicit five-scale weights.
    pub fn with_weights(mut self, weights: [f64; 5]) -> Self {
        self.weights = weights;
        self
    }
}

impl Default for MsSsim {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for MsSsim {
    fn name(&self) -> &'static str {
        "ms-ssim"
    }

    fn compare(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<MetricOutput> {
        let (mut width, mut height, mut reference_luma, mut distorted_luma) =
            collect_luma_pairs(reference, distorted)?;
        let mut scale_ssim = Vec::with_capacity(self.weights.len());
        let mut scale_cs = Vec::with_capacity(self.weights.len());
        let mut scale_widths = Vec::with_capacity(self.weights.len());
        let mut scale_heights = Vec::with_capacity(self.weights.len());

        for scale in 0..self.weights.len() {
            let stats = windowed_luma_stats(
                &reference_luma,
                &distorted_luma,
                ImageSize { width, height },
                WindowLayout {
                    width: self.window_width,
                    height: self.window_height,
                    stride_x: self.stride_x,
                    stride_y: self.stride_y,
                },
                SsimConstants {
                    c1: self.c1,
                    c2: self.c2,
                },
            )?;
            scale_ssim.push(stats.mean_ssim);
            scale_cs.push(stats.mean_cs);
            scale_widths.push(width);
            scale_heights.push(height);

            if scale + 1 == self.weights.len() || (width == 1 && height == 1) {
                break;
            }

            let (next_width, next_height, next_reference) =
                downsample_by_two(&reference_luma, width, height);
            let (_, _, next_distorted) = downsample_by_two(&distorted_luma, width, height);
            width = next_width;
            height = next_height;
            reference_luma = next_reference;
            distorted_luma = next_distorted;
        }

        let scales = scale_ssim.len();
        let weight_sum: f64 = self.weights[..scales].iter().copied().sum();
        if weight_sum <= 0.0 || !weight_sum.is_finite() {
            return Err(Error::unsupported(
                "MS-SSIM weights must sum to a positive finite value",
            ));
        }

        let mut score = 1.0;
        for scale in 0..scales {
            let weight = self.weights[scale] / weight_sum;
            let term = if scale + 1 == scales {
                scale_ssim[scale]
            } else {
                scale_cs[scale]
            };
            score *= clamp_ms_ssim_term(term).powf(weight);
        }

        let mut output = MetricOutput::new("ms-ssim", score, "unitless", Direction::HigherIsBetter)
            .with_detail("scales", scales as f64)
            .with_detail("window_width", self.window_width as f64)
            .with_detail("window_height", self.window_height as f64)
            .with_detail("stride_x", self.stride_x as f64)
            .with_detail("stride_y", self.stride_y as f64)
            .with_detail("weight_sum", weight_sum);

        for scale in 0..scales {
            output = output
                .with_detail(format!("scale{}_ssim", scale + 1), scale_ssim[scale])
                .with_detail(
                    format!("scale{}_contrast_structure", scale + 1),
                    scale_cs[scale],
                )
                .with_detail(
                    format!("scale{}_width", scale + 1),
                    scale_widths[scale] as f64,
                )
                .with_detail(
                    format!("scale{}_height", scale + 1),
                    scale_heights[scale] as f64,
                );
        }

        Ok(output)
    }
}

#[derive(Debug, Clone, Copy)]
struct WindowStats {
    mean_ssim: f64,
    mean_cs: f64,
    min_ssim: f64,
    max_ssim: f64,
    windows: usize,
    window_width: usize,
    window_height: usize,
}

#[derive(Debug, Clone, Copy)]
struct ImageSize {
    width: usize,
    height: usize,
}

#[derive(Debug, Clone, Copy)]
struct WindowLayout {
    width: usize,
    height: usize,
    stride_x: usize,
    stride_y: usize,
}

#[derive(Debug, Clone, Copy)]
struct SsimConstants {
    c1: f64,
    c2: f64,
}

fn collect_luma_pairs(
    reference: &FrameView<'_, Validated>,
    distorted: &FrameView<'_, Validated>,
) -> Result<(usize, usize, Vec<f64>, Vec<f64>)> {
    let dims = ensure_same_dimensions(reference, distorted)?;
    let (width, height) = dims.as_usize()?;
    let samples = dims.pixels()?;
    let mut reference_luma = Vec::with_capacity(samples);
    let mut distorted_luma = Vec::with_capacity(samples);

    for_each_sample_pair(
        reference,
        distorted,
        SampleDomain::Luma,
        |reference, distorted| {
            reference_luma.push(reference);
            distorted_luma.push(distorted);
        },
    )?;

    Ok((width, height, reference_luma, distorted_luma))
}

fn windowed_luma_stats(
    reference: &[f64],
    distorted: &[f64],
    image: ImageSize,
    layout: WindowLayout,
    constants: SsimConstants,
) -> Result<WindowStats> {
    let ImageSize { width, height } = image;
    if width == 0 || height == 0 {
        return Err(Error::unsupported("SSIM requires non-zero dimensions"));
    }
    if reference.len() != width * height || distorted.len() != width * height {
        return Err(Error::invalid_frame(
            "luma buffer length does not match dimensions",
        ));
    }

    let window_width = layout.width.max(1).min(width);
    let window_height = layout.height.max(1).min(height);
    let stride_x = layout.stride_x.max(1);
    let stride_y = layout.stride_y.max(1);
    let x_starts = window_starts(width, window_width, stride_x);
    let y_starts = window_starts(height, window_height, stride_y);

    let mut windows = 0usize;
    let mut sum_ssim = 0.0;
    let mut sum_cs = 0.0;
    let mut min_ssim = f64::INFINITY;
    let mut max_ssim = f64::NEG_INFINITY;

    for y0 in y_starts {
        for &x0 in &x_starts {
            let (ssim, cs) = local_ssim(
                reference,
                distorted,
                image,
                x0,
                y0,
                WindowLayout {
                    width: window_width,
                    height: window_height,
                    stride_x,
                    stride_y,
                },
                constants,
            );
            windows += 1;
            sum_ssim += ssim;
            sum_cs += cs;
            min_ssim = min_ssim.min(ssim);
            max_ssim = max_ssim.max(ssim);
        }
    }

    if windows == 0 {
        return Err(Error::unsupported("SSIM produced no local windows"));
    }

    Ok(WindowStats {
        mean_ssim: sum_ssim / windows as f64,
        mean_cs: sum_cs / windows as f64,
        min_ssim,
        max_ssim,
        windows,
        window_width,
        window_height,
    })
}

fn window_starts(size: usize, window: usize, stride: usize) -> Vec<usize> {
    let max_start = size.saturating_sub(window);
    let mut starts = Vec::new();
    let mut current = 0usize;
    loop {
        starts.push(current);
        if current == max_start {
            break;
        }
        let next = current.saturating_add(stride).min(max_start);
        if next == current {
            break;
        }
        current = next;
    }
    starts
}

fn local_ssim(
    reference: &[f64],
    distorted: &[f64],
    image: ImageSize,
    x0: usize,
    y0: usize,
    layout: WindowLayout,
    constants: SsimConstants,
) -> (f64, f64) {
    let width = image.width;
    let window_width = layout.width;
    let window_height = layout.height;
    let n = (window_width * window_height) as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut sum_xy = 0.0;

    for y in y0..(y0 + window_height) {
        let row = y * width;
        for x in x0..(x0 + window_width) {
            let xr = reference[row + x];
            let yd = distorted[row + x];
            sum_x += xr;
            sum_y += yd;
            sum_x2 += xr * xr;
            sum_y2 += yd * yd;
            sum_xy += xr * yd;
        }
    }

    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let denom = ((window_width * window_height).saturating_sub(1)).max(1) as f64;
    let var_x = (sum_x2 - (sum_x * sum_x / n)).max(0.0) / denom;
    let var_y = (sum_y2 - (sum_y * sum_y / n)).max(0.0) / denom;
    let cov_xy = (sum_xy - (sum_x * sum_y / n)) / denom;

    ssim_components(
        mean_x,
        mean_y,
        var_x,
        var_y,
        cov_xy,
        constants.c1,
        constants.c2,
    )
}

fn ssim_components(
    mean_x: f64,
    mean_y: f64,
    var_x: f64,
    var_y: f64,
    cov_xy: f64,
    c1: f64,
    c2: f64,
) -> (f64, f64) {
    let luminance = (2.0 * mean_x * mean_y + c1) / (mean_x * mean_x + mean_y * mean_y + c1);
    let contrast_structure = (2.0 * cov_xy + c2) / (var_x + var_y + c2);
    (luminance * contrast_structure, contrast_structure)
}

fn downsample_by_two(samples: &[f64], width: usize, height: usize) -> (usize, usize, Vec<f64>) {
    let next_width = width.div_ceil(2);
    let next_height = height.div_ceil(2);
    let mut out = vec![0.0; next_width * next_height];

    for y in 0..next_height {
        for x in 0..next_width {
            let mut sum = 0.0;
            let mut count = 0.0;
            for dy in 0..2 {
                let source_y = y * 2 + dy;
                if source_y >= height {
                    continue;
                }
                for dx in 0..2 {
                    let source_x = x * 2 + dx;
                    if source_x >= width {
                        continue;
                    }
                    sum += samples[source_y * width + source_x];
                    count += 1.0;
                }
            }
            out[y * next_width + x] = sum / count;
        }
    }

    (next_width, next_height, out)
}

fn clamp_ms_ssim_term(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
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

    #[test]
    fn sliding_window_rejects_zero_stride() {
        assert!(WindowedSsim::with_sliding_window(8, 8, 0, 4).is_err());
        assert!(WindowedSsim::with_sliding_window(8, 8, 4, 0).is_err());
    }

    #[test]
    fn ms_ssim_is_one_for_identical_input() {
        let pixels: Vec<u8> = (0..1024).map(|value| (value % 251) as u8).collect();
        let frame = FrameOwned::packed_tight(pixels, 32, 32, PixelFormat::Luma8).unwrap();

        let output = MsSsim::new()
            .compare(&frame.as_view(), &frame.as_view())
            .unwrap();

        assert!((output.score - 1.0).abs() < 1e-12);
        assert!(*output.details.get("scales").unwrap() >= 1.0);
    }

    #[test]
    fn windowed_and_ms_ssim_drop_for_distortion() {
        let reference_pixels = vec![96u8; 64];
        let mut distorted_pixels = reference_pixels.clone();
        for (index, value) in distorted_pixels.iter_mut().enumerate() {
            if index % 2 == 0 {
                *value = 192;
            }
        }
        let reference =
            FrameOwned::packed_tight(reference_pixels, 8, 8, PixelFormat::Luma8).unwrap();
        let distorted =
            FrameOwned::packed_tight(distorted_pixels, 8, 8, PixelFormat::Luma8).unwrap();

        let windowed = WindowedSsim::new()
            .compare(&reference.as_view(), &distorted.as_view())
            .unwrap();
        let multi_scale = MsSsim::new()
            .compare(&reference.as_view(), &distorted.as_view())
            .unwrap();

        assert!(windowed.score < 1.0);
        assert!(multi_scale.score < 1.0);
    }

    #[test]
    fn ms_ssim_rejects_zero_window_parameters() {
        assert!(MsSsim::with_window(0, 8, 4, 4).is_err());
        assert!(MsSsim::with_window(8, 8, 0, 4).is_err());
    }
}
