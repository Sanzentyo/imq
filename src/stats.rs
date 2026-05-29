//! Image statistics, color balance, and visual tendency analysis.

use crate::frame::{FrameView, PixelFormat, Validated};
use crate::{Error, Result};

/// Options controlling image statistics.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageStatisticsOptions {
    /// Number of histogram bins to emit for channel/luma/HSV distributions.
    pub histogram_bins: usize,
}

impl Default for ImageStatisticsOptions {
    fn default() -> Self {
        Self { histogram_bins: 16 }
    }
}

/// Full image statistics report.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageStatistics {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of pixels.
    pub pixels: u64,
    /// Pixel format used for analysis.
    pub pixel_format: PixelFormat,
    /// Per-channel statistics.
    pub channels: ChannelStatisticsSet,
    /// Perceptual luma statistics.
    pub luma: ChannelStatistics,
    /// HSV-derived statistics.
    pub hsv: HsvStatistics,
    /// RGB balance and cast indicators.
    pub color_balance: ColorBalance,
    /// High-level visual tendencies.
    pub tendencies: ImageTendencies,
    /// Spatial brightness distribution.
    pub spatial: SpatialStatistics,
    /// Histograms.
    pub histograms: ImageHistograms,
}

/// Per-channel RGB(A) statistics.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChannelStatisticsSet {
    /// Red channel.
    pub red: ChannelStatistics,
    /// Green channel.
    pub green: ChannelStatistics,
    /// Blue channel.
    pub blue: ChannelStatistics,
    /// Alpha channel, when present.
    pub alpha: Option<ChannelStatistics>,
}

/// Statistics for one normalized channel.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChannelStatistics {
    /// Minimum value, normalized to 0..1.
    pub min: f64,
    /// Maximum value, normalized to 0..1.
    pub max: f64,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Median value.
    pub median: f64,
    /// First percentile.
    pub p01: f64,
    /// Fifth percentile.
    pub p05: f64,
    /// Ninety-fifth percentile.
    pub p95: f64,
    /// Ninety-ninth percentile.
    pub p99: f64,
    /// Shannon entropy, in bits.
    pub entropy_bits: f64,
    /// Ratio of samples near 0.
    pub clipped_low_ratio: f64,
    /// Ratio of samples near 1.
    pub clipped_high_ratio: f64,
}

/// HSV-derived statistics.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HsvStatistics {
    /// Circular mean hue in degrees.
    pub mean_hue_degrees: f64,
    /// Mean saturation.
    pub mean_saturation: f64,
    /// Mean value/brightness.
    pub mean_value: f64,
    /// Saturation channel statistics.
    pub saturation: ChannelStatistics,
    /// Value channel statistics.
    pub value: ChannelStatistics,
}

/// RGB balance and color cast indicators.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorBalance {
    /// Mean red value.
    pub red_mean: f64,
    /// Mean green value.
    pub green_mean: f64,
    /// Mean blue value.
    pub blue_mean: f64,
    /// RGB mean ratios summing to 1.
    pub normalized_rgb: [f64; 3],
    /// Red minus blue mean. Positive is warmer.
    pub red_minus_blue: f64,
    /// Green minus average magenta axis.
    pub green_magenta: f64,
    /// Simple warm/cool score.
    pub warm_cool_score: f64,
    /// Approximate white balance gains to equalize channel means.
    pub white_balance_gains: [f64; 3],
    /// Dominant mean channel.
    pub dominant_channel: String,
}

/// High-level visual tendencies inferred from statistics.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageTendencies {
    /// Exposure label.
    pub exposure: String,
    /// Contrast label.
    pub contrast: String,
    /// Saturation label.
    pub saturation: String,
    /// Temperature label.
    pub temperature: String,
    /// Tint label.
    pub tint: String,
    /// Whether the image appears close to grayscale.
    pub likely_grayscale: bool,
    /// Has significant shadow clipping.
    pub shadow_clipping: bool,
    /// Has significant highlight clipping.
    pub highlight_clipping: bool,
    /// Has a measurable color cast.
    pub color_cast: bool,
    /// Has non-opaque alpha pixels.
    pub has_transparency: bool,
    /// Has a large transparent area.
    pub mostly_transparent: bool,
    /// Has a strongly dominant color channel.
    pub dominant_channel_bias: bool,
    /// Has low dynamic range in luma.
    pub low_dynamic_range: bool,
}

/// Spatial brightness statistics.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialStatistics {
    /// Mean luma in the image center.
    pub center_luma_mean: f64,
    /// Mean luma in the image border.
    pub border_luma_mean: f64,
    /// Center minus border luma.
    pub center_border_delta: f64,
    /// Top half mean luma.
    pub top_luma_mean: f64,
    /// Bottom half mean luma.
    pub bottom_luma_mean: f64,
    /// Left half mean luma.
    pub left_luma_mean: f64,
    /// Right half mean luma.
    pub right_luma_mean: f64,
}

/// Histogram set.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageHistograms {
    /// Number of bins.
    pub bins: usize,
    /// Red histogram.
    pub red: Vec<u64>,
    /// Green histogram.
    pub green: Vec<u64>,
    /// Blue histogram.
    pub blue: Vec<u64>,
    /// Alpha histogram, when present.
    pub alpha: Option<Vec<u64>>,
    /// Luma histogram.
    pub luma: Vec<u64>,
    /// Hue histogram.
    pub hue: Vec<u64>,
    /// Saturation histogram.
    pub saturation: Vec<u64>,
    /// Value histogram.
    pub value: Vec<u64>,
}

/// Computes image statistics for a validated frame.
pub fn image_statistics(
    frame: &FrameView<'_, Validated>,
    options: ImageStatisticsOptions,
) -> Result<ImageStatistics> {
    let dims = frame.dimensions();
    let bins = options.histogram_bins.clamp(1, 256);
    let pixels = dims.pixels()?;
    let mut red = Vec::with_capacity(pixels);
    let mut green = Vec::with_capacity(pixels);
    let mut blue = Vec::with_capacity(pixels);
    let mut alpha = Vec::new();
    let mut luma = Vec::with_capacity(pixels);
    let mut hue = Vec::with_capacity(pixels);
    let mut saturation = Vec::with_capacity(pixels);
    let mut value = Vec::with_capacity(pixels);
    let mut spatial = SpatialAccumulator::default();

    for_each_rgba(frame, |x, y, rgba| {
        let [r, g, b, a] = rgba;
        let y_luma = luma709(r, g, b);
        let (h, s, v) = rgb_to_hsv(r, g, b);
        red.push(r);
        green.push(g);
        blue.push(b);
        if !a.is_nan() {
            alpha.push(a);
        }
        luma.push(y_luma);
        hue.push(h / 360.0);
        saturation.push(s);
        value.push(v);
        spatial.push(x, y, dims.width, dims.height, y_luma);
    })?;

    let red_stats = channel_statistics(red.clone());
    let green_stats = channel_statistics(green.clone());
    let blue_stats = channel_statistics(blue.clone());
    let alpha_stats = (!alpha.is_empty()).then(|| channel_statistics(alpha.clone()));
    let luma_stats = channel_statistics(luma.clone());
    let saturation_stats = channel_statistics(saturation.clone());
    let value_stats = channel_statistics(value.clone());
    let mean_hue_degrees = circular_hue_mean(&hue);
    let color_balance = color_balance(&red_stats, &green_stats, &blue_stats);
    let tendencies = image_tendencies(
        &luma_stats,
        &saturation_stats,
        alpha_stats.as_ref(),
        &color_balance,
    );
    let histograms = ImageHistograms {
        bins,
        red: histogram(&red, bins),
        green: histogram(&green, bins),
        blue: histogram(&blue, bins),
        alpha: (!alpha.is_empty()).then(|| histogram(&alpha, bins)),
        luma: histogram(&luma, bins),
        hue: histogram(&hue, bins),
        saturation: histogram(&saturation, bins),
        value: histogram(&value, bins),
    };

    Ok(ImageStatistics {
        width: dims.width,
        height: dims.height,
        pixels: u64::try_from(pixels).unwrap_or(u64::MAX),
        pixel_format: frame.pixel_format(),
        channels: ChannelStatisticsSet {
            red: red_stats,
            green: green_stats,
            blue: blue_stats,
            alpha: alpha_stats,
        },
        luma: luma_stats,
        hsv: HsvStatistics {
            mean_hue_degrees,
            mean_saturation: saturation_stats.mean,
            mean_value: value_stats.mean,
            saturation: saturation_stats,
            value: value_stats,
        },
        color_balance,
        tendencies,
        spatial: spatial.finish(),
        histograms,
    })
}

fn for_each_rgba(
    frame: &FrameView<'_, Validated>,
    mut f: impl FnMut(u32, u32, [f64; 4]),
) -> Result<()> {
    let dims = frame.dimensions();
    let (width, height) = dims.as_usize()?;
    let plane = frame.plane(0)?;
    match frame.pixel_format() {
        PixelFormat::Rgba8 => each_packed_u8(width, height, plane, 4, |x, y, p| {
            f(x, y, rgba(p[0], p[1], p[2], Some(p[3])))
        }),
        PixelFormat::Rgb8 => each_packed_u8(width, height, plane, 3, |x, y, p| {
            f(x, y, rgba(p[0], p[1], p[2], None))
        }),
        PixelFormat::Bgra8 => each_packed_u8(width, height, plane, 4, |x, y, p| {
            f(x, y, rgba(p[2], p[1], p[0], Some(p[3])))
        }),
        PixelFormat::Bgr8 => each_packed_u8(width, height, plane, 3, |x, y, p| {
            f(x, y, rgba(p[2], p[1], p[0], None))
        }),
        PixelFormat::Luma8 => each_packed_u8(width, height, plane, 1, |x, y, p| {
            f(x, y, rgba(p[0], p[0], p[0], None))
        }),
        other => Err(Error::invalid_frame(format!(
            "image statistics currently require RGB8/RGBA8/BGR8/BGRA8/Luma8, got {other:?}"
        ))),
    }
}

fn each_packed_u8(
    width: usize,
    height: usize,
    plane: crate::frame::PlaneView<'_>,
    channels: usize,
    mut f: impl FnMut(u32, u32, &[u8]),
) -> Result<()> {
    let row_bytes = width
        .checked_mul(channels)
        .ok_or_else(|| Error::invalid_frame("statistics row size overflows usize"))?;
    for y in 0..height {
        let row = plane.row(y, row_bytes)?;
        for x in 0..width {
            let start = x * channels;
            f(x as u32, y as u32, &row[start..start + channels]);
        }
    }
    Ok(())
}

fn rgba(r: u8, g: u8, b: u8, a: Option<u8>) -> [f64; 4] {
    [
        f64::from(r) / 255.0,
        f64::from(g) / 255.0,
        f64::from(b) / 255.0,
        a.map(|v| f64::from(v) / 255.0).unwrap_or(f64::NAN),
    ]
}

fn luma709(r: f64, g: f64, b: f64) -> f64 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let hue = if delta <= f64::EPSILON {
        0.0
    } else if (max - r).abs() <= f64::EPSILON {
        60.0 * ((g - b) / delta).rem_euclid(6.0)
    } else if (max - g).abs() <= f64::EPSILON {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let saturation = if max <= f64::EPSILON {
        0.0
    } else {
        delta / max
    };
    (hue, saturation, max)
}

fn channel_statistics(mut values: Vec<f64>) -> ChannelStatistics {
    values.sort_by(|a, b| a.total_cmp(b));
    let count = values.len().max(1);
    let sum: f64 = values.iter().sum();
    let mean = sum / count as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / count as f64;
    let clipped_low = values.iter().filter(|value| **value <= 1.0 / 255.0).count();
    let clipped_high = values
        .iter()
        .filter(|value| **value >= 254.0 / 255.0)
        .count();
    ChannelStatistics {
        min: *values.first().unwrap_or(&0.0),
        max: *values.last().unwrap_or(&0.0),
        mean,
        std_dev: variance.sqrt(),
        median: percentile_sorted(&values, 0.50),
        p01: percentile_sorted(&values, 0.01),
        p05: percentile_sorted(&values, 0.05),
        p95: percentile_sorted(&values, 0.95),
        p99: percentile_sorted(&values, 0.99),
        entropy_bits: entropy_bits(&values, 256),
        clipped_low_ratio: clipped_low as f64 / count as f64,
        clipped_high_ratio: clipped_high as f64 / count as f64,
    }
}

fn percentile_sorted(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f64 * p.clamp(0.0, 1.0)).round() as usize;
    values[index]
}

fn entropy_bits(values: &[f64], bins: usize) -> f64 {
    let counts = histogram(values, bins);
    let total = values.len().max(1) as f64;
    counts
        .into_iter()
        .filter(|count| *count > 0)
        .map(|count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum()
}

fn histogram(values: &[f64], bins: usize) -> Vec<u64> {
    let bins = bins.max(1);
    let mut counts = vec![0; bins];
    values.iter().for_each(|value| {
        let index = (value.clamp(0.0, 1.0) * bins as f64).floor() as usize;
        counts[index.min(bins - 1)] += 1;
    });
    counts
}

fn circular_hue_mean(hue_normalized: &[f64]) -> f64 {
    if hue_normalized.is_empty() {
        return 0.0;
    }
    let (sin_sum, cos_sum) = hue_normalized.iter().fold((0.0, 0.0), |(s, c), hue| {
        let radians = hue * std::f64::consts::TAU;
        (s + radians.sin(), c + radians.cos())
    });
    sin_sum.atan2(cos_sum).to_degrees().rem_euclid(360.0)
}

fn color_balance(
    red: &ChannelStatistics,
    green: &ChannelStatistics,
    blue: &ChannelStatistics,
) -> ColorBalance {
    let sum = (red.mean + green.mean + blue.mean).max(f64::EPSILON);
    let normalized_rgb = [red.mean / sum, green.mean / sum, blue.mean / sum];
    let gray = sum / 3.0;
    let gains = [
        gray / red.mean.max(1.0 / 255.0),
        gray / green.mean.max(1.0 / 255.0),
        gray / blue.mean.max(1.0 / 255.0),
    ];
    let means = [
        ("red", red.mean),
        ("green", green.mean),
        ("blue", blue.mean),
    ];
    let dominant_channel = means
        .into_iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| "none".to_string());
    ColorBalance {
        red_mean: red.mean,
        green_mean: green.mean,
        blue_mean: blue.mean,
        normalized_rgb,
        red_minus_blue: red.mean - blue.mean,
        green_magenta: green.mean - (red.mean + blue.mean) * 0.5,
        warm_cool_score: red.mean - blue.mean + 0.25 * (green.mean - blue.mean),
        white_balance_gains: gains,
        dominant_channel,
    }
}

fn image_tendencies(
    luma: &ChannelStatistics,
    saturation: &ChannelStatistics,
    alpha: Option<&ChannelStatistics>,
    balance: &ColorBalance,
) -> ImageTendencies {
    ImageTendencies {
        exposure: if luma.mean < 0.25 {
            "dark".to_string()
        } else if luma.mean > 0.75 {
            "bright".to_string()
        } else {
            "balanced".to_string()
        },
        contrast: if luma.std_dev < 0.08 {
            "low".to_string()
        } else if luma.std_dev > 0.24 {
            "high".to_string()
        } else {
            "medium".to_string()
        },
        saturation: if saturation.mean < 0.10 {
            "muted".to_string()
        } else if saturation.mean > 0.55 {
            "vivid".to_string()
        } else {
            "moderate".to_string()
        },
        temperature: if balance.red_minus_blue > 0.05 {
            "warm".to_string()
        } else if balance.red_minus_blue < -0.05 {
            "cool".to_string()
        } else {
            "neutral".to_string()
        },
        tint: if balance.green_magenta > 0.04 {
            "green".to_string()
        } else if balance.green_magenta < -0.04 {
            "magenta".to_string()
        } else {
            "neutral".to_string()
        },
        likely_grayscale: saturation.mean < 0.03,
        shadow_clipping: luma.clipped_low_ratio > 0.01,
        highlight_clipping: luma.clipped_high_ratio > 0.01,
        color_cast: balance
            .red_minus_blue
            .abs()
            .max(balance.green_magenta.abs())
            > 0.06,
        has_transparency: alpha.is_some_and(|alpha| alpha.min < 1.0),
        mostly_transparent: alpha.is_some_and(|alpha| alpha.mean < 0.5),
        dominant_channel_bias: balance
            .normalized_rgb
            .iter()
            .any(|channel| (*channel - 1.0 / 3.0).abs() > 0.08),
        low_dynamic_range: (luma.p95 - luma.p05) < 0.20,
    }
}

#[derive(Default)]
struct SpatialAccumulator {
    center: RunningMean,
    border: RunningMean,
    top: RunningMean,
    bottom: RunningMean,
    left: RunningMean,
    right: RunningMean,
}

impl SpatialAccumulator {
    fn push(&mut self, x: u32, y: u32, width: u32, height: u32, luma: f64) {
        let in_center =
            x >= width / 4 && x < width * 3 / 4 && y >= height / 4 && y < height * 3 / 4;
        if in_center {
            self.center.push(luma);
        } else {
            self.border.push(luma);
        }
        if y < height / 2 {
            self.top.push(luma);
        } else {
            self.bottom.push(luma);
        }
        if x < width / 2 {
            self.left.push(luma);
        } else {
            self.right.push(luma);
        }
    }

    fn finish(self) -> SpatialStatistics {
        let center = self.center.mean();
        let border = self.border.mean();
        SpatialStatistics {
            center_luma_mean: center,
            border_luma_mean: border,
            center_border_delta: center - border,
            top_luma_mean: self.top.mean(),
            bottom_luma_mean: self.bottom.mean(),
            left_luma_mean: self.left.mean(),
            right_luma_mean: self.right.mean(),
        }
    }
}

#[derive(Default)]
struct RunningMean {
    sum: f64,
    count: u64,
}

impl RunningMean {
    fn push(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::image_crate;

    #[test]
    fn detects_red_color_balance() {
        let image = image::RgbImage::from_pixel(4, 4, image::Rgb([255, 0, 0]));
        let frame = image_crate::from_rgb_image(image).unwrap();
        let stats = image_statistics(&frame.as_view(), ImageStatisticsOptions::default()).unwrap();
        assert_eq!(stats.color_balance.dominant_channel, "red");
        assert_eq!(stats.tendencies.temperature, "warm");
        assert!(stats.tendencies.color_cast);
    }
}
