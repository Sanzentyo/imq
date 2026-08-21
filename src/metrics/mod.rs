//! Quality metrics and Sans-I/O comparison engine.

use crate::frame::{ColorSpace, Dimensions, FrameView, PixelFormat, Validated};
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::thread;

const PARALLEL_ERROR_MIN_PIXELS: usize = 512 * 512;
const ERROR_HISTOGRAM_BINS: usize = 4096;

mod basic;
mod ssim;

pub use basic::{Mae, MaxAbsoluteError, Mse, Psnr, Rmse};
pub use ssim::{MsSsim, Ssim, WindowedSsim};

/// Whether a higher value means better quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Direction {
    /// Higher scores are better.
    HigherIsBetter,
    /// Lower scores are better.
    LowerIsBetter,
    /// No universal ordering.
    Neutral,
}

/// Which samples should be compared.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SampleDomain {
    /// Compare perceptual luma only.
    #[default]
    Luma,
    /// Compare RGB/color channels only; alpha is ignored.
    Color,
    /// Compare every stored component, including alpha/chroma planes.
    All,
    /// Compare a specific plane by raw sample codes.
    Plane(usize),
    /// Compare derived render-parity samples with a pixel mask.
    Render(RenderDomain),
}

/// Channels compared by render-parity domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RenderChannels {
    /// RGB channels.
    Rgb,
    /// RGBA channels.
    Rgba,
    /// Perceptual grayscale/luma.
    Gray,
    /// Binary mask values.
    Binary,
    /// HSV channels, using circular hue distance.
    Hsv,
    /// HSVA channels, using circular hue distance plus alpha.
    Hsva,
}

/// Pixel inclusion mask for render-parity domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RenderMask {
    /// Include all pixels.
    All,
    /// Include pixels where either image has alpha > 0.
    Visible,
    /// Include pixels where both images have alpha == 1.
    Opaque,
    /// Include pixels where either image has non-black RGB.
    NonBlack,
}

/// A render-parity comparison domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RenderDomain {
    /// Channels to compare.
    pub channels: RenderChannels,
    /// Pixel mask to apply.
    pub mask: RenderMask,
    /// Border/edge radius to exclude. `0` disables interior filtering.
    pub interior_radius: usize,
}

/// One metric result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetricOutput {
    /// Stable metric name.
    pub name: String,
    /// Numeric score. Some metrics may be `f64::INFINITY` for perfect matches.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::value"))]
    pub score: f64,
    /// Score unit, e.g. `dB`, `unitless`, `normalized_code`.
    pub unit: String,
    /// Direction of quality.
    pub direction: Direction,
    /// Extra metric-specific values.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_f64::map"))]
    pub details: BTreeMap<String, f64>,
}

/// Alpha bucket counts for a frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AlphaBucketCounts {
    /// Pixels with alpha == 0.
    pub transparent: u64,
    /// Pixels with alpha == 1.0.
    pub opaque: u64,
    /// Pixels with 0 < alpha < 1.0.
    pub partial: u64,
}

/// Alpha comparison diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AlphaDiagnostics {
    /// Reference/expected alpha buckets.
    pub reference: AlphaBucketCounts,
    /// Distorted/actual alpha buckets.
    pub distorted: AlphaBucketCounts,
    /// Pixels with any alpha mismatch.
    pub mismatch_count: u64,
    /// Maximum alpha delta in 8-bit code units.
    pub max_delta: u8,
    /// Alpha mismatches with a delta greater than one 8-bit LSB.
    pub mismatches_beyond_one_lsb: u64,
}

impl MetricOutput {
    /// Creates a metric output.
    pub fn new(
        name: impl Into<String>,
        score: f64,
        unit: impl Into<String>,
        direction: Direction,
    ) -> Self {
        Self {
            name: name.into(),
            score,
            unit: unit.into(),
            direction,
            details: BTreeMap::new(),
        }
    }

    /// Adds a detail value.
    pub fn with_detail(mut self, key: impl Into<String>, value: f64) -> Self {
        self.details.insert(key.into(), value);
        self
    }

    fn with_error_details(mut self, stats: &ErrorStats, domain: SampleDomain) -> Self {
        self.details
            .insert("samples".to_string(), stats.count as f64);
        self.details
            .insert("channel_sample_count".to_string(), stats.count as f64);
        self.details
            .insert("pixel_count".to_string(), stats.pixel_count as f64);
        self.details.insert("mae".to_string(), stats.mae());
        self.details.insert("mse".to_string(), stats.mse());
        self.details.insert("rmse".to_string(), stats.rmse());
        self.details
            .insert("mean_error".to_string(), stats.mean_error());
        self.details
            .insert("max_channel_delta".to_string(), stats.max_abs);
        if let Some(location) = stats.max_error_location {
            self.details
                .insert("max_error_x".to_string(), location.x as f64);
            self.details
                .insert("max_error_y".to_string(), location.y as f64);
            self.details
                .insert("max_error_channel".to_string(), location.channel as f64);
        }
        self.details
            .insert("max_pixel_delta".to_string(), stats.max_pixel_delta);
        self.details
            .insert("mse_code".to_string(), stats.mse() * 255.0 * 255.0);
        self.details
            .insert("rmse_code".to_string(), stats.rmse() * 255.0);
        self.details
            .insert("mae_code".to_string(), stats.mae() * 255.0);
        self.details
            .insert("mean_error_code".to_string(), stats.mean_error() * 255.0);
        self.details
            .insert("max_channel_delta_code".to_string(), stats.max_abs * 255.0);
        self.details.insert(
            "max_pixel_delta_code".to_string(),
            stats.max_pixel_delta * 255.0,
        );
        self.details
            .insert("changed_samples".to_string(), stats.changed_samples as f64);
        self.details.insert(
            "changed_sample_ratio".to_string(),
            stats.changed_samples as f64 / stats.count as f64,
        );
        if stats.pixel_observations == stats.pixel_count {
            self.details
                .insert("changed_pixels".to_string(), stats.changed_pixels as f64);
            self.details.insert(
                "changed_pixel_ratio".to_string(),
                stats.changed_pixels as f64 / stats.pixel_count as f64,
            );
            for (index, threshold) in [1u8, 2, 4, 8].into_iter().enumerate() {
                let count = stats.pixels_beyond_lsb[index];
                self.details
                    .insert(format!("pixels_beyond_{threshold}_lsb"), count as f64);
                self.details.insert(
                    format!("pixel_ratio_beyond_{threshold}_lsb"),
                    count as f64 / stats.pixel_count as f64,
                );
            }
        }
        for (index, threshold) in [1u8, 2, 4, 8].into_iter().enumerate() {
            let count = stats.samples_beyond_lsb[index];
            self.details
                .insert(format!("samples_beyond_{threshold}_lsb"), count as f64);
            self.details.insert(
                format!("sample_ratio_beyond_{threshold}_lsb"),
                count as f64 / stats.count as f64,
            );
        }
        self.details.insert(
            "error_histogram_bins".to_string(),
            ERROR_HISTOGRAM_BINS as f64,
        );
        for (label, percentile) in [("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)] {
            let value = stats.absolute_error_percentile(percentile);
            self.details.insert(format!("abs_error_{label}"), value);
            self.details
                .insert(format!("abs_error_{label}_code"), value * 255.0);
        }
        self.details
            .insert("channel_count".to_string(), stats.channel_count as f64);
        for channel in 0..stats.channel_count {
            let channel_stats = stats.channels[channel];
            let channel_histogram = &stats.channel_histograms[channel];
            let generic = format!("channel_{channel}");
            add_channel_error_details(
                &mut self.details,
                &generic,
                channel_stats,
                channel_histogram,
            );
            if let Some(label) =
                stats.channel_labels[channel].or_else(|| semantic_channel_label(domain, channel))
            {
                add_channel_error_details(
                    &mut self.details,
                    label,
                    channel_stats,
                    channel_histogram,
                );
            }
        }
        self
    }
}

/// Computes alpha diagnostics. Alpha-less formats are treated as fully opaque.
pub fn alpha_diagnostics(
    reference: &FrameView<'_, Validated>,
    distorted: &FrameView<'_, Validated>,
) -> Result<AlphaDiagnostics> {
    let dims = ensure_same_dimensions(reference, distorted)?;
    let (w, h) = dims.as_usize()?;
    let mut diagnostics = AlphaDiagnostics::default();
    for y in 0..h {
        for x in 0..w {
            let a = alpha_code(reference, x, y)?;
            let b = alpha_code(distorted, x, y)?;
            push_alpha_bucket(&mut diagnostics.reference, a);
            push_alpha_bucket(&mut diagnostics.distorted, b);
            let delta = a.abs_diff(b);
            if delta != 0 {
                diagnostics.mismatch_count += 1;
                diagnostics.max_delta = diagnostics.max_delta.max(delta);
                if delta > 1 {
                    diagnostics.mismatches_beyond_one_lsb += 1;
                }
            }
        }
    }
    Ok(diagnostics)
}

/// Trait implemented by Sans-I/O metrics.
pub trait Metric: Send + Sync {
    /// Stable name.
    fn name(&self) -> &'static str;

    /// Compares `reference` and `distorted`.
    fn compare(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<MetricOutput>;
}

/// Metric selector used by CLI/configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetricSpec {
    /// Metric name: `mse`, `rmse`, `psnr`, `mae`, `maxae`, `ssim`, `wssim`, or `ms-ssim`.
    pub name: String,
    /// Sample domain.
    pub domain: SampleDomain,
}

impl MetricSpec {
    /// Creates a metric spec.
    pub fn new(name: impl Into<String>, domain: SampleDomain) -> Self {
        Self {
            name: name.into(),
            domain,
        }
    }

    /// Parses `name[:domain]`, where domain is `luma`, `color`, `all`, `planeN`,
    /// or a render-parity domain such as `rgb-visible-interior2px`.
    pub fn parse(input: &str) -> Result<Self> {
        let mut parts = input.splitn(2, ':');
        let name = parts.next().unwrap_or_default().trim().to_ascii_lowercase();
        if name.is_empty() {
            return Err(Error::UnknownMetric(input.to_string()));
        }
        let domain = match parts.next().map(str::trim).filter(|s| !s.is_empty()) {
            None => SampleDomain::Luma,
            Some("luma") | Some("y") => SampleDomain::Luma,
            Some("color") | Some("rgb") => SampleDomain::Color,
            Some("all") => SampleDomain::All,
            Some("rgba") => SampleDomain::Render(RenderDomain {
                channels: RenderChannels::Rgba,
                mask: RenderMask::All,
                interior_radius: 0,
            }),
            Some(s) if parse_render_domain(s).is_some() => {
                SampleDomain::Render(parse_render_domain(s).expect("checked is_some"))
            }
            Some(s) if s.starts_with("plane") => {
                let idx = s
                    .trim_start_matches("plane")
                    .parse::<usize>()
                    .map_err(|_| Error::UnsupportedInput(format!("invalid metric domain `{s}`")))?;
                SampleDomain::Plane(idx)
            }
            Some(other) => {
                return Err(Error::UnsupportedInput(format!(
                    "unknown sample domain `{other}`"
                )));
            }
        };
        Ok(Self::new(name, domain))
    }

    /// Stable output key for this metric spec.
    pub fn key(&self) -> String {
        match self.domain_label() {
            Some(domain) => format!("{}:{domain}", self.canonical_name()),
            None => self.canonical_name().to_string(),
        }
    }

    /// Canonical metric name.
    pub fn canonical_name(&self) -> &str {
        match self.name.as_str() {
            "max_ae" | "max-error" => "maxae",
            other => other,
        }
    }

    /// Domain label used in output keys.
    pub fn domain_label(&self) -> Option<String> {
        match self.domain {
            SampleDomain::Luma => None,
            SampleDomain::Color => Some("color".to_string()),
            SampleDomain::All => Some("all".to_string()),
            SampleDomain::Plane(index) => Some(format!("plane{index}")),
            SampleDomain::Render(domain) => Some(domain.label()),
        }
    }

    /// Builds a metric instance from this spec.
    pub fn build(&self) -> Result<Box<dyn Metric>> {
        match self.name.as_str() {
            "mse" => Ok(Box::new(Mse::new(self.domain))),
            "rmse" => Ok(Box::new(Rmse::new(self.domain))),
            "psnr" => Ok(Box::new(Psnr::new(self.domain))),
            "mae" => Ok(Box::new(Mae::new(self.domain))),
            "maxae" | "max_ae" | "max-error" => Ok(Box::new(MaxAbsoluteError::new(self.domain))),
            "ssim" | "global-ssim" | "global_ssim" => Ok(Box::new(Ssim::new())),
            "wssim" | "windowed-ssim" | "windowed_ssim" | "ssim-windowed" | "ssim_windowed" => {
                Ok(Box::new(WindowedSsim::new()))
            }
            "ms-ssim" | "ms_ssim" | "msssim" => Ok(Box::new(MsSsim::new())),
            other => Err(Error::UnknownMetric(other.to_string())),
        }
    }
}

impl RenderDomain {
    fn label(self) -> String {
        let channels = match self.channels {
            RenderChannels::Rgb => "rgb",
            RenderChannels::Rgba => "rgba",
            RenderChannels::Gray => "gray",
            RenderChannels::Binary => "binary",
            RenderChannels::Hsv => "hsv",
            RenderChannels::Hsva => "hsva",
        };
        let mask = match (self.mask, self.interior_radius) {
            (RenderMask::All, 0) => {
                return if matches!(self.channels, RenderChannels::Rgb) {
                    "rgb-all".to_string()
                } else {
                    channels.to_string()
                };
            }
            (RenderMask::All, _) => "interior",
            (RenderMask::Visible, 0) => "visible",
            (RenderMask::Visible, _) => "visible-interior",
            (RenderMask::Opaque, 0) => "opaque",
            (RenderMask::Opaque, _) => "interior",
            (RenderMask::NonBlack, 0) => "nonblack",
            (RenderMask::NonBlack, _) => "nonblack-interior",
        };
        if self.interior_radius == 0 {
            format!("{channels}-{mask}")
        } else {
            format!("{channels}-{mask}{}px", self.interior_radius)
        }
    }
}

/// A set of metrics to run together.
pub struct MetricSet {
    entries: Vec<MetricEntry>,
}

enum MetricEntry {
    BuiltIn(MetricSpec),
    Custom(Box<dyn Metric>),
}

impl MetricSet {
    /// Creates an empty metric set.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Returns the default metric set: PSNR, SSIM, MSE, MAE, max absolute error.
    pub fn defaults() -> Self {
        Self::from_specs(&[
            MetricSpec::new("psnr", SampleDomain::Luma),
            MetricSpec::new("ssim", SampleDomain::Luma),
            MetricSpec::new("mse", SampleDomain::Luma),
            MetricSpec::new("mae", SampleDomain::Luma),
            MetricSpec::new("maxae", SampleDomain::Luma),
        ])
        .expect("default metric specs are valid")
    }

    /// Builds a set from specs.
    pub fn from_specs(specs: &[MetricSpec]) -> Result<Self> {
        let mut set = Self::new();
        for spec in specs {
            spec.build()?;
            set.entries.push(MetricEntry::BuiltIn(spec.clone()));
        }
        Ok(set)
    }

    /// Parses a comma-separated metric list.
    pub fn from_csv(input: &str) -> Result<Self> {
        let specs = input
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(MetricSpec::parse)
            .collect::<Result<Vec<_>>>()?;
        Self::from_specs(&specs)
    }

    /// Adds one metric.
    pub fn push(&mut self, metric: Box<dyn Metric>) {
        self.entries.push(MetricEntry::Custom(metric));
    }

    /// Runs all metrics.
    pub fn compare(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<Vec<MetricOutput>> {
        let mut outputs = Vec::with_capacity(self.entries.len());
        let mut error_stats = Vec::<(SampleDomain, ErrorStats)>::new();
        for entry in &self.entries {
            let output = match entry {
                MetricEntry::BuiltIn(spec) => {
                    if is_error_metric(spec.name.as_str()) {
                        let stats =
                            stats_for_domain(&mut error_stats, spec.domain, reference, distorted)?;
                        error_metric_output(spec, stats)
                    } else {
                        spec.build()?.compare(reference, distorted)?
                    }
                }
                MetricEntry::Custom(metric) => metric.compare(reference, distorted)?,
            };
            outputs.push(output);
        }
        Ok(outputs)
    }

    /// Returns `true` if no metrics are configured.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for MetricSet {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Compares two frames with the default metric set.
pub fn compare_with_defaults(
    reference: &FrameView<'_, Validated>,
    distorted: &FrameView<'_, Validated>,
) -> Result<Vec<MetricOutput>> {
    MetricSet::defaults().compare(reference, distorted)
}

pub(crate) fn ensure_same_dimensions(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
) -> Result<Dimensions> {
    if a.dimensions() != b.dimensions() {
        return Err(Error::incompatible(format!(
            "dimensions differ: {:?} vs {:?}",
            a.dimensions(),
            b.dimensions()
        )));
    }
    Ok(a.dimensions())
}

pub(crate) fn for_each_sample_pair<F>(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
    domain: SampleDomain,
    mut f: F,
) -> Result<usize>
where
    F: FnMut(f64, f64),
{
    let dims = ensure_same_dimensions(a, b)?;
    let (w, h) = dims.as_usize()?;
    let mut count = 0usize;

    match domain {
        SampleDomain::Luma => {
            for y in 0..h {
                for x in 0..w {
                    f(read_luma(a, x, y)?, read_luma(b, x, y)?);
                    count += 1;
                }
            }
        }
        SampleDomain::Color => {
            for y in 0..h {
                for x in 0..w {
                    let ca = read_rgb(a, x, y)?;
                    let cb = read_rgb(b, x, y)?;
                    for c in 0..3 {
                        f(ca[c], cb[c]);
                        count += 1;
                    }
                }
            }
        }
        SampleDomain::All => {
            if a.pixel_format().is_yuv() || b.pixel_format().is_yuv() {
                if a.pixel_format() != b.pixel_format() {
                    return Err(Error::incompatible(
                        "SampleDomain::All requires identical YUV pixel formats",
                    ));
                }
                count = for_each_yuv_storage_sample_pair(a, b, &mut f)?;
            } else {
                let channels_a = stored_channel_count(a.pixel_format());
                let channels_b = stored_channel_count(b.pixel_format());
                if channels_a != channels_b {
                    return Err(Error::incompatible(format!(
                        "channel counts differ for all-channel compare: {channels_a} vs {channels_b}"
                    )));
                }
                for y in 0..h {
                    for x in 0..w {
                        let ca = read_stored_channels(a, x, y)?;
                        let cb = read_stored_channels(b, x, y)?;
                        for c in 0..channels_a {
                            f(ca[c], cb[c]);
                            count += 1;
                        }
                    }
                }
            }
        }
        SampleDomain::Plane(index) => {
            count = for_each_plane_sample_pair(a, b, index, &mut f)?;
        }
        SampleDomain::Render(domain) => {
            count = for_each_render_sample_pair(a, b, domain, |sample| {
                f(sample.reference, sample.distorted);
            })?;
        }
    }

    if count == 0 {
        Err(Error::unsupported(
            "metric received zero comparable samples",
        ))
    } else {
        Ok(count)
    }
}

pub(crate) fn aggregate_error(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
    domain: SampleDomain,
) -> Result<ErrorStats> {
    if let Some(stats) = aggregate_error_parallel(a, b, domain)? {
        return Ok(stats);
    }
    aggregate_error_rows(a, b, domain, 0, None)
}

fn aggregate_error_rows(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
    domain: SampleDomain,
    start_y: usize,
    end_y: Option<usize>,
) -> Result<ErrorStats> {
    let dims = ensure_same_dimensions(a, b)?;
    let (w, h) = dims.as_usize()?;
    let end_y = end_y.unwrap_or(h).min(h);
    let mut stats = ErrorStats {
        channel_labels: error_channel_labels(domain, a.pixel_format(), b.pixel_format()),
        ..ErrorStats::default()
    };

    match domain {
        SampleDomain::Luma => {
            for y in start_y..end_y {
                for x in 0..w {
                    let x_value = read_luma(a, x, y)?;
                    let y_value = read_luma(b, x, y)?;
                    let delta = x_value - y_value;
                    stats.push_sample(delta, Some(0), Some(SampleLocation { x, y, channel: 0 }));
                    stats.push_pixel(delta.abs());
                    stats.max_pixel_delta = stats.max_pixel_delta.max(delta.abs());
                }
            }
            stats.pixel_count = (end_y - start_y) * w;
        }
        SampleDomain::Color => {
            for y in start_y..end_y {
                for x in 0..w {
                    let ca = read_rgb(a, x, y)?;
                    let cb = read_rgb(b, x, y)?;
                    let mut pixel_sum_sq = 0.0;
                    let mut pixel_max = 0.0_f64;
                    for c in 0..3 {
                        let delta = ca[c] - cb[c];
                        stats.push_sample(
                            delta,
                            Some(c),
                            Some(SampleLocation { x, y, channel: c }),
                        );
                        pixel_sum_sq += delta * delta;
                        pixel_max = pixel_max.max(delta.abs());
                    }
                    stats.push_pixel(pixel_max);
                    stats.max_pixel_delta = stats.max_pixel_delta.max(pixel_sum_sq.sqrt());
                }
            }
            stats.pixel_count = (end_y - start_y) * w;
        }
        SampleDomain::All if !a.pixel_format().is_yuv() && !b.pixel_format().is_yuv() => {
            let channels_a = stored_channel_count(a.pixel_format());
            let channels_b = stored_channel_count(b.pixel_format());
            if channels_a != channels_b {
                return Err(Error::incompatible(format!(
                    "channel counts differ for all-channel compare: {channels_a} vs {channels_b}"
                )));
            }
            for y in start_y..end_y {
                for x in 0..w {
                    let ca = read_stored_channels(a, x, y)?;
                    let cb = read_stored_channels(b, x, y)?;
                    let mut pixel_sum_sq = 0.0;
                    let mut pixel_max = 0.0_f64;
                    for c in 0..channels_a {
                        let delta = ca[c] - cb[c];
                        stats.push_sample(
                            delta,
                            Some(c),
                            Some(SampleLocation { x, y, channel: c }),
                        );
                        pixel_sum_sq += delta * delta;
                        pixel_max = pixel_max.max(delta.abs());
                    }
                    stats.push_pixel(pixel_max);
                    stats.max_pixel_delta = stats.max_pixel_delta.max(pixel_sum_sq.sqrt());
                }
            }
            stats.pixel_count = (end_y - start_y) * w;
        }
        SampleDomain::All => {
            if start_y != 0 || end_y != h {
                return Err(Error::unsupported(
                    "row-range aggregation is unsupported for stored YUV samples",
                ));
            }
            return aggregate_yuv_storage_error(a, b);
        }
        SampleDomain::Render(domain) => {
            return aggregate_render_error_rows(a, b, domain, start_y, Some(end_y));
        }
        _ => {
            if start_y != 0 || end_y != h {
                return Err(Error::unsupported(
                    "row-range aggregation is unsupported for this sample domain",
                ));
            }
            for_each_sample_pair(a, b, domain, |x, y| {
                let delta = x - y;
                stats.push_sample(
                    delta,
                    matches!(domain, SampleDomain::Plane(_)).then_some(0),
                    None,
                );
                stats.push_pixel(delta.abs());
                stats.max_pixel_delta = stats.max_pixel_delta.max(delta.abs());
            })?;
            stats.pixel_count = match domain {
                SampleDomain::All => dims.pixels().unwrap_or(stats.count),
                _ => stats.count,
            };
        }
    }

    if stats.count == 0 {
        return Err(Error::unsupported(
            "metric received zero comparable samples",
        ));
    }
    Ok(stats)
}

fn aggregate_yuv_storage_error(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
) -> Result<ErrorStats> {
    ensure_same_dimensions(a, b)?;
    if a.pixel_format() != b.pixel_format() || !a.pixel_format().is_yuv() {
        return Err(Error::incompatible(
            "SampleDomain::All requires identical YUV pixel formats",
        ));
    }

    let format = a.pixel_format();
    let mut stats = ErrorStats {
        channel_labels: [Some("y"), Some("u"), Some("v"), None],
        ..ErrorStats::default()
    };
    for plane_index in 0..format.plane_count() {
        let plane_dimensions = format.plane_dimensions(a.dimensions(), plane_index)?;
        let (row_bytes, height) = plane_dimensions.as_usize()?;
        let reference_plane = a.plane(plane_index)?;
        let distorted_plane = b.plane(plane_index)?;
        for y in 0..height {
            let reference_row = reference_plane.row(y, row_bytes)?;
            let distorted_row = distorted_plane.row(y, row_bytes)?;
            for x in 0..row_bytes {
                let channel = if format == PixelFormat::Nv12 && plane_index == 1 {
                    1 + x % 2
                } else {
                    plane_index
                };
                let delta = (reference_row[x] as f64 - distorted_row[x] as f64) / 255.0;
                stats.push_sample(delta, Some(channel), None);
            }
        }
    }
    stats.pixel_count = a.dimensions().pixels()?;
    if stats.count == 0 {
        return Err(Error::unsupported(
            "metric received zero comparable samples",
        ));
    }
    stats.max_pixel_delta = stats.max_abs;
    Ok(stats)
}

fn aggregate_error_parallel(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
    domain: SampleDomain,
) -> Result<Option<ErrorStats>> {
    let dims = ensure_same_dimensions(a, b)?;
    if !can_parallelize_error_domain(a, b, domain) || dims.pixels()? < PARALLEL_ERROR_MIN_PIXELS {
        return Ok(None);
    }
    let (_, h) = dims.as_usize()?;
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(h);
    if workers <= 1 {
        return Ok(None);
    }
    let rows_per_worker = h.div_ceil(workers);
    let partials = thread::scope(|scope| {
        let handles = (0..workers)
            .filter_map(|worker| {
                let start_y = worker * rows_per_worker;
                let end_y = ((worker + 1) * rows_per_worker).min(h);
                (start_y < end_y).then(|| {
                    scope.spawn(move || aggregate_error_rows(a, b, domain, start_y, Some(end_y)))
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| Error::unsupported("metric worker panicked"))?
            })
            .collect::<Result<Vec<_>>>()
    })?;
    Ok(Some(combine_error_stats(partials)))
}

fn can_parallelize_error_domain(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
    domain: SampleDomain,
) -> bool {
    matches!(
        domain,
        SampleDomain::Luma | SampleDomain::Color | SampleDomain::Render(_)
    ) || (matches!(domain, SampleDomain::All)
        && !a.pixel_format().is_yuv()
        && !b.pixel_format().is_yuv())
}

fn combine_error_stats(partials: Vec<ErrorStats>) -> ErrorStats {
    partials
        .into_iter()
        .fold(ErrorStats::default(), |mut acc, stats| {
            acc.count += stats.count;
            acc.pixel_count += stats.pixel_count;
            acc.sum_sq += stats.sum_sq;
            acc.sum_abs += stats.sum_abs;
            acc.sum_signed += stats.sum_signed;
            if stats.max_abs > acc.max_abs {
                acc.max_error_location = stats.max_error_location;
            }
            acc.max_abs = acc.max_abs.max(stats.max_abs);
            acc.max_pixel_delta = acc.max_pixel_delta.max(stats.max_pixel_delta);
            acc.changed_samples += stats.changed_samples;
            acc.pixel_observations += stats.pixel_observations;
            acc.changed_pixels += stats.changed_pixels;
            for (target, source) in acc
                .samples_beyond_lsb
                .iter_mut()
                .zip(stats.samples_beyond_lsb)
            {
                *target += source;
            }
            for (target, source) in acc
                .pixels_beyond_lsb
                .iter_mut()
                .zip(stats.pixels_beyond_lsb)
            {
                *target += source;
            }
            acc.channel_count = acc.channel_count.max(stats.channel_count);
            if acc.channel_labels.iter().all(Option::is_none) {
                acc.channel_labels = stats.channel_labels;
            }
            for channel in 0..stats.channel_count {
                acc.channels[channel].merge(stats.channels[channel]);
                for (target, source) in acc.channel_histograms[channel]
                    .iter_mut()
                    .zip(stats.channel_histograms[channel].iter())
                {
                    *target += source;
                }
            }
            for (target, source) in acc.histogram.iter_mut().zip(stats.histogram.iter()) {
                *target += source;
            }
            acc
        })
}

fn stats_for_domain<'a>(
    cached: &'a mut Vec<(SampleDomain, ErrorStats)>,
    domain: SampleDomain,
    reference: &FrameView<'_, Validated>,
    distorted: &FrameView<'_, Validated>,
) -> Result<&'a ErrorStats> {
    if let Some(index) = cached
        .iter()
        .position(|(cached_domain, _)| *cached_domain == domain)
    {
        return Ok(&cached[index].1);
    }
    let stats = aggregate_error(reference, distorted, domain)?;
    cached.push((domain, stats));
    Ok(&cached.last().expect("pushed stats").1)
}

fn is_error_metric(name: &str) -> bool {
    matches!(
        name,
        "mse" | "rmse" | "psnr" | "mae" | "maxae" | "max_ae" | "max-error"
    )
}

fn parse_render_domain(input: &str) -> Option<RenderDomain> {
    let (channel_name, suffix) = input.split_once('-').unwrap_or((input, "all"));
    let channels = match channel_name {
        "rgb" => RenderChannels::Rgb,
        "rgba" => RenderChannels::Rgba,
        "gray" | "grey" | "luma" => RenderChannels::Gray,
        "binary" | "mask" | "bin" => RenderChannels::Binary,
        "hsv" => RenderChannels::Hsv,
        "hsva" => RenderChannels::Hsva,
        _ => return None,
    };
    let (mask, interior_radius) = parse_render_mask_suffix(suffix)?;
    Some(RenderDomain {
        channels,
        mask,
        interior_radius,
    })
}

fn parse_render_mask_suffix(suffix: &str) -> Option<(RenderMask, usize)> {
    match suffix {
        "all" => Some((RenderMask::All, 0)),
        "visible" => Some((RenderMask::Visible, 0)),
        "opaque" => Some((RenderMask::Opaque, 0)),
        "nonblack" | "non-black" => Some((RenderMask::NonBlack, 0)),
        _ => {
            if let Some(radius) = parse_interior_suffix(suffix) {
                return Some((RenderMask::Opaque, radius));
            }
            if let Some(rest) = suffix.strip_prefix("visible-") {
                return parse_interior_suffix(rest).map(|radius| (RenderMask::Visible, radius));
            }
            if let Some(rest) = suffix.strip_prefix("opaque-") {
                return parse_interior_suffix(rest).map(|radius| (RenderMask::Opaque, radius));
            }
            if let Some(rest) = suffix
                .strip_prefix("nonblack-")
                .or_else(|| suffix.strip_prefix("non-black-"))
            {
                return parse_interior_suffix(rest).map(|radius| (RenderMask::NonBlack, radius));
            }
            None
        }
    }
}

fn parse_interior_suffix(suffix: &str) -> Option<usize> {
    if suffix == "interior" {
        return Some(1);
    }
    suffix
        .strip_prefix("interior")
        .and_then(|rest| rest.strip_suffix("px"))
        .and_then(|digits| digits.parse::<usize>().ok())
        .filter(|radius| *radius > 0)
}

fn error_metric_output(spec: &MetricSpec, stats: &ErrorStats) -> MetricOutput {
    let name = spec.key();
    let peak = 1.0;
    match spec.name.as_str() {
        "mse" => MetricOutput::new(
            name,
            stats.mse(),
            "normalized_code^2",
            Direction::LowerIsBetter,
        )
        .with_error_details(stats, spec.domain),
        "rmse" => MetricOutput::new(
            name,
            stats.rmse(),
            "normalized_code",
            Direction::LowerIsBetter,
        )
        .with_error_details(stats, spec.domain),
        "psnr" => {
            let mse = stats.mse();
            let psnr = if mse == 0.0 {
                f64::INFINITY
            } else {
                10.0 * ((peak * peak) / mse).log10()
            };
            MetricOutput::new(name, psnr, "dB", Direction::HigherIsBetter)
                .with_detail("mse", mse)
                .with_error_details(stats, spec.domain)
        }
        "mae" => MetricOutput::new(
            name,
            stats.mae(),
            "normalized_code",
            Direction::LowerIsBetter,
        )
        .with_error_details(stats, spec.domain),
        "maxae" | "max_ae" | "max-error" => MetricOutput::new(
            name,
            stats.max_abs,
            "normalized_code",
            Direction::LowerIsBetter,
        )
        .with_error_details(stats, spec.domain),
        _ => unreachable!("error metric names are filtered before output"),
    }
}

#[derive(Debug, Clone, Copy)]
struct RenderSample {
    reference: f64,
    distorted: f64,
}

fn aggregate_render_error_rows(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
    domain: RenderDomain,
    start_y: usize,
    end_y: Option<usize>,
) -> Result<ErrorStats> {
    let dims = ensure_same_dimensions(a, b)?;
    let (w, h) = dims.as_usize()?;
    let end_y = end_y.unwrap_or(h).min(h);
    let mut stats = ErrorStats {
        channel_labels: error_channel_labels(
            SampleDomain::Render(domain),
            a.pixel_format(),
            b.pixel_format(),
        ),
        ..ErrorStats::default()
    };
    for y in start_y..end_y {
        for x in 0..w {
            if !render_pixel_included(a, b, domain, x, y, w, h)? {
                continue;
            }
            let reference = render_values(a, domain.channels, x, y)?;
            let distorted = render_values(b, domain.channels, x, y)?;
            let channels = render_channel_count(domain.channels);
            let mut pixel_sum_sq = 0.0;
            let mut pixel_max = 0.0_f64;
            for c in 0..channels {
                let delta = render_channel_delta(domain.channels, c, reference[c], distorted[c]);
                stats.push_sample(delta, Some(c), Some(SampleLocation { x, y, channel: c }));
                pixel_sum_sq += delta * delta;
                pixel_max = pixel_max.max(delta.abs());
            }
            stats.pixel_count += 1;
            stats.push_pixel(pixel_max);
            stats.max_pixel_delta = stats.max_pixel_delta.max(pixel_sum_sq.sqrt());
        }
    }
    if stats.count == 0 {
        return Err(Error::unsupported(
            "metric received zero comparable samples",
        ));
    }
    Ok(stats)
}

fn for_each_render_sample_pair(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
    domain: RenderDomain,
    mut f: impl FnMut(RenderSample),
) -> Result<usize> {
    let dims = ensure_same_dimensions(a, b)?;
    let (w, h) = dims.as_usize()?;
    let mut count = 0;
    for y in 0..h {
        for x in 0..w {
            if !render_pixel_included(a, b, domain, x, y, w, h)? {
                continue;
            }
            let reference = render_values(a, domain.channels, x, y)?;
            let distorted = render_values(b, domain.channels, x, y)?;
            for c in 0..render_channel_count(domain.channels) {
                f(RenderSample {
                    reference: reference[c],
                    distorted: distorted[c],
                });
                count += 1;
            }
        }
    }
    Ok(count)
}

fn render_pixel_included(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
    domain: RenderDomain,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> Result<bool> {
    if domain.interior_radius == 0 {
        return render_base_pixel_included(a, b, domain.mask, x, y);
    }
    let radius = domain.interior_radius;
    if radius >= w || radius >= h || x < radius || y < radius || x >= w - radius || y >= h - radius
    {
        return Ok(false);
    }
    for ny in (y - radius)..=(y + radius) {
        for nx in (x - radius)..=(x + radius) {
            if !render_interior_pixel_included(a, b, domain.mask, nx, ny)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn render_interior_pixel_included(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
    mask: RenderMask,
    x: usize,
    y: usize,
) -> Result<bool> {
    match mask {
        RenderMask::Visible => Ok(read_alpha(a, x, y)? > 0.0 && read_alpha(b, x, y)? > 0.0),
        _ => render_base_pixel_included(a, b, mask, x, y),
    }
}

fn render_base_pixel_included(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
    mask: RenderMask,
    x: usize,
    y: usize,
) -> Result<bool> {
    match mask {
        RenderMask::All => Ok(true),
        RenderMask::Visible => Ok(read_alpha(a, x, y)? > 0.0 || read_alpha(b, x, y)? > 0.0),
        RenderMask::Opaque => Ok(read_alpha(a, x, y)? >= 1.0 && read_alpha(b, x, y)? >= 1.0),
        RenderMask::NonBlack => {
            let ra = read_rgb(a, x, y)?;
            let rb = read_rgb(b, x, y)?;
            Ok(ra.into_iter().chain(rb).any(|sample| sample > 0.0))
        }
    }
}

fn render_values(
    frame: &FrameView<'_, Validated>,
    channels: RenderChannels,
    x: usize,
    y: usize,
) -> Result<[f64; 4]> {
    match channels {
        RenderChannels::Rgb => {
            let rgb = read_rgb(frame, x, y)?;
            Ok([rgb[0], rgb[1], rgb[2], 0.0])
        }
        RenderChannels::Rgba => {
            let rgb = read_rgb(frame, x, y)?;
            Ok([rgb[0], rgb[1], rgb[2], read_alpha(frame, x, y)?])
        }
        RenderChannels::Gray => Ok([read_luma(frame, x, y)?, 0.0, 0.0, 0.0]),
        RenderChannels::Binary => Ok([read_binary(frame, x, y)?, 0.0, 0.0, 0.0]),
        RenderChannels::Hsv => {
            let hsv = read_hsv(frame, x, y)?;
            Ok([hsv[0], hsv[1], hsv[2], 0.0])
        }
        RenderChannels::Hsva => {
            let hsv = read_hsv(frame, x, y)?;
            Ok([hsv[0], hsv[1], hsv[2], read_alpha(frame, x, y)?])
        }
    }
}

fn render_channel_count(channels: RenderChannels) -> usize {
    match channels {
        RenderChannels::Rgb | RenderChannels::Hsv => 3,
        RenderChannels::Rgba | RenderChannels::Hsva => 4,
        RenderChannels::Gray | RenderChannels::Binary => 1,
    }
}

fn render_channel_delta(
    channels: RenderChannels,
    channel: usize,
    reference: f64,
    distorted: f64,
) -> f64 {
    if matches!(channels, RenderChannels::Hsv | RenderChannels::Hsva) && channel == 0 {
        let direct = reference - distorted;
        if direct.abs() <= 0.5 {
            direct
        } else if direct > 0.0 {
            direct - 1.0
        } else {
            direct + 1.0
        }
    } else {
        reference - distorted
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ChannelErrorStats {
    count: usize,
    sum_sq: f64,
    sum_abs: f64,
    sum_signed: f64,
    max_abs: f64,
}

#[derive(Debug, Clone, Copy)]
struct SampleLocation {
    x: usize,
    y: usize,
    channel: usize,
}

impl ChannelErrorStats {
    fn push(&mut self, delta: f64) {
        let absolute = delta.abs();
        self.count += 1;
        self.sum_sq += delta * delta;
        self.sum_abs += absolute;
        self.sum_signed += delta;
        self.max_abs = self.max_abs.max(absolute);
    }

    fn merge(&mut self, other: Self) {
        self.count += other.count;
        self.sum_sq += other.sum_sq;
        self.sum_abs += other.sum_abs;
        self.sum_signed += other.sum_signed;
        self.max_abs = self.max_abs.max(other.max_abs);
    }

    fn mse(self) -> f64 {
        self.sum_sq / self.count as f64
    }

    fn rmse(self) -> f64 {
        self.mse().sqrt()
    }

    fn mae(self) -> f64 {
        self.sum_abs / self.count as f64
    }

    fn mean_error(self) -> f64 {
        self.sum_signed / self.count as f64
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ErrorStats {
    pub count: usize,
    pub pixel_count: usize,
    pub sum_sq: f64,
    pub sum_abs: f64,
    pub sum_signed: f64,
    pub max_abs: f64,
    pub max_pixel_delta: f64,
    max_error_location: Option<SampleLocation>,
    changed_samples: usize,
    samples_beyond_lsb: [usize; 4],
    pixel_observations: usize,
    changed_pixels: usize,
    pixels_beyond_lsb: [usize; 4],
    channel_count: usize,
    channels: [ChannelErrorStats; 4],
    channel_histograms: [Box<[u64]>; 4],
    channel_labels: [Option<&'static str>; 4],
    histogram: Box<[u64]>,
}

impl Default for ErrorStats {
    fn default() -> Self {
        Self {
            count: 0,
            pixel_count: 0,
            sum_sq: 0.0,
            sum_abs: 0.0,
            sum_signed: 0.0,
            max_abs: 0.0,
            max_pixel_delta: 0.0,
            max_error_location: None,
            changed_samples: 0,
            samples_beyond_lsb: [0; 4],
            pixel_observations: 0,
            changed_pixels: 0,
            pixels_beyond_lsb: [0; 4],
            channel_count: 0,
            channels: [ChannelErrorStats::default(); 4],
            channel_histograms: std::array::from_fn(|_| {
                vec![0; ERROR_HISTOGRAM_BINS + 1].into_boxed_slice()
            }),
            channel_labels: [None; 4],
            histogram: vec![0; ERROR_HISTOGRAM_BINS + 1].into_boxed_slice(),
        }
    }
}

impl ErrorStats {
    fn push_sample(
        &mut self,
        delta: f64,
        channel: Option<usize>,
        location: Option<SampleLocation>,
    ) {
        let absolute = delta.abs();
        self.count += 1;
        self.sum_sq += delta * delta;
        self.sum_abs += absolute;
        self.sum_signed += delta;
        if absolute > self.max_abs {
            self.max_abs = absolute;
            self.max_error_location = location;
        }
        if absolute != 0.0 {
            self.changed_samples += 1;
        }
        for (index, threshold) in [1.0, 2.0, 4.0, 8.0].into_iter().enumerate() {
            if absolute * 255.0 > threshold {
                self.samples_beyond_lsb[index] += 1;
            }
        }
        let histogram_index = if !absolute.is_finite() || absolute >= 1.0 {
            ERROR_HISTOGRAM_BINS
        } else {
            (absolute * ERROR_HISTOGRAM_BINS as f64).round() as usize
        };
        self.histogram[histogram_index] += 1;

        if let Some(channel) = channel.filter(|channel| *channel < self.channels.len()) {
            self.channel_count = self.channel_count.max(channel + 1);
            self.channels[channel].push(delta);
            self.channel_histograms[channel][histogram_index] += 1;
        }
    }

    fn push_pixel(&mut self, max_channel_absolute_error: f64) {
        self.pixel_observations += 1;
        if max_channel_absolute_error != 0.0 {
            self.changed_pixels += 1;
        }
        for (index, threshold) in [1.0, 2.0, 4.0, 8.0].into_iter().enumerate() {
            if max_channel_absolute_error * 255.0 > threshold {
                self.pixels_beyond_lsb[index] += 1;
            }
        }
    }

    pub fn mse(&self) -> f64 {
        self.sum_sq / self.count as f64
    }

    pub fn rmse(&self) -> f64 {
        self.mse().sqrt()
    }

    pub fn mae(&self) -> f64 {
        self.sum_abs / self.count as f64
    }

    fn mean_error(&self) -> f64 {
        self.sum_signed / self.count as f64
    }

    fn absolute_error_percentile(&self, percentile: f64) -> f64 {
        histogram_percentile(&self.histogram, self.count, self.max_abs, percentile)
    }
}

fn add_channel_error_details(
    details: &mut BTreeMap<String, f64>,
    prefix: &str,
    stats: ChannelErrorStats,
    histogram: &[u64],
) {
    details.insert(format!("{prefix}_samples"), stats.count as f64);
    details.insert(format!("{prefix}_mse"), stats.mse());
    details.insert(format!("{prefix}_rmse"), stats.rmse());
    details.insert(format!("{prefix}_mae"), stats.mae());
    details.insert(format!("{prefix}_mean_error"), stats.mean_error());
    details.insert(format!("{prefix}_max_delta"), stats.max_abs);
    let mse = stats.mse();
    details.insert(
        format!("{prefix}_psnr"),
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (1.0 / mse).log10()
        },
    );
    details.insert(format!("{prefix}_mae_code"), stats.mae() * 255.0);
    details.insert(
        format!("{prefix}_mean_error_code"),
        stats.mean_error() * 255.0,
    );
    details.insert(format!("{prefix}_max_delta_code"), stats.max_abs * 255.0);
    for (label, percentile) in [("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)] {
        let value = histogram_percentile(histogram, stats.count, stats.max_abs, percentile);
        details.insert(format!("{prefix}_abs_error_{label}"), value);
        details.insert(format!("{prefix}_abs_error_{label}_code"), value * 255.0);
    }
}

fn histogram_percentile(histogram: &[u64], count: usize, max_abs: f64, percentile: f64) -> f64 {
    let rank = (percentile.clamp(0.0, 1.0) * count as f64).ceil().max(1.0) as u64;
    let mut cumulative = 0u64;
    for (index, bin_count) in histogram.iter().copied().enumerate() {
        cumulative += bin_count;
        if cumulative >= rank {
            return if index == ERROR_HISTOGRAM_BINS && max_abs > 1.0 {
                max_abs
            } else {
                index as f64 / ERROR_HISTOGRAM_BINS as f64
            };
        }
    }
    max_abs
}

fn semantic_channel_label(domain: SampleDomain, channel: usize) -> Option<&'static str> {
    let labels: &[&str] = match domain {
        SampleDomain::Luma => &["luma"],
        SampleDomain::Color => &["red", "green", "blue"],
        SampleDomain::Plane(_) => &["plane"],
        SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Rgb,
            ..
        }) => &["red", "green", "blue"],
        SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Rgba,
            ..
        }) => &["red", "green", "blue", "alpha"],
        SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Gray,
            ..
        }) => &["gray"],
        SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Binary,
            ..
        }) => &["binary"],
        SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Hsv,
            ..
        }) => &["hue", "saturation", "value"],
        SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Hsva,
            ..
        }) => &["hue", "saturation", "value", "alpha"],
        SampleDomain::All => &[],
    };
    labels.get(channel).copied()
}

fn error_channel_labels(
    domain: SampleDomain,
    reference_format: PixelFormat,
    distorted_format: PixelFormat,
) -> [Option<&'static str>; 4] {
    let labels: &[&str] = match domain {
        SampleDomain::Luma => &["luma"],
        SampleDomain::Color => &["red", "green", "blue"],
        SampleDomain::Plane(_) => &["plane"],
        SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Rgb,
            ..
        }) => &["red", "green", "blue"],
        SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Rgba,
            ..
        }) => &["red", "green", "blue", "alpha"],
        SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Gray,
            ..
        }) => &["gray"],
        SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Binary,
            ..
        }) => &["binary"],
        SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Hsv,
            ..
        }) => &["hue", "saturation", "value"],
        SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Hsva,
            ..
        }) => &["hue", "saturation", "value", "alpha"],
        SampleDomain::All if reference_format != distorted_format => &[],
        SampleDomain::All => match reference_format {
            PixelFormat::Luma8 | PixelFormat::Luma16Le => &["luma"],
            PixelFormat::Rgb8 | PixelFormat::Rgb16Le | PixelFormat::RgbF32 => {
                &["red", "green", "blue"]
            }
            PixelFormat::Rgba8 | PixelFormat::Rgba16Le | PixelFormat::RgbaF32 => {
                &["red", "green", "blue", "alpha"]
            }
            PixelFormat::Bgr8 => &["blue", "green", "red"],
            PixelFormat::Bgra8 => &["blue", "green", "red", "alpha"],
            PixelFormat::Hsv8 => &["hue", "saturation", "value"],
            PixelFormat::Hsva8 => &["hue", "saturation", "value", "alpha"],
            PixelFormat::Yuv444p8
            | PixelFormat::Yuv422p8
            | PixelFormat::Yuv420p8
            | PixelFormat::Nv12 => &["y", "u", "v"],
            PixelFormat::Binary1Lsb | PixelFormat::Binary1Msb => &["binary"],
        },
    };
    let mut output = [None; 4];
    for (index, label) in labels.iter().copied().enumerate().take(output.len()) {
        output[index] = Some(label);
    }
    output
}

fn stored_channel_count(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Rgba8
        | PixelFormat::Bgra8
        | PixelFormat::Rgba16Le
        | PixelFormat::RgbaF32
        | PixelFormat::Hsva8 => 4,
        PixelFormat::Rgb8
        | PixelFormat::Bgr8
        | PixelFormat::Rgb16Le
        | PixelFormat::RgbF32
        | PixelFormat::Hsv8 => 3,
        PixelFormat::Luma8
        | PixelFormat::Luma16Le
        | PixelFormat::Binary1Lsb
        | PixelFormat::Binary1Msb => 1,
        PixelFormat::Yuv444p8
        | PixelFormat::Yuv422p8
        | PixelFormat::Yuv420p8
        | PixelFormat::Nv12 => 3,
    }
}

fn read_luma(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<f64> {
    match frame.pixel_format() {
        PixelFormat::Luma8 | PixelFormat::Luma16Le => Ok(read_stored_channels(frame, x, y)?[0]),
        PixelFormat::Binary1Lsb | PixelFormat::Binary1Msb => Ok(read_binary_native(frame, x, y)?),
        PixelFormat::Rgb8
        | PixelFormat::Rgba8
        | PixelFormat::Bgr8
        | PixelFormat::Bgra8
        | PixelFormat::Rgb16Le
        | PixelFormat::Rgba16Le
        | PixelFormat::RgbF32
        | PixelFormat::RgbaF32 => {
            let rgb = read_rgb(frame, x, y)?;
            let (kr, kg, kb) = luma_weights(frame.format().color_space);
            Ok(rgb[0] * kr + rgb[1] * kg + rgb[2] * kb)
        }
        PixelFormat::Hsv8 | PixelFormat::Hsva8 => {
            let rgb = hsv_to_rgb(read_hsv(frame, x, y)?);
            let (kr, kg, kb) = luma_weights(frame.format().color_space);
            Ok(rgb[0] * kr + rgb[1] * kg + rgb[2] * kb)
        }
        PixelFormat::Yuv444p8
        | PixelFormat::Yuv422p8
        | PixelFormat::Yuv420p8
        | PixelFormat::Nv12 => read_yuv(frame, x, y).map(|yuv| yuv[0]),
    }
}

fn read_rgb(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<[f64; 3]> {
    match frame.pixel_format() {
        PixelFormat::Luma8 | PixelFormat::Luma16Le => {
            let l = read_stored_channels(frame, x, y)?[0];
            Ok([l, l, l])
        }
        PixelFormat::Binary1Lsb | PixelFormat::Binary1Msb => {
            let l = read_binary_native(frame, x, y)?;
            Ok([l, l, l])
        }
        PixelFormat::Rgb8
        | PixelFormat::Rgba8
        | PixelFormat::Rgb16Le
        | PixelFormat::Rgba16Le
        | PixelFormat::RgbF32
        | PixelFormat::RgbaF32 => {
            let c = read_stored_channels(frame, x, y)?;
            Ok([c[0], c[1], c[2]])
        }
        PixelFormat::Bgr8 | PixelFormat::Bgra8 => {
            let c = read_stored_channels(frame, x, y)?;
            Ok([c[2], c[1], c[0]])
        }
        PixelFormat::Hsv8 | PixelFormat::Hsva8 => Ok(hsv_to_rgb(read_hsv(frame, x, y)?)),
        PixelFormat::Yuv444p8
        | PixelFormat::Yuv422p8
        | PixelFormat::Yuv420p8
        | PixelFormat::Nv12 => {
            let yuv = read_yuv(frame, x, y)?;
            Ok(yuv_to_rgb(yuv, frame.format().color_space))
        }
    }
}

fn read_alpha(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<f64> {
    match frame.pixel_format() {
        PixelFormat::Rgba8
        | PixelFormat::Bgra8
        | PixelFormat::Rgba16Le
        | PixelFormat::RgbaF32
        | PixelFormat::Hsva8 => Ok(read_stored_channels(frame, x, y)?[3]),
        _ => Ok(1.0),
    }
}

fn alpha_code(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<u8> {
    Ok((read_alpha(frame, x, y)?.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn push_alpha_bucket(buckets: &mut AlphaBucketCounts, alpha: u8) {
    match alpha {
        0 => buckets.transparent += 1,
        255 => buckets.opaque += 1,
        _ => buckets.partial += 1,
    }
}

fn read_binary(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<f64> {
    let alpha = read_alpha(frame, x, y)?;
    if alpha < 1.0 {
        return Ok(if alpha > 0.0 { 1.0 } else { 0.0 });
    }
    Ok(if read_luma(frame, x, y)? >= 0.5 {
        1.0
    } else {
        0.0
    })
}

fn read_hsv(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<[f64; 3]> {
    if matches!(frame.pixel_format(), PixelFormat::Hsv8 | PixelFormat::Hsva8) {
        let c = read_stored_channels(frame, x, y)?;
        return Ok([c[0], c[1], c[2]]);
    }
    let rgb = read_rgb(frame, x, y)?;
    Ok(rgb_to_hsv(rgb[0], rgb[1], rgb[2]))
}

fn rgb_to_hsv(r: f64, g: f64, b: f64) -> [f64; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let hue = if delta == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if max == g {
        (((b - r) / delta) + 2.0) / 6.0
    } else {
        (((r - g) / delta) + 4.0) / 6.0
    };
    let saturation = if max == 0.0 { 0.0 } else { delta / max };
    [hue, saturation, max]
}

fn hsv_to_rgb(hsv: [f64; 3]) -> [f64; 3] {
    let h = hsv[0].rem_euclid(1.0) * 6.0;
    let s = hsv[1].clamp(0.0, 1.0);
    let v = hsv[2].clamp(0.0, 1.0);
    if s == 0.0 {
        return [v, v, v];
    }
    let i = h.floor();
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i as u8 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

fn read_binary_native(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<f64> {
    let plane = frame.plane(0)?;
    let (w, _) = frame.dimensions().as_usize()?;
    let row_bytes = w.div_ceil(8);
    let row = plane.row(y, row_bytes)?;
    let byte = row[x / 8];
    let bit = match frame.pixel_format() {
        PixelFormat::Binary1Lsb => (byte >> (x % 8)) & 1,
        PixelFormat::Binary1Msb => (byte >> (7 - (x % 8))) & 1,
        _ => {
            return Err(Error::unsupported(
                "read_binary_native requires binary format",
            ));
        }
    };
    Ok(f64::from(bit))
}

fn read_stored_channels(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<[f64; 4]> {
    let fmt = frame.pixel_format();
    let plane = frame.plane(0)?;
    let bpp = fmt
        .bytes_per_pixel()
        .ok_or_else(|| Error::unsupported("read_stored_channels requires packed format"))?;
    let row_width = usize::try_from(frame.dimensions().width)
        .map_err(|_| Error::invalid_frame("width overflows usize"))?;
    let row = plane.row(
        y,
        row_width
            .checked_mul(bpp)
            .ok_or_else(|| Error::invalid_frame("row bytes overflow"))?,
    )?;
    let off = x
        .checked_mul(bpp)
        .ok_or_else(|| Error::invalid_frame("pixel offset overflow"))?;
    let max = fmt.max_code_value();
    let out = match fmt {
        PixelFormat::Luma8 => [row[off] as f64 / max, 0.0, 0.0, 1.0],
        PixelFormat::Hsv8 => [
            row[off] as f64 / max,
            row[off + 1] as f64 / max,
            row[off + 2] as f64 / max,
            1.0,
        ],
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => [
            row[off] as f64 / max,
            row[off + 1] as f64 / max,
            row[off + 2] as f64 / max,
            1.0,
        ],
        PixelFormat::Hsva8 => [
            row[off] as f64 / max,
            row[off + 1] as f64 / max,
            row[off + 2] as f64 / max,
            row[off + 3] as f64 / max,
        ],
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => [
            row[off] as f64 / max,
            row[off + 1] as f64 / max,
            row[off + 2] as f64 / max,
            row[off + 3] as f64 / max,
        ],
        PixelFormat::Luma16Le => [read_u16_le(row, off)? as f64 / max, 0.0, 0.0, 1.0],
        PixelFormat::Rgb16Le => [
            read_u16_le(row, off)? as f64 / max,
            read_u16_le(row, off + 2)? as f64 / max,
            read_u16_le(row, off + 4)? as f64 / max,
            1.0,
        ],
        PixelFormat::Rgba16Le => [
            read_u16_le(row, off)? as f64 / max,
            read_u16_le(row, off + 2)? as f64 / max,
            read_u16_le(row, off + 4)? as f64 / max,
            read_u16_le(row, off + 6)? as f64 / max,
        ],
        PixelFormat::RgbF32 => [
            read_f32_ne(row, off)? as f64,
            read_f32_ne(row, off + 4)? as f64,
            read_f32_ne(row, off + 8)? as f64,
            1.0,
        ],
        PixelFormat::RgbaF32 => [
            read_f32_ne(row, off)? as f64,
            read_f32_ne(row, off + 4)? as f64,
            read_f32_ne(row, off + 8)? as f64,
            read_f32_ne(row, off + 12)? as f64,
        ],
        PixelFormat::Binary1Lsb | PixelFormat::Binary1Msb => {
            return Err(Error::unsupported(
                "read_stored_channels cannot read bit-packed binary formats",
            ));
        }
        _ => {
            return Err(Error::unsupported(
                "read_stored_channels cannot read YUV formats",
            ));
        }
    };
    Ok(out)
}

fn read_yuv(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<[f64; 3]> {
    let fmt = frame.pixel_format();
    let y_plane = frame.plane(0)?;
    let y_row_bytes = usize::try_from(frame.dimensions().width)
        .map_err(|_| Error::invalid_frame("width overflows usize"))?;
    let y_row = y_plane.row(y, y_row_bytes)?;
    let yv = y_row[x] as f64 / 255.0;

    let chroma = match fmt {
        PixelFormat::Yuv444p8 => (x, y),
        PixelFormat::Yuv422p8 => (x / 2, y),
        PixelFormat::Yuv420p8 | PixelFormat::Nv12 => (x / 2, y / 2),
        _ => return Err(Error::unsupported("read_yuv requires YUV format")),
    };

    match fmt {
        PixelFormat::Yuv444p8 | PixelFormat::Yuv422p8 | PixelFormat::Yuv420p8 => {
            let u_plane = frame.plane(1)?;
            let v_plane = frame.plane(2)?;
            let u_dims = fmt.plane_dimensions(frame.dimensions(), 1)?;
            let row_bytes = usize::try_from(u_dims.width)
                .map_err(|_| Error::invalid_frame("chroma row width overflows usize"))?;
            let u = u_plane.row(chroma.1, row_bytes)?[chroma.0] as f64 / 255.0;
            let v = v_plane.row(chroma.1, row_bytes)?[chroma.0] as f64 / 255.0;
            Ok([yv, u, v])
        }
        PixelFormat::Nv12 => {
            let uv_plane = frame.plane(1)?;
            let uv_dims = fmt.plane_dimensions(frame.dimensions(), 1)?;
            let row_bytes = usize::try_from(uv_dims.width)
                .map_err(|_| Error::invalid_frame("uv row width overflows usize"))?;
            let row = uv_plane.row(chroma.1, row_bytes)?;
            let off = chroma.0 * 2;
            Ok([yv, row[off] as f64 / 255.0, row[off + 1] as f64 / 255.0])
        }
        _ => unreachable!(),
    }
}

fn yuv_to_rgb(yuv: [f64; 3], cs: ColorSpace) -> [f64; 3] {
    let y = yuv[0];
    let u = yuv[1] - 0.5;
    let v = yuv[2] - 0.5;
    let (rv, gu, gv, bu) = match cs {
        ColorSpace::Bt601 => (1.402, 0.344_136, 0.714_136, 1.772),
        ColorSpace::Bt2020 => (1.4746, 0.164_553, 0.571_353, 1.8814),
        _ => (1.5748, 0.187_324, 0.468_124, 1.8556),
    };
    [
        clamp01(y + rv * v),
        clamp01(y - gu * u - gv * v),
        clamp01(y + bu * u),
    ]
}

fn luma_weights(cs: ColorSpace) -> (f64, f64, f64) {
    match cs {
        ColorSpace::Bt601 => (0.299, 0.587, 0.114),
        ColorSpace::Bt2020 => (0.2627, 0.6780, 0.0593),
        _ => (0.2126, 0.7152, 0.0722),
    }
}

fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

fn read_u16_le(row: &[u8], off: usize) -> Result<u16> {
    let bytes: [u8; 2] = row
        .get(off..off + 2)
        .ok_or_else(|| Error::invalid_frame("u16 sample out of bounds"))?
        .try_into()
        .expect("slice length checked");
    Ok(u16::from_le_bytes(bytes))
}

fn read_f32_ne(row: &[u8], off: usize) -> Result<f32> {
    let bytes: [u8; 4] = row
        .get(off..off + 4)
        .ok_or_else(|| Error::invalid_frame("f32 sample out of bounds"))?
        .try_into()
        .expect("slice length checked");
    Ok(f32::from_ne_bytes(bytes))
}

fn for_each_plane_sample_pair<F>(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
    index: usize,
    f: &mut F,
) -> Result<usize>
where
    F: FnMut(f64, f64),
{
    if a.pixel_format() != b.pixel_format() {
        return Err(Error::incompatible(
            "plane-domain compare requires identical pixel formats",
        ));
    }
    let fmt = a.pixel_format();
    let dims = fmt.plane_dimensions(a.dimensions(), index)?;
    let (row_bytes, h) = dims.as_usize()?;
    let pa = a.plane(index)?;
    let pb = b.plane(index)?;
    let mut count = 0;
    for y in 0..h {
        let ra = pa.row(y, row_bytes)?;
        let rb = pb.row(y, row_bytes)?;
        for x in 0..row_bytes {
            f(ra[x] as f64 / 255.0, rb[x] as f64 / 255.0);
            count += 1;
        }
    }
    Ok(count)
}

fn for_each_yuv_storage_sample_pair<F>(
    a: &FrameView<'_, Validated>,
    b: &FrameView<'_, Validated>,
    f: &mut F,
) -> Result<usize>
where
    F: FnMut(f64, f64),
{
    let mut count = 0;
    for index in 0..a.pixel_format().plane_count() {
        count += for_each_plane_sample_pair(a, b, index, f)?;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FrameOwned, PixelFormat};

    #[test]
    fn mse_zero_for_identical() {
        let f = FrameOwned::packed_tight(vec![10, 20, 30, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let s = aggregate_error(&f.as_view(), &f.as_view(), SampleDomain::Color).unwrap();
        assert_eq!(s.mse(), 0.0);
    }

    #[test]
    fn parses_windowed_ssim_aliases() {
        for name in [
            "wssim",
            "windowed-ssim",
            "windowed_ssim",
            "ssim-windowed",
            "ssim_windowed",
        ] {
            let metric = MetricSpec::parse(name).unwrap().build().unwrap();
            assert_eq!(metric.name(), "wssim");
        }
    }

    #[test]
    fn parses_ms_ssim_aliases() {
        for name in ["ms-ssim", "ms_ssim", "msssim"] {
            let metric = MetricSpec::parse(name).unwrap().build().unwrap();
            assert_eq!(metric.name(), "ms-ssim");
        }
    }

    #[test]
    fn parses_configurable_interior_radius() {
        let spec = MetricSpec::parse("psnr:rgb-visible-interior2px").unwrap();
        assert_eq!(
            spec.domain,
            SampleDomain::Render(RenderDomain {
                channels: RenderChannels::Rgb,
                mask: RenderMask::Visible,
                interior_radius: 2,
            })
        );
        assert_eq!(spec.key(), "psnr:rgb-visible-interior2px");
    }

    #[test]
    fn rgb_interior_excludes_border_and_alpha_edges() {
        let mut reference = vec![0u8; 4 * 3 * 3];
        let mut distorted = vec![0u8; 4 * 3 * 3];
        for px in 0..9 {
            reference[px * 4 + 3] = 255;
            distorted[px * 4 + 3] = 255;
        }
        reference[4 * 4] = 10;
        distorted[4 * 4] = 20;
        let a = FrameOwned::packed_tight(reference, 3, 3, PixelFormat::Rgba8).unwrap();
        let b = FrameOwned::packed_tight(distorted, 3, 3, PixelFormat::Rgba8).unwrap();
        let domain = SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Rgb,
            mask: RenderMask::Opaque,
            interior_radius: 1,
        });
        let stats = aggregate_error(&a.as_view(), &b.as_view(), domain).unwrap();
        assert_eq!(stats.pixel_count, 1);
        assert_eq!(stats.count, 3);
    }

    #[test]
    fn visible_interior_requires_both_images_visible_in_neighborhood() {
        let mut reference = vec![0u8; 4 * 3 * 3];
        let mut distorted = vec![0u8; 4 * 3 * 3];
        for px in 0..9 {
            reference[px * 4 + 3] = 255;
            distorted[px * 4 + 3] = 255;
        }
        distorted[3] = 0;
        let a = FrameOwned::packed_tight(reference, 3, 3, PixelFormat::Rgba8).unwrap();
        let b = FrameOwned::packed_tight(distorted, 3, 3, PixelFormat::Rgba8).unwrap();
        let domain = SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Rgb,
            mask: RenderMask::Visible,
            interior_radius: 1,
        });
        let err = aggregate_error(&a.as_view(), &b.as_view(), domain).unwrap_err();
        assert!(err.to_string().contains("zero comparable samples"));
    }

    #[test]
    fn oversized_interior_radius_returns_error_without_overflow() {
        let frame = FrameOwned::packed_tight(vec![0, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let domain = SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Rgb,
            mask: RenderMask::Visible,
            interior_radius: usize::MAX,
        });
        let error = aggregate_error(&frame.as_view(), &frame.as_view(), domain).unwrap_err();
        assert!(error.to_string().contains("zero comparable samples"));
    }

    #[test]
    fn alpha_diagnostics_counts_buckets_and_mismatches() {
        let reference =
            FrameOwned::packed_tight(vec![0, 0, 0, 0, 0, 0, 0, 128], 2, 1, PixelFormat::Rgba8)
                .unwrap();
        let distorted =
            FrameOwned::packed_tight(vec![0, 0, 0, 1, 0, 0, 0, 255], 2, 1, PixelFormat::Rgba8)
                .unwrap();
        let diagnostics = alpha_diagnostics(&reference.as_view(), &distorted.as_view()).unwrap();
        assert_eq!(diagnostics.reference.transparent, 1);
        assert_eq!(diagnostics.reference.partial, 1);
        assert_eq!(diagnostics.distorted.partial, 1);
        assert_eq!(diagnostics.distorted.opaque, 1);
        assert_eq!(diagnostics.mismatch_count, 2);
        assert_eq!(diagnostics.max_delta, 127);
        assert_eq!(diagnostics.mismatches_beyond_one_lsb, 1);
    }

    #[test]
    fn hsv_hue_delta_wraps_around() {
        let reference =
            FrameOwned::packed_tight(vec![1, 255, 255], 1, 1, PixelFormat::Hsv8).unwrap();
        let distorted =
            FrameOwned::packed_tight(vec![254, 255, 255], 1, 1, PixelFormat::Hsv8).unwrap();
        let domain = SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Hsv,
            mask: RenderMask::All,
            interior_radius: 0,
        });
        let stats = aggregate_error(&reference.as_view(), &distorted.as_view(), domain).unwrap();
        assert!(stats.max_abs < 0.02);
    }

    #[test]
    fn binary_native_compares_bits() {
        let reference = FrameOwned::new(
            Dimensions::new(4, 1).unwrap(),
            crate::frame::FormatSpec::new(PixelFormat::Binary1Lsb),
            vec![crate::frame::OwnedPlane::new(vec![0b0000_1011], 1)],
        )
        .unwrap();
        let distorted = FrameOwned::new(
            Dimensions::new(4, 1).unwrap(),
            crate::frame::FormatSpec::new(PixelFormat::Binary1Lsb),
            vec![crate::frame::OwnedPlane::new(vec![0b0000_1001], 1)],
        )
        .unwrap();
        let domain = SampleDomain::Render(RenderDomain {
            channels: RenderChannels::Binary,
            mask: RenderMask::All,
            interior_radius: 0,
        });
        let stats = aggregate_error(&reference.as_view(), &distorted.as_view(), domain).unwrap();
        assert_eq!(stats.pixel_count, 4);
        assert_eq!(stats.max_abs, 1.0);
    }

    #[test]
    fn error_metrics_report_distribution_and_semantic_channels() {
        let reference =
            FrameOwned::packed_tight(vec![0, 0, 0, 0, 0, 0], 2, 1, PixelFormat::Rgb8).unwrap();
        let distorted =
            FrameOwned::packed_tight(vec![0, 10, 20, 30, 40, 50], 2, 1, PixelFormat::Rgb8).unwrap();
        let output = MetricSet::from_csv("mae:color")
            .unwrap()
            .compare(&reference.as_view(), &distorted.as_view())
            .unwrap()
            .remove(0);

        assert_eq!(output.details["pixel_count"], 2.0);
        assert_eq!(output.details["channel_sample_count"], 6.0);
        assert_eq!(output.details["changed_samples"], 5.0);
        assert_eq!(output.details["changed_pixels"], 2.0);
        assert_eq!(output.details["pixels_beyond_8_lsb"], 2.0);
        assert_eq!(output.details["samples_beyond_1_lsb"], 5.0);
        assert_eq!(output.details["samples_beyond_8_lsb"], 5.0);
        assert_eq!(output.details["channel_count"], 3.0);
        assert!((output.details["red_mae_code"] - 15.0).abs() < 1e-12);
        assert!((output.details["red_abs_error_p99_code"] - 30.0).abs() < 0.04);
        assert!((output.details["green_mae_code"] - 25.0).abs() < 1e-12);
        assert!((output.details["blue_mae_code"] - 35.0).abs() < 1e-12);
        assert!((output.details["max_pixel_delta_code"] - 70.710_678).abs() < 1e-5);
        assert_eq!(output.details["max_error_x"], 1.0);
        assert_eq!(output.details["max_error_y"], 0.0);
        assert_eq!(output.details["max_error_channel"], 2.0);
        assert!((output.details["abs_error_p50_code"] - 20.0).abs() < 0.04);
        assert!((output.details["abs_error_p99_code"] - 50.0).abs() < 0.04);
    }

    #[test]
    fn error_histograms_merge_without_losing_percentiles() {
        let reference =
            FrameOwned::packed_tight(vec![0, 0, 0, 0], 2, 2, PixelFormat::Luma8).unwrap();
        let distorted =
            FrameOwned::packed_tight(vec![0, 10, 20, 30], 2, 2, PixelFormat::Luma8).unwrap();
        let top = aggregate_error_rows(
            &reference.as_view(),
            &distorted.as_view(),
            SampleDomain::Luma,
            0,
            Some(1),
        )
        .unwrap();
        let bottom = aggregate_error_rows(
            &reference.as_view(),
            &distorted.as_view(),
            SampleDomain::Luma,
            1,
            Some(2),
        )
        .unwrap();
        let combined = combine_error_stats(vec![top, bottom]);
        let full = aggregate_error(
            &reference.as_view(),
            &distorted.as_view(),
            SampleDomain::Luma,
        )
        .unwrap();

        assert_eq!(combined.count, full.count);
        assert_eq!(combined.changed_samples, full.changed_samples);
        assert_eq!(
            combined.absolute_error_percentile(0.95),
            full.absolute_error_percentile(0.95)
        );
        assert_eq!(combined.channels[0].sum_abs, full.channels[0].sum_abs);
    }

    #[test]
    fn plane_domain_reports_channel_statistics() {
        let reference = FrameOwned::packed_tight(vec![0, 10], 2, 1, PixelFormat::Luma8).unwrap();
        let distorted = FrameOwned::packed_tight(vec![0, 20], 2, 1, PixelFormat::Luma8).unwrap();
        let output = MetricSet::from_csv("mae:plane0")
            .unwrap()
            .compare(&reference.as_view(), &distorted.as_view())
            .unwrap()
            .remove(0);
        assert_eq!(output.details["channel_count"], 1.0);
        assert!((output.details["plane_mae_code"] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn nv12_all_domain_keeps_y_u_v_channel_statistics() {
        let dimensions = Dimensions::new(2, 2).unwrap();
        let reference = FrameOwned::new(
            dimensions,
            crate::frame::FormatSpec::new(PixelFormat::Nv12),
            vec![
                crate::frame::OwnedPlane::new(vec![0; 4], 2),
                crate::frame::OwnedPlane::new(vec![0; 2], 2),
            ],
        )
        .unwrap();
        let distorted = FrameOwned::new(
            dimensions,
            crate::frame::FormatSpec::new(PixelFormat::Nv12),
            vec![
                crate::frame::OwnedPlane::new(vec![0, 10, 0, 10], 2),
                crate::frame::OwnedPlane::new(vec![20, 30], 2),
            ],
        )
        .unwrap();
        let output = MetricSet::from_csv("mae:all")
            .unwrap()
            .compare(&reference.as_view(), &distorted.as_view())
            .unwrap()
            .remove(0);

        assert_eq!(output.details["channel_count"], 3.0);
        assert_eq!(output.details["channel_0_samples"], 4.0);
        assert_eq!(output.details["channel_1_samples"], 1.0);
        assert_eq!(output.details["channel_2_samples"], 1.0);
        assert!((output.details["channel_0_mae_code"] - 5.0).abs() < 1e-12);
        assert!((output.details["channel_1_mae_code"] - 20.0).abs() < 1e-12);
        assert!((output.details["channel_2_mae_code"] - 30.0).abs() < 1e-12);
        assert!((output.details["y_mae_code"] - 5.0).abs() < 1e-12);
        assert!((output.details["u_mae_code"] - 20.0).abs() < 1e-12);
        assert!((output.details["v_mae_code"] - 30.0).abs() < 1e-12);
    }
}
