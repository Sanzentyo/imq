//! Quality metrics and Sans-I/O comparison engine.

use crate::frame::{ColorSpace, Dimensions, FrameView, PixelFormat, Validated};
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::thread;

const PARALLEL_ERROR_MIN_PIXELS: usize = 512 * 512;

mod basic;
mod ssim;

pub use basic::{Mae, MaxAbsoluteError, Mse, Psnr, Rmse};
pub use ssim::Ssim;

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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
}

/// One metric result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetricOutput {
    /// Stable metric name.
    pub name: String,
    /// Numeric score. Some metrics may be `f64::INFINITY` for perfect matches.
    pub score: f64,
    /// Score unit, e.g. `dB`, `unitless`, `normalized_code`.
    pub unit: String,
    /// Direction of quality.
    pub direction: Direction,
    /// Extra metric-specific values.
    pub details: BTreeMap<String, f64>,
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
    /// Metric name: `mse`, `rmse`, `psnr`, `mae`, `maxae`, `ssim`.
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

    /// Parses `name[:domain]`, where domain is `luma`, `color`, `all`, or `planeN`.
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

    /// Builds a metric instance from this spec.
    pub fn build(&self) -> Result<Box<dyn Metric>> {
        match self.name.as_str() {
            "mse" => Ok(Box::new(Mse::new(self.domain))),
            "rmse" => Ok(Box::new(Rmse::new(self.domain))),
            "psnr" => Ok(Box::new(Psnr::new(self.domain))),
            "mae" => Ok(Box::new(Mae::new(self.domain))),
            "maxae" | "max_ae" | "max-error" => Ok(Box::new(MaxAbsoluteError::new(self.domain))),
            "ssim" => Ok(Box::new(Ssim::new())),
            other => Err(Error::UnknownMetric(other.to_string())),
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
    let (_, h) = dims.as_usize()?;
    let end_y = end_y.unwrap_or(h).min(h);
    let mut sum_sq = 0.0;
    let mut sum_abs = 0.0;
    let mut max_abs = 0.0;
    let mut count = 0usize;

    match domain {
        SampleDomain::Luma => {
            for y in start_y..end_y {
                for x in 0..usize::try_from(dims.width)
                    .map_err(|_| Error::invalid_frame("width overflows usize"))?
                {
                    let x_value = read_luma(a, x, y)?;
                    let y_value = read_luma(b, x, y)?;
                    push_error_sample(&mut sum_sq, &mut sum_abs, &mut max_abs, x_value, y_value);
                    count += 1;
                }
            }
        }
        SampleDomain::Color => {
            for y in start_y..end_y {
                for x in 0..usize::try_from(dims.width)
                    .map_err(|_| Error::invalid_frame("width overflows usize"))?
                {
                    let ca = read_rgb(a, x, y)?;
                    let cb = read_rgb(b, x, y)?;
                    for c in 0..3 {
                        push_error_sample(&mut sum_sq, &mut sum_abs, &mut max_abs, ca[c], cb[c]);
                        count += 1;
                    }
                }
            }
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
                for x in 0..usize::try_from(dims.width)
                    .map_err(|_| Error::invalid_frame("width overflows usize"))?
                {
                    let ca = read_stored_channels(a, x, y)?;
                    let cb = read_stored_channels(b, x, y)?;
                    for c in 0..channels_a {
                        push_error_sample(&mut sum_sq, &mut sum_abs, &mut max_abs, ca[c], cb[c]);
                        count += 1;
                    }
                }
            }
        }
        _ => {
            if start_y != 0 || end_y != h {
                return Err(Error::unsupported(
                    "row-range aggregation is unsupported for this sample domain",
                ));
            }
            count = for_each_sample_pair(a, b, domain, |x, y| {
                push_error_sample(&mut sum_sq, &mut sum_abs, &mut max_abs, x, y);
            })?;
        }
    }

    if count == 0 {
        return Err(Error::unsupported(
            "metric received zero comparable samples",
        ));
    }
    Ok(ErrorStats {
        count,
        sum_sq,
        sum_abs,
        max_abs,
    })
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
    matches!(domain, SampleDomain::Luma | SampleDomain::Color)
        || (matches!(domain, SampleDomain::All)
            && !a.pixel_format().is_yuv()
            && !b.pixel_format().is_yuv())
}

fn push_error_sample(
    sum_sq: &mut f64,
    sum_abs: &mut f64,
    max_abs: &mut f64,
    reference: f64,
    distorted: f64,
) {
    let d = reference - distorted;
    let ad = d.abs();
    *sum_sq += d * d;
    *sum_abs += ad;
    if ad > *max_abs {
        *max_abs = ad;
    }
}

fn combine_error_stats(partials: Vec<ErrorStats>) -> ErrorStats {
    partials
        .into_iter()
        .fold(ErrorStats::default(), |mut acc, stats| {
            acc.count += stats.count;
            acc.sum_sq += stats.sum_sq;
            acc.sum_abs += stats.sum_abs;
            acc.max_abs = acc.max_abs.max(stats.max_abs);
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

fn error_metric_output(spec: &MetricSpec, stats: &ErrorStats) -> MetricOutput {
    match spec.name.as_str() {
        "mse" => MetricOutput::new(
            "mse",
            stats.mse(),
            "normalized_code^2",
            Direction::LowerIsBetter,
        )
        .with_detail("samples", stats.count as f64),
        "rmse" => MetricOutput::new(
            "rmse",
            stats.rmse(),
            "normalized_code",
            Direction::LowerIsBetter,
        )
        .with_detail("samples", stats.count as f64),
        "psnr" => {
            let mse = stats.mse();
            let psnr = if mse == 0.0 {
                f64::INFINITY
            } else {
                10.0 * (1.0 / mse).log10()
            };
            MetricOutput::new("psnr", psnr, "dB", Direction::HigherIsBetter)
                .with_detail("mse", mse)
                .with_detail("samples", stats.count as f64)
        }
        "mae" => MetricOutput::new(
            "mae",
            stats.mae(),
            "normalized_code",
            Direction::LowerIsBetter,
        )
        .with_detail("samples", stats.count as f64),
        "maxae" | "max_ae" | "max-error" => MetricOutput::new(
            "maxae",
            stats.max_abs,
            "normalized_code",
            Direction::LowerIsBetter,
        )
        .with_detail("samples", stats.count as f64),
        _ => unreachable!("error metric names are filtered before output"),
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ErrorStats {
    pub count: usize,
    pub sum_sq: f64,
    pub sum_abs: f64,
    pub max_abs: f64,
}

impl ErrorStats {
    pub fn mse(&self) -> f64 {
        self.sum_sq / self.count as f64
    }

    pub fn rmse(&self) -> f64 {
        self.mse().sqrt()
    }

    pub fn mae(&self) -> f64 {
        self.sum_abs / self.count as f64
    }
}

fn stored_channel_count(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Rgba8 | PixelFormat::Bgra8 | PixelFormat::Rgba16Le | PixelFormat::RgbaF32 => 4,
        PixelFormat::Rgb8 | PixelFormat::Bgr8 | PixelFormat::Rgb16Le | PixelFormat::RgbF32 => 3,
        PixelFormat::Luma8 | PixelFormat::Luma16Le => 1,
        PixelFormat::Yuv444p8
        | PixelFormat::Yuv422p8
        | PixelFormat::Yuv420p8
        | PixelFormat::Nv12 => 3,
    }
}

fn read_luma(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<f64> {
    match frame.pixel_format() {
        PixelFormat::Luma8 | PixelFormat::Luma16Le => Ok(read_stored_channels(frame, x, y)?[0]),
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
        PixelFormat::Yuv444p8
        | PixelFormat::Yuv422p8
        | PixelFormat::Yuv420p8
        | PixelFormat::Nv12 => {
            let yuv = read_yuv(frame, x, y)?;
            Ok(yuv_to_rgb(yuv, frame.format().color_space))
        }
    }
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
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => [
            row[off] as f64 / max,
            row[off + 1] as f64 / max,
            row[off + 2] as f64 / max,
            1.0,
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
}
