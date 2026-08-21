# Metrics

## Implemented in the Sans I/O core

- `mse` — mean squared error over the selected sample domain.
- `rmse` — square root of MSE.
- `psnr` — `20 * log10(max_sample / sqrt(MSE))`; returns infinity for identical samples.
- `mae` — mean absolute error.
- `maxae` — maximum absolute error.
- `ssim` — global luma SSIM. Aliases: `global-ssim`, `global_ssim`.
- `wssim` — box-windowed luma SSIM. Aliases: `windowed-ssim`, `windowed_ssim`, `ssim-windowed`, `ssim_windowed`.
- `ms-ssim` — deterministic luma multi-scale SSIM. Aliases: `ms_ssim`, `msssim`.

Metric names can include domains where the metric supports sample-domain selection:

```text
psnr          # luma by default
psnr:luma
psnr:color    # RGB/luma-expanded color channels, alpha ignored
mse:all       # all stored channels; for YUV requires matching YUV layout
mse:plane0    # raw plane comparison
```

`ssim`, `wssim`, and `ms-ssim` currently operate on luma samples. RGB-like inputs are converted to luma through frame color-space metadata, and YUV inputs use the Y plane directly.

## Sample domains

`SampleDomain::Luma` converts RGB-like inputs to luma using the frame color-space metadata. For YUV inputs, the Y plane is used directly.

`SampleDomain::Color` compares RGB triples. YUV inputs are converted to RGB using simple BT.601/BT.709/BT.2020 matrices. This is appropriate for quick full-reference diagnostics; color-managed HDR workflows should convert externally or use the explicit helpers in `imq::color`.

`SampleDomain::All` compares stored components. For RGBA this includes alpha. For YUV it compares Y/U/V storage samples and requires matching YUV layouts.
The aggregate is storage-sample weighted, so subsampled chroma contributes in
proportion to the bytes actually stored. Per-channel details keep Y, U, and V
separate (including deinterleaving NV12 U/V) so callers can also enforce a
channel-balanced policy explicitly.

`SampleDomain::Plane(n)` compares raw plane bytes normalized to 0..1 for a single plane.

## Error distributions and channel diagnostics

All built-in error metrics (`mse`, `rmse`, `psnr`, `mae`, and `maxae`) share one
domain scan inside `MetricSet`. Their `MetricOutput::details` include:

- `abs_error_p50`, `abs_error_p90`, `abs_error_p95`, and `abs_error_p99`, plus
  corresponding `_code` values in 8-bit code units;
- `changed_samples` and `changed_sample_ratio`;
- exact sample counts/ratios beyond 1, 2, 4, and 8 normalized 8-bit LSBs;
- changed-pixel counts/ratios and pixel counts beyond the same tolerances when
  the domain has an unambiguous pixel grouping;
- `pixel_count` separately from `channel_sample_count`;
- generic `channel_0_*` through `channel_3_*` statistics and semantic names
  such as `red_mae`, `green_rmse`, `blue_psnr`, `blue_max_delta`, signed
  `mean_error`, channel-level `abs_error_p50`/`p90`/`p95`/`p99`, and
  `alpha_mae` where the domain has unambiguous channels. Signed error is
  `reference - distorted`.

Percentiles use a deterministic 4096-bin streaming histogram. This avoids an
image-sized allocation while retaining sub-8-bit-code precision. The bin count
is reported as `error_histogram_bins` so downstream consumers can identify the
quantization policy.

## Windowed SSIM

`WindowedSsim::new()` preserves the first windowed-SSIM implementation: 8x8
luma windows with 8-pixel strides and common SSIM constants. When a dimension
is not divisible by eight, the final full-size window is edge-anchored and can
overlap its predecessor. `WindowedSsim::with_sliding_window(width, height,
stride_x, stride_y)` enables other layouts such as 8x8 with 4-pixel stride.

The CLI metric aliases build the default `WindowedSsim::new()` configuration.
Besides mean/min/max, windowed SSIM reports deterministic `p01_window_ssim`,
`p05_window_ssim`, and `p50_window_ssim` details. Low-tail gates catch a small
bad region that can disappear in the whole-frame mean. `worst_window_x` and
`worst_window_y` identify the top-left sample of the minimum-SSIM window.

## MS-SSIM

`MsSsim::new()` uses 8x8 local windows, 4-pixel strides, 2x2 box downsampling, and the standard five-scale weights:

```text
[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
```

For images that reach 1x1 before five scales, the available weights are renormalized over the scales actually computed. This keeps tiny images deterministic and avoids fabricating missing scales.
Each scale also reports its window count, minimum and p05 local SSIM, and the
coordinates of its worst local window, making small structural regressions
visible even when the combined score changes little.

## Diff images and heatmaps

`imq::diff::diff_image` creates an in-memory RGBA8 diff image using one of three modes:

- `AbsoluteRgb` — per-channel absolute RGB deltas.
- `Heatmap` — scalar RGB error magnitude mapped to a heatmap.
- `SignedLuma` — red for brighter distorted pixels, blue for darker distorted pixels.

With `image-codecs`, `RgbaImageData::save_png` writes the result as PNG.

## CI gates

`imq::gate` evaluates rules such as:

```text
psnr>=40
wssim>=0.98
mae<=0.01
```

Rules are evaluated against `MetricOutput` rows and return a structured `GateEvaluation` with per-rule pass/fail messages.

A rule can select a numeric detail using `metric.detail`, for example:

```text
mae:color.red_mae<=0.01
maxae:rgba.alpha_max_delta_code<=1
mae:color.abs_error_p99<=0.02
```

`BaselineThresholdRule` compares the candidate with a previous/baseline
candidate using the same reference. Its right-hand side supports `baseline`, an
additive allowance, or a scale:

```text
psnr>=baseline-0.5
ssim>=baseline*0.995
mae:color.abs_error_p99<=baseline*1.05
```

`evaluate_baseline_thresholds` returns a structured `BaselineGateEvaluation`
containing candidate value, baseline value, derived threshold, and pass/fail
status for every rule. `QualityGateReport` combines those checks with all metric
rows, dimensions, and reference/candidate/baseline format metadata for a durable
CI artifact.

## Structured floating-point values

Finite metric scores and details are serialized as ordinary numbers. Because
JSON has no IEEE-754 non-finite literals, perfect-match PSNR and other
non-finite values use the reversible strings `"Infinity"`, `"-Infinity"`, and
`"NaN"` in JSON-compatible reports. A missing optional value alone is `null`.
The same representation is accepted when deserializing reports, so a perfect
score is never confused with missing data.

## Video aggregation and timestamp pairing

`imq::video_analysis::aggregate_video_report` computes finite/non-finite sample
counts, mean, time-weighted mean, standard deviation, min, max, median,
configured percentiles, and worst-frame information from a `VideoReport`.
`pair_frames_by_timestamp` pairs reference/distorted frames by nearest PTS with
a configurable maximum delta and optional distorted-frame reuse.
`align_frames_by_timestamp` adds unmatched-frame and residual diagnostics, and
the transform helpers support explicit offset/clock-drift correction.

## External metrics

`imq::external::ExternalMetricCommand` runs an explicit external command and parses the first finite float from stdout into a `MetricOutput`. Use this for VMAF, LPIPS scripts, or organization-specific metrics while keeping command execution outside the Sans-I/O core.

## Color and HDR helpers

`imq::color` includes full/limited range expansion, sRGB/BT.1886/HLG/PQ transfer helpers, luma weights, and RGB/YUV conversion helpers. These functions make color assumptions explicit for HDR or color-managed workflows.

## Reference vectors

`imq::reference_vectors::assert_reference_vectors()` runs small codec-free Luma8 fixtures through MSE, MAE, and PSNR to detect normalization drift.

## Extending metrics

Implement the `Metric` trait:

```rust
use imq::{Direction, FrameView, Metric, MetricOutput, Result, Validated};

struct MyMetric;

impl Metric for MyMetric {
    fn name(&self) -> &'static str { "my_metric" }

    fn compare(&self, reference: &FrameView<'_, Validated>, distorted: &FrameView<'_, Validated>) -> Result<MetricOutput> {
        let _ = (reference, distorted);
        Ok(MetricOutput::new("my_metric", 0.0, "unitless", Direction::Neutral))
    }
}
```

For I/O or model-heavy metrics, keep the core computation separable from file/process/model loading so it remains testable and deterministic.
