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

`SampleDomain::Plane(n)` compares raw plane bytes normalized to 0..1 for a single plane.

## Windowed SSIM

`WindowedSsim::new()` preserves the first windowed-SSIM implementation: 8x8 non-overlapping luma windows with common SSIM constants. `WindowedSsim::with_sliding_window(width, height, stride_x, stride_y)` enables overlapping windows such as 8x8 with 4-pixel stride.

The CLI metric aliases build the default `WindowedSsim::new()` configuration.

## MS-SSIM

`MsSsim::new()` uses 8x8 local windows, 4-pixel strides, 2x2 box downsampling, and the standard five-scale weights:

```text
[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
```

For images that reach 1x1 before five scales, the available weights are renormalized over the scales actually computed. This keeps tiny images deterministic and avoids fabricating missing scales.

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

## Video aggregation and timestamp pairing

`imq::video_analysis::aggregate_video_report` computes mean, min, max, median, configured percentiles, and worst-frame information from a `VideoReport`. `pair_frames_by_timestamp` pairs reference/distorted frames by nearest PTS with a configurable maximum delta and optional distorted-frame reuse.

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
