# Metrics

## Implemented in the Sans I/O core

- `mse` — mean squared error over the selected sample domain.
- `rmse` — square root of MSE.
- `psnr` — `20 * log10(max_sample / sqrt(MSE))`; returns infinity for identical samples.
- `mae` — mean absolute error.
- `maxae` — maximum absolute error.
- `ssim` — global luma SSIM over the whole frame.
- `wssim` — sample-weighted mean of non-overlapping windowed luma SSIM values. Aliases: `windowed-ssim`, `windowed_ssim`, `ssim-windowed`, and `ssim_windowed`.

Metric names can include domains:

```text
psnr          # luma by default
psnr:luma
psnr:color    # RGB/luma-expanded color channels, alpha ignored
mse:all       # all stored channels; for YUV requires matching YUV layout
mse:plane0    # raw plane comparison
wssim         # non-overlapping 8x8 luma windows by default
```

`ssim` and `wssim` currently operate on perceptual luma. Domain suffixes are
accepted by the metric parser for consistency, but these SSIM variants do not
switch to color/all/plane domains.

## Sample domains

`SampleDomain::Luma` converts RGB-like inputs to luma using the frame color-space metadata. For YUV inputs, the Y plane is used directly.

`SampleDomain::Color` compares RGB triples. YUV inputs are converted to RGB using simple BT.601/BT.709/BT.2020 matrices. This is appropriate for quick full-reference diagnostics; color-managed HDR workflows should convert externally to a well-defined linear or perceptual space.

`SampleDomain::All` compares stored components. For RGBA this includes alpha. For YUV it compares Y/U/V storage samples and requires matching YUV layouts.

`SampleDomain::Plane(n)` compares raw plane bytes normalized to 0..1 for a single plane.

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
