# Advanced Quality Workflows

This document maps the 1–10 implementation plan to concrete modules and command examples.

## 1. Windowed SSIM

Implemented as `imq::metrics::WindowedSsim` and CLI metric aliases `wssim`, `windowed-ssim`, `windowed_ssim`, `ssim-windowed`, and `ssim_windowed`.

```rust
use imq::{Metric, WindowedSsim};

let metric = WindowedSsim::with_sliding_window(8, 8, 4, 4)?;
let output = metric.compare(&reference, &distorted)?;
```

## 2. MS-SSIM

Implemented as `imq::metrics::MsSsim` and CLI metric aliases `ms-ssim`, `ms_ssim`, and `msssim`.

```rust
use imq::{Metric, MsSsim};

let output = MsSsim::new().compare(&reference, &distorted)?;
```

## 3. Diff image / heatmap output

Implemented in `imq::diff`. The supplemental CLI writes PNG files:

```bash
cargo run --bin imq-quality -- diff reference.png distorted.png diff.png --mode heatmap
cargo run --bin imq-quality -- diff reference.png distorted.png signed.png --mode signed-luma --scale 8
```

## 4. CI threshold / fail gate

Implemented in `imq::gate`. The supplemental CLI exits non-zero when a gate fails:

```bash
cargo run --bin imq-quality -- gate reference.png distorted.png \
  --metrics psnr,ssim,wssim,ms-ssim,mae \
  --rule 'psnr>=40' --rule 'wssim>=0.98' --rule 'mae<=0.01'
```

Error-distribution and semantic-channel details can be gated directly. A prior
candidate can also act as a relative baseline, while `--report` keeps the full
metric rows and both gate evaluations in machine-readable JSON:

```bash
cargo run --bin imq-quality -- gate reference.png candidate.png \
  --baseline previous.png \
  --metrics psnr,mae:color,maxae:rgba \
  --rule 'psnr>=baseline-0.5' \
  --rule 'mae:color.abs_error_p99<=baseline*1.05' \
  --rule 'maxae:rgba.alpha_max_delta_code<=1' \
  --report quality-gate.json
```

## 5. Video aggregation policy

Implemented in `imq::video_analysis::aggregate_video_report`. It computes
finite/non-finite sample counts, mean, time-weighted mean, standard deviation,
min, max, median, requested percentiles, and worst-frame metadata from a
`VideoReport`.

## 6. Timestamp-aware video pairing

Implemented in `imq::video_analysis`. `pair_frames_by_timestamp` pairs nearest
PTS frames with `TimestampPairingOptions { max_delta_seconds,
allow_reuse_distorted }`. `align_frames_by_timestamp` additionally reports
unmatched frames and residuals; `estimate_timestamp_transform` estimates affine
clock offset/drift for the timestamp-aware comparison path (enabled by default
unless the caller supplies a transform or disables estimation). The same path
is available as `imq video ... --align timestamp`, with alignment diagnostics in
text, JSON, YAML, TOML, CSV, and SQLite reports.

## 7. Color management / HDR helpers

Implemented in `imq::color`. Helpers cover limited/full range expansion, transfer decoding/encoding for sRGB, BT.1886, HLG, and PQ, luma weights, RGB luma, and YUV-to-RGB conversion.

## 8. GPU windowed SSIM

Implemented as `GpuContext::windowed_ssim_rgba8` under the `gpu` feature. The WGSL shader evaluates local SSIM windows for RGBA8 frames and the CPU performs the final window mean after readback.

```rust
use imq::gpu::GpuContext;

let gpu = GpuContext::new()?;
let output = gpu.windowed_ssim_rgba8(&reference, &distorted, 8, 8, 4, 4)?;
```

## 9. External metric wrapper

Implemented in `imq::external`. The supplemental CLI supports placeholder expansion and first-float stdout parsing:

```bash
cargo run --bin imq-quality -- external reference.mp4 distorted.mp4 \
  --program ffmpeg-quality-metrics \
  --arg '{reference}' --arg '{distorted}' \
  --metric-name vmaf --direction higher
```

## 10. Reference-vector tests

Implemented in `imq::reference_vectors`. The built-in codec-free fixtures exercise MSE, MAE, and PSNR normalization.

```rust
imq::assert_reference_vectors()?;
```

## Notes

The main `imq` binary exposes the common image/video, suite, GPU, and timestamp
alignment workflows. `imq-quality` remains the supplemental CLI for specialized
quality gates and helper operations.
