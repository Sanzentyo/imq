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

## 5. Video aggregation policy

Implemented in `imq::video_analysis::aggregate_video_report`. It computes mean, min, max, median, requested percentiles, and worst-frame metadata from a `VideoReport`.

## 6. Timestamp-aware video pairing

Implemented in `imq::video_analysis::pair_frames_by_timestamp`. It pairs nearest PTS frames with `TimestampPairingOptions { max_delta_seconds, allow_reuse_distorted }`.

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

The main `imq` binary remains focused on existing image/video compare workflows. `imq-quality` is a supplemental CLI so these advanced helpers can be used without destabilizing the main command surface.
