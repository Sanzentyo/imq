# ffmpeg adapter

The `ffmpeg` feature adds `video::ffmpeg`.

## What it does

1. `probe_video` runs `ffprobe` and parses JSON stream metadata.
2. `FfmpegFrameIter` spawns `ffmpeg` and asks for `-pix_fmt rgba -f rawvideo pipe:1`.
3. Each full frame is wrapped as `FrameOwned::packed_tight(..., PixelFormat::Rgba8)`.
4. `compare_videos` feeds decoded frame pairs to `MetricSet`.

The comparison core never knows that frames came from a subprocess.

## Example

```rust
use imq::metrics::MetricSet;
use imq::video::{compare_videos, FfmpegOptions, VideoCompareOptions};

let metrics = MetricSet::from_csv("psnr,ssim,mse")?;
let ffmpeg = FfmpegOptions::default();
let opts = VideoCompareOptions { every: 30, max_frames: Some(120) };
let report = compare_videos("reference.mp4", "distorted.mp4", &ffmpeg, &opts, &metrics)?;
# Ok::<(), imq::Error>(())
```

## Specific frame extraction

`decode_single_frame(path, index, options)` uses a `select=eq(n\,index)` filter and returns one RGBA8 `FrameOwned`.

CLI:

```bash
imq extract-frame input.mp4 120 frame-120.png
```

## Scaling

Set `FfmpegOptions::scale` or CLI `--width/--height` to compare videos with different source dimensions after a common ffmpeg scale step.

## Notes

- This adapter expects `ffmpeg` and `ffprobe` to be available on PATH unless custom paths are supplied.
- Time-base and VFR alignment are intentionally minimal in this first implementation. `compare_videos` compares decode-order frame pairs. For production VFR workflows, add timestamp-aware pairing before feeding frames to the core.
