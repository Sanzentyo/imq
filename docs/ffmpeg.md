# ffmpeg adapter

The `ffmpeg` feature adds `video::ffmpeg`.

## What it does

1. `probe_video` runs `ffprobe` and parses dimensions, rates, time base,
   duration, bitrate, pixel format, color range/matrix/transfer/primaries, and
   field order. `probe_video_frames` returns exact per-frame PTS/duration and
   coding metadata.
2. `FfmpegFrameIter` spawns `ffmpeg` and asks for `-pix_fmt rgba -f rawvideo pipe:1`.
3. Each full frame is wrapped as `FrameOwned::packed_tight(..., PixelFormat::Rgba8)`.
4. `compare_videos` feeds decoded frame pairs to `MetricSet`.
5. `compare_videos_streaming` invokes a callback and retains only aggregate
   state, allowing bounded-memory long-video analysis.
6. `compare_videos_by_timestamp` pairs exact PTS values, reports unmatched and
   invalid timestamps, and can estimate affine clock offset/drift for VFR,
   duplicated-frame, and dropped-frame inputs.

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

For VFR-safe comparison:

```rust
use imq::video::{TimestampVideoCompareOptions, compare_videos_by_timestamp};

let report = compare_videos_by_timestamp(
    "reference.mkv",
    "candidate.mkv",
    &ffmpeg,
    &TimestampVideoCompareOptions::default(),
    &metrics,
)?;
println!(
    "unmatched candidate frames: {}",
    report.alignment.unmatched_distorted_indices.len()
);
# Ok::<(), imq::Error>(())
```

The same path is exposed by the main CLI, including structured alignment
diagnostics:

```bash
imq video reference-vfr.mkv candidate-vfr.mkv --align timestamp \
  --max-timestamp-delta 0.02 --format json
```

Automatic affine offset/drift estimation is the default. Supply
`--timestamp-scale` and/or `--timestamp-offset` to use a known transform;
`--no-estimate-timestamp-transform` applies identity alignment when no explicit
transform is provided. `--allow-reuse-distorted` permits a candidate frame to
serve multiple reference timestamps.

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
- `compare_videos` intentionally remains the faster decode-order path. Use
  `compare_videos_by_timestamp` whenever presentation timelines may differ.
- Decoder iterators detect truncated raw frames, drain stderr concurrently to
  avoid subprocess pipe deadlocks, and terminate/wait for child processes on
  early drop.
