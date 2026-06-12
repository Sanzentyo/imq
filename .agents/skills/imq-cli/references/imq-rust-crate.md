# imq Rust Crate Reference

Use this when the user wants to call `imq` from Rust instead of shelling out to
the `imq` CLI. Use the CLI when the task is a one-off local comparison; use the
crate when comparison is part of an application, renderer, test harness, or
pipeline.

## Dependency

Until a crates.io release exists, depend on the Git repository. Prefer a fixed
tag or revision for reproducible builds.

```toml
[dependencies]
imq = { git = "https://github.com/Sanzentyo/imq.git", tag = "v0.1.0", default-features = false }
```

Common feature sets:

```toml
# Core frame + metrics + serde reports, no filesystem decoding.
imq = { git = "https://github.com/Sanzentyo/imq.git", tag = "v0.1.0", default-features = false, features = ["serde"] }

# Load PNG/JPEG/WebP/etc. through the image crate.
imq = { git = "https://github.com/Sanzentyo/imq.git", tag = "v0.1.0", default-features = false, features = ["serde", "image-codecs"] }

# Video comparison through external ffmpeg/ffprobe.
imq = { git = "https://github.com/Sanzentyo/imq.git", tag = "v0.1.0", default-features = false, features = ["serde", "image-codecs", "ffmpeg"] }
```

The default feature set includes `std`, `serde`, `image-codecs`, `cli`, `tui`,
and `preview`. Prefer `default-features = false` for library code unless those
frontends are wanted.

## Sans-I/O Frames

Use `FrameView` for borrowed buffers and `FrameOwned` for owned buffers.
`stride` is bytes per row.

```rust
use imq::{FrameView, MetricSet, PixelFormat, Result};

fn main() -> Result<()> {
    let reference_rgba = [0, 0, 0, 255, 255, 255, 255, 255];
    let distorted_rgba = [0, 0, 0, 255, 250, 250, 250, 255];

    let reference = FrameView::packed(&reference_rgba, 2, 1, PixelFormat::Rgba8, 8)?.validate()?;
    let distorted = FrameView::packed(&distorted_rgba, 2, 1, PixelFormat::Rgba8, 8)?.validate()?;

    let metrics = MetricSet::from_csv("psnr:rgb-visible,mse:rgba,mae:rgb-all")?;
    for metric in metrics.compare(&reference, &distorted)? {
        println!("{} = {} {}", metric.name, metric.score, metric.unit);
    }
    Ok(())
}
```

For owned tight-packed input:

```rust
use imq::{FrameOwned, PixelFormat, Result};

fn rgba_frame(bytes: Vec<u8>, width: u32, height: u32) -> Result<FrameOwned> {
    FrameOwned::packed_tight(bytes, width, height, PixelFormat::Rgba8)
}
```

Supported core pixel formats include `Luma8`, `Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`,
`Hsv8`, `Hsva8`, `Binary1Lsb`, `Binary1Msb`, 16-bit little-endian formats,
float RGB/RGBA, YUV 4:2:0, and NV12.

## Loading Encoded Images

Enable `image-codecs` and use the image crate adapter.

```rust
use imq::adapters::image_crate;
use imq::{ComparisonReport, MetricSet, Result};

fn main() -> Result<()> {
    let reference = image_crate::load_image_path("reference.png")?;
    let candidate = image_crate::load_image_path("candidate.png")?;

    let metrics = MetricSet::from_csv("psnr:rgb-visible,ssim,mse:rgba")?;
    let outputs = metrics.compare(&reference.as_view(), &candidate.as_view())?;

    let report = ComparisonReport::new(
        reference.dimensions(),
        reference.format(),
        candidate.format(),
        outputs,
    )
    .with_labels("reference.png", "candidate.png");

    println!("{}", report.to_json_pretty()?);

    Ok(())
}
```

Use `image_crate::decode_image_bytes(&bytes)` when input comes from HTTP, SSH,
stdin, or another byte stream.

## imqraw Bundles

Use `imqraw` when a renderer or test harness should emit exact raw frames with
labels and tags. See `imqraw-library.md` for JS/TS and fuller bundle examples.

```rust
use imq::{
    FrameOwned, PixelFormat, RawImageBundle, RawImageRecord, Result,
    decode_imqraw_bundle, encode_imqraw_bundle,
};

fn main() -> Result<()> {
    let reference = RawImageRecord::new(
        Some("reference".to_string()),
        vec!["ref".to_string()],
        FrameOwned::packed_tight(vec![0, 0, 0, 255], 1, 1, PixelFormat::Rgba8)?,
    );
    let candidate = RawImageRecord::new(
        Some("candidate".to_string()),
        vec!["dist".to_string()],
        FrameOwned::packed_tight(vec![4, 4, 4, 255], 1, 1, PixelFormat::Rgba8)?,
    );

    let bytes = encode_imqraw_bundle(&RawImageBundle::new(vec![reference, candidate]))?;
    let decoded = decode_imqraw_bundle(&bytes)?;
    let reference = &decoded.select_tag("ref")?.frame;
    let candidate = &decoded.select_tag("dist")?.frame;

    let metrics = imq::MetricSet::from_csv("psnr:rgb-visible")?;
    let _outputs = metrics.compare(&reference.as_view(), &candidate.as_view())?;
    Ok(())
}
```

## Video

Enable `ffmpeg` for video helpers. They call external `ffmpeg` and `ffprobe`;
missing tools are returned as errors.

```rust
use imq::metrics::MetricSet;
use imq::video::{FfmpegOptions, VideoCompareOptions, compare_videos};
use imq::Result;

fn main() -> Result<()> {
    let metrics = MetricSet::from_csv("psnr,ssim,mse")?;
    let report = compare_videos(
        "reference.mp4",
        "candidate.mp4",
        &FfmpegOptions::default(),
        &VideoCompareOptions { every: 30, max_frames: Some(120) },
        &metrics,
    )?;
    println!("frames compared: {}", report.frames.len());
    Ok(())
}
```

Use `imq::video::decode_single_frame(path, frame_index, &FfmpegOptions::default())`
when a video frame should be compared by the still-image metric pipeline.

## Metric and Report Patterns

`MetricSet::from_csv` accepts the same metric domain strings as the CLI, such as
`psnr`, `ssim`, `mse:rgba`, `psnr:rgb-visible`, `psnr:rgb-interior2px`,
`psnr:rgb-nonblack-interior2px`, `mae:gray`, and `maxae:binary`.

Use `ComparisonReport` for still-image reports and `VideoReport` for full video
reports. With `serde` enabled, reports can be serialized by the caller or
rendered with `to_json_pretty()`.

## Remote and Frame-Pair Helpers

With the `std` feature enabled, the crate exposes parsing/types for remote
inputs and ordered video frame pairs:

```rust
use imq::{InputSpec, parse_video_frame_pairs};

fn main() -> imq::Result<()> {
    let input = InputSpec::parse("ssh://user@example.com:2222/tmp/out.png")?;
    let pairs = parse_video_frame_pairs("0:1,(30,31,30)")?;
    println!("{} {}", input.display_label(), pairs.len());
    Ok(())
}
```

The CLI owns SSH process execution and copy-mode policy. Library callers should
use these types to preserve compatible parsing, then provide their own transport
or call CLI-level code where appropriate.
