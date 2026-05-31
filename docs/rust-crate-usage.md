# Rust Crate Usage

This guide shows how to use `imq` as a Rust library without going through the
CLI. The core APIs are Sans I/O: callers provide image planes and choose where
bytes come from or go to.

## Dependency

Use the Git repository directly until a crates.io release is published:

```toml
[dependencies]
imq = { git = "https://github.com/Sanzentyo/imq.git", tag = "v0.1.0", default-features = false }
```

Enable optional adapters only when needed:

```toml
imq = { git = "https://github.com/Sanzentyo/imq.git", tag = "v0.1.0", features = ["image-codecs"] }
```

## Compare In-Memory Frames

```rust
use imq::{FrameView, MetricSet, PixelFormat, Result};

fn main() -> Result<()> {
    let reference_rgba = [0, 0, 0, 255, 255, 255, 255, 255];
    let distorted_rgba = [0, 0, 0, 255, 250, 250, 250, 255];

    let reference = FrameView::packed(&reference_rgba, 2, 1, PixelFormat::Rgba8, 8)?.validate()?;
    let distorted = FrameView::packed(&distorted_rgba, 2, 1, PixelFormat::Rgba8, 8)?.validate()?;
    let metrics = MetricSet::from_csv("psnr:color,mse:color,mae:color")?;

    for metric in metrics.compare(&reference, &distorted)? {
        println!("{} = {} {}", metric.name, metric.score, metric.unit);
    }
    Ok(())
}
```

## Encode an imqraw Bundle

`imqraw` stores validated frames without compression. This is useful for
stdin/stdout pipelines, fixtures, and exact cross-tool interchange.

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
    let bundle = RawImageBundle::new(vec![reference]);
    let bytes = encode_imqraw_bundle(&bundle)?;
    let decoded = decode_imqraw_bundle(&bytes)?;

    assert_eq!(decoded.select_tag("ref")?.label.as_deref(), Some("reference"));
    Ok(())
}
```

## Image Crate Adapter

With `image-codecs`, decode common image files through the `image` crate:

```rust
use imq::Result;
use imq::adapters::image_crate;

fn main() -> Result<()> {
    let frame = image_crate::load_image_path("image.png")?;
    println!("{:?} {:?}", frame.dimensions(), frame.format());
    Ok(())
}
```

## Sample Project

A standalone sample project is available at:

```bash
cargo run --manifest-path samples/imq-crate-usage/Cargo.toml
```
