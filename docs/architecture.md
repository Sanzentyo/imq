# Architecture

`imq` is split into a deterministic Sans I/O core and optional edge adapters.

## Core

The core types live in `frame.rs`, `metrics/`, and `report.rs`.

```text
&[u8] / camera / decoder / Vec<u8>
      |
      v
FrameView<'a, Unchecked> -- validate() --> FrameView<'a, Validated>
      |
      v
MetricSet / Metric trait
      |
      v
MetricOutput / ComparisonReport / VideoReport
```

The core does not open files, spawn processes, read terminals, or allocate GPU resources. `FrameView` can borrow strided packed buffers or planar buffers. `FrameOwned` is the owned counterpart for adapters that naturally allocate, such as image decoding and rawvideo frame capture.

## Typestate

`FrameView<'a, Unchecked>` can be built from external buffers cheaply. A metric requires `FrameView<'a, Validated>`, so the comparison path cannot accidentally use unchecked dimensions/strides.

Validation checks:

- non-zero dimensions;
- plane count expected by `PixelFormat`;
- row-byte requirements against stride;
- minimum accessible byte length for the last row of each plane.

## ADTs and newtypes

The public API uses small ADTs instead of stringly typed values:

- `PixelFormat` covers packed RGB/RGBA/BGR/BGRA, 16-bit little-endian RGB/luma, float RGB/RGBA, planar YUV, and NV12;
- `ColorSpace`, `Transfer`, and `ColorRange` carry color metadata;
- `Dimensions` validates non-zero sizes;
- `MetricSpec` parses CLI/configuration strings into typed metric requests;
- `SampleDomain` separates luma, RGB/color, all stored components, and raw plane comparison.

## Optional adapters

`adapters::image_crate` decodes still images to `FrameOwned`, currently normalizing `DynamicImage` to RGBA8 for broad compatibility.

`video::ffmpeg` is an I/O adapter. It runs `ffprobe` for metadata and `ffmpeg` for RGBA rawvideo frames on stdout, then hands those frames to the same `FrameOwned`/`MetricSet` core used for still images.

`gpu` is another edge adapter. It receives validated borrowed frames, compacts rows if required, dispatches WGSL compute, and returns normal `MetricOutput`-compatible results.

`nn` converts validated frames into Burn tensors and provides extension traits for pretrained perceptual models.
