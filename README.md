# imq

`imq` is a Rust-first image/video quality evaluation crate and CLI. The design is **Sans I/O at the core**: the library compares borrowed image/video frames supplied by the caller, while decoding, filesystem access, subprocesses, terminal UI, GPU dispatch, and neural-network backends live in optional modules and features.

The crate is intended as a practical starting point for full-reference evaluation pipelines:

- still-image metrics: MSE, RMSE, PSNR, MAE, max absolute error, global SSIM;
- frame and video comparison through `ffmpeg`/`ffprobe` rawvideo pipes;
- `image` crate adapters for broad still-image decoding;
- zero-copy borrowed inputs for `&[u8]`, camera buffers, planar YUV, NV12, packed RGB/RGBA/BGR/BGRA, and owned `Vec<u8>`;
- optional `wgpu` compute kernel for RGBA8 MSE;
- optional Burn tensor adapter and feature-distance scaffolding for LPIPS/DISTS/MUSIQ/CLIP-IQA-like research metrics;
- CLI and optional TUI.

## Feature flags

| Feature | Purpose |
| --- | --- |
| `std` | Standard-library support. |
| `serde` | JSON reports and serializable structs. |
| `image-codecs` | Decode still images through the `image` crate. |
| `imqraw-image` | Conversion helpers from `image` crate image types into imqraw records. Disabled by default. |
| `ffmpeg` | Shell out to `ffmpeg`/`ffprobe` and stream RGBA frames from stdout. |
| `preview` | Terminal image/video previews, using `ffmpeg` for video thumbnails. |
| `cpu-only` | Disable hardware decode attempts for preview thumbnail extraction. |
| `gpu` | `wgpu` compute kernels. |
| `nn-burn` | Burn tensor adapters and NN metric scaffolding. |
| `nn-burn-wgpu` | Burn with WGPU backend features. |
| `nn-burn-ndarray` | Burn with ndarray backend features. |
| `cli` | `imq` binary. |
| `tui` | Ratatui/crossterm interactive frontend. |

Default features are `std`, `serde`, `image-codecs`, `cli`, `tui`, and `preview`.
`imqraw-image` is intentionally not a default feature; the raw bundle format
itself is available without codec adapter conversions.

## CLI examples

Compare images or videos with automatic handling based on file extensions:

```bash
cargo run --bin imq -- compare reference.png distorted.webp --metrics psnr,ssim,mse,mae,maxae
cargo run --bin imq -- compare reference.mp4 distorted.mp4 --every 30 --max-frames 120
cargo run --bin imq -- c image.png -s --format yaml
cargo run --bin imq -- compare reference.png distorted.webp -s --format json
cargo run --bin imq -- stats image.png --format toml
cat image.png | cargo run --bin imq -- stats - --format json
cat image.rgba | cargo run --bin imq -- stats - --stdin-format raw --raw-width 1920 --raw-height 1080 --raw-pixel-format rgba8
cargo run --bin imq -- pack --tag 1:reference --tag 2:candidate -o pair.imqraw reference.png distorted.png
cat pair.imqraw | cargo run --bin imq -- image - - --stdin-format imqraw --stdin-reference-tag reference --stdin-distorted-tag candidate
cargo run --bin imq -- bundle-info pair.imqraw --format json
```

Explicit still-image comparison remains available:

```bash
cargo run --bin imq -- image reference.png distorted.webp --metrics psnr,ssim,mse,mae,maxae
cargo run --bin imq -- image reference.png distorted.webp --metrics psnr:color,mse:all --json
cargo run --bin imq -- image reference.png distorted.webp --format yaml --output report.yaml
cargo run --bin imq -- image reference.png distorted.webp --format csv --sqlite reports.sqlite
```

Explicit video comparison by decoding RGBA frames with `ffmpeg`:

```bash
cargo run --bin imq -- video reference.mp4 distorted.mp4 --every 30 --max-frames 120
cargo run --bin imq -- video reference.mp4 distorted.mp4 --width 1920 --height 1080 --json
cargo run --bin imq -- video reference.mp4 distorted.mp4 --format toml --output video-report.toml
```

Structured output is available on `compare`, `stats`, `image`, `video`, `probe`,
and `formats` with `--format text|json|yaml|toml|csv`; `--json` is kept as an
alias for `--format json`. Use `--output PATH` to write the selected
representation to a file. Use `--sqlite PATH` to append reports, metric rows,
probe rows, image statistics, and format hints to SQLite tables.

Use `-` as an image input to read encoded image bytes from stdin. For raw packed
stdin bytes, pass `--stdin-format raw` with `--raw-width`, `--raw-height`, and
`--raw-pixel-format rgb8|rgba8|bgr8|bgra8|luma8`. For multi-image stdin/stdout
pipelines, `imq pack` writes an `imqraw` bundle: a little-endian, uncompressed,
lossless raw container containing one or more validated frames, labels, and
tags. `--stdin-format imqraw` reads that bundle; `--stdin-index`/`--stdin-tag`
select one image for `stats`, while `--stdin-reference-index`,
`--stdin-reference-tag`, `--stdin-distorted-index`, and
`--stdin-distorted-tag` select the two images when both image arguments are `-`.
Stdin can be used for both image arguments only with `--stdin-format imqraw`.

Extract one decoded video frame:

```bash
cargo run --bin imq -- extract-frame input.mov 150 frame-150.png
```

Run the TUI:

```bash
cargo run --bin imq -- tui reference.png distorted-a.png distorted-b.png
cargo run --bin imq -- tui
cargo run --bin imq -- tui ./images
cargo run --bin imq -- tui --preview-cache 64 ./images
```

The TUI shows a colorized metric table and an image file browser. Use the
browser to move through folders, set one reference image, and toggle multiple
comparison targets without restarting the program. It supports Vim-style navigation:
`j`/`k` move, `h` goes to the parent directory, `l` opens/selects, and `g`/`G`
jump to the first/last entry. Use `r` to set the reference, `Space` or `d` to
toggle targets, `v` to cycle current/side-by-side/diff previews, `/` to filter
by name, `e` to filter to the selected extension, `u` to clear filters, `i` to
toggle directories, `o` to toggle non-media files, and `s` to cycle sort order.
Selected reference/target images are also shown as low-resolution, aspect-ratio
preserving thumbnails next to the comparison table.

List still-image formats exposed by the image adapter:

```bash
cargo run --bin imq -- formats
```

Preview images or video thumbnails in the terminal:

```bash
cargo run --bin imq -- preview image.png
cargo run --bin imq -- p image.png
cargo run --bin imq -- preview --display kitty --cols 2 a.png b.png clip.mp4
cargo run --bin imq -- preview --size 120x60 --fit cover image.png
cargo run --bin imq -- preview -a image.png
cargo run --bin imq -- preview --decode cpu clip.mp4
```

When `--size` is omitted, `preview` derives a per-item preview size from the
terminal dimensions, display mode, and requested montage rows/columns. Known
native graphics terminals and Sixel terminals get a higher pixel-resolution
default than ANSI block rendering, without an artificial fixed maximum. Auto
mode prefers Kitty graphics protocol for terminals such as Ghostty, then Sixel
for terminals such as Windows Terminal, then iTerm2 inline images, then ANSI
blocks. This works over SSH when the terminal identity is visible through
`TERM`, `TERM_PROGRAM`, `WT_SESSION`, or similar environment hints; if SSH hides
the local terminal identity, auto mode falls back to ANSI so an image is still
drawn. Use `--display kitty`, `--display sixel`, `--display iterm2`,
`IMQ_IMAGE_PROTOCOL=kitty|sixel|iterm2|ansi`, `IMQ_KITTY=1`, `IMQ_SIXEL=1`,
`IMQ_ITERM2=1`, or the matching `IMQ_NO_*` variables to override auto detection.
Fit modes are `contain`, `cover`, and `stretch`. When `preview` needs
to enlarge a small source to fill the terminal-derived display area, it uses
nearest-neighbor scaling so source pixels become larger instead of blurred.
Shrink paths use `fast_image_resize`. Use `-a`/`--actual-size` to render the decoded
source pixels at their exact dimensions instead of fitting the terminal area. In
the TUI, the initial preview resolution is derived from the current preview
panel area and terminal font size; use `+`/`-` to change preview resolution, `f`
to cycle the fit mode, and `a` to toggle exact-pixel display. Common option
shorts include `-m` for metrics, `preview -s` for size, `preview -D` for display
mode, `-j` for JSON output, and `tui -C` for preview cache size.
Pressing `+` at the source dimensions switches to `max`, which keeps using each
selected file's own maximum preview resolution.
The TUI queries the terminal for native image support and renders through Kitty,
Sixel, or iTerm2 protocols when available, falling back to half-blocks. Decoded
previews are resized with `fast_image_resize` and cached in memory; use
`--preview-cache N` to tune how many previews are kept.

Common subcommand aliases are available: `i` for `image`, `v` for `video`, `p`
for `preview`, `t` for `tui`, `fmt` for `formats`, `raw-pack`/`bundle` for
`pack`, `raw-info` for `bundle-info`, and `x`/`extract` for `extract-frame`.

## Agent skill

This repository includes a repo-scoped agent skill at `.agents/skills/imq-cli/`.
It guides agents through installing and using the `imq` CLI for image and video
quality checks.

## Library: Sans I/O comparison

```rust
use imq::{FrameView, MetricSet, PixelFormat};

let reference_rgba: &[u8] = &[0, 0, 0, 255, 255, 255, 255, 255];
let distorted_rgba: &[u8] = &[0, 0, 0, 255, 250, 250, 250, 255];

let reference = FrameView::packed(reference_rgba, 2, 1, PixelFormat::Rgba8, 2 * 4)?.validate()?;
let distorted = FrameView::packed(distorted_rgba, 2, 1, PixelFormat::Rgba8, 2 * 4)?.validate()?;

let metrics = MetricSet::from_csv("psnr:color,ssim,mse")?;
let report = metrics.compare(&reference, &distorted)?;
# Ok::<(), imq::Error>(())
```

## Raw cameras, YUV, and low-copy input

The core frame model is borrowed:

- `FrameView<'a, Unchecked>` points at caller-owned planes;
- `.validate()` checks dimensions, plane count, strides, and minimum accessible row lengths, producing `FrameView<'a, Validated>`;
- `FrameOwned` owns `Vec<u8>` planes and re-borrows to `FrameView<'_, Validated>`.

Examples:

```rust
use imq::{FrameView, PlaneView, PixelFormat};

// Planar YUV420p borrowed from a camera/video decoder.
let y = PlaneView::new(y_plane, y_stride);
let u = PlaneView::new(u_plane, u_stride);
let v = PlaneView::new(v_plane, v_stride);
let frame = FrameView::yuv_planar(width, height, PixelFormat::Yuv420p8, y, u, v)?.validate()?;
# Ok::<(), imq::Error>(())
```

```rust
use imq::{FrameView, PlaneView};

// NV12 borrowed from a hardware camera path.
let frame = FrameView::nv12(width, height, PlaneView::new(y, y_stride), PlaneView::new(uv, uv_stride))?.validate()?;
# Ok::<(), imq::Error>(())
```

## imqraw bundle format

The `imqraw` module provides Sans I/O encode/decode helpers for a compact raw
bundle designed for pipes and cross-platform tool interchange. It stores frames
without compression, so codec artifacts and encoder settings cannot affect
metrics. Each bundle begins with `IMQRAW1\n`, then little-endian metadata and
verbatim plane bytes. Multiple images, labels, and tags are supported.

```rust
use imq::{FrameOwned, PixelFormat, RawImageBundle, RawImageRecord};

let frame = FrameOwned::packed_tight(vec![0, 0, 0, 255], 1, 1, PixelFormat::Rgba8)?;
let bundle = RawImageBundle::new(vec![RawImageRecord::new(
    Some("reference".to_string()),
    vec!["ref".to_string()],
    frame,
)]);
let bytes = imq::encode_imqraw_bundle(&bundle)?;
let decoded = imq::decode_imqraw_bundle(&bytes)?;
assert_eq!(decoded.select_tag("ref")?.label.as_deref(), Some("reference"));
# Ok::<(), imq::Error>(())
```

With `--features imqraw-image`, helper constructors are enabled for common
`image` crate types such as `DynamicImage`, `RgbaImage`, and `RgbImage`. That
feature is opt-in so the raw container can stay independent from codec adapters.

## GPU path

With `--features gpu`, `GpuContext::error_stats_rgba8` dispatches a WGSL
compute shader for RGBA8 frames. Tight RGBA8 buffers are uploaded directly;
strided rows are compacted only when required. The shader reduces squared
error, absolute error, and max absolute error in one pass, so MSE, RMSE, PSNR,
MAE, and maxAE can be reported from one dispatch.

```rust
use imq::gpu::GpuContext;

let gpu = GpuContext::new()?;
let stats = gpu.error_stats_rgba8(&reference, &distorted)?;
for metric in stats.into_normalized_metric_outputs() {
    println!("{}: {}", metric.name, metric.score);
}
# Ok::<(), imq::Error>(())
```

Run the built-in synthetic benchmark with:

```bash
cargo run --release --example benchmark --features gpu -- --width 3840 --height 2160 --iterations 3
```

## Burn / NN metrics

With `--features nn-burn`, `frame_to_nchw_rgb_tensor` converts validated frames to `[1, 3, H, W]` Burn tensors. `BurnL2Metric` demonstrates tensor-space metrics, and `LpipsLikeMetric` accepts a caller-provided `BurnFeatureExtractor` so you can plug in LPIPS, DISTS, MUSIQ, CLIP-IQA, or your own trained model.

This crate deliberately does not bundle pretrained weights. That keeps licensing and provenance explicit and lets applications choose Burn backends and checkpoints.

## Crate structure

```text
src/
  frame.rs              borrowed/owned frame types, formats, typestate validation
  metrics/              Sans I/O metrics and sample iteration
  adapters/             image crate/raw adapter helpers
  video/ffmpeg.rs       ffprobe + ffmpeg rawvideo stdout pipe
  gpu/                  optional wgpu context + WGSL kernels
  nn/                   optional Burn tensor adapters and feature metric scaffolding
  bin/imq.rs            CLI/TUI frontend
```

See `docs/architecture.md`, `docs/metrics.md`, `docs/ffmpeg.md`, and `docs/nn-burn.md` for implementation notes.

## Current limitations

- SSIM is a global luma implementation, not windowed MS-SSIM.
- GPU acceleration currently covers RGBA8 MSE, RMSE, PSNR, MAE, and maxAE error statistics. SSIM still runs on CPU.
- Neural metrics provide Burn integration and feature-distance scaffolding; model definitions/checkpoints are caller-supplied.
- The `ffmpeg` module uses external executables by path. The core library remains pure Rust and Sans I/O.
