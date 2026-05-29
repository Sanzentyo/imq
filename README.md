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
`--raw-pixel-format rgb8|rgba8|bgr8|bgra8|luma8`. Stdin can be used for only one
image argument per command.

Extract one decoded video frame:

```bash
cargo run --bin imq -- extract-frame input.mov 150 frame-150.png
```

Run the TUI:

```bash
cargo run --bin imq -- tui reference.png distorted.png
cargo run --bin imq -- tui
cargo run --bin imq -- tui ./images
cargo run --bin imq -- tui --preview-cache 64 ./images
```

The TUI shows a colorized metric table and an image file browser. Use the
browser to move through folders and assign images to the reference/distorted
slots without restarting the program. It supports Vim-style navigation:
`j`/`k` move, `h` goes to the parent directory, `l` opens/selects, and `g`/`G`
jump to the first/last entry.

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
cargo run --bin imq -- preview --decode cpu clip.mp4
```

When `--size` is omitted, `preview` derives a per-item preview size from the
terminal dimensions, display mode, and requested montage rows/columns. Known
native graphics terminals and Sixel terminals get a higher pixel-resolution
default than ANSI block rendering, without an artificial fixed maximum. Auto
mode prefers Kitty graphics protocol for terminals such as Ghostty, then Sixel,
then ANSI blocks. Use `--display kitty`, `--display sixel`, `IMQ_KITTY=1`,
`IMQ_NO_KITTY=1`, `IMQ_SIXEL=1`, or `IMQ_NO_SIXEL=1` to override auto
detection. Fit modes are `contain`, `cover`, and `stretch`. When `preview` needs
to enlarge a small source to fill the terminal-derived display area, it uses
nearest-neighbor scaling so source pixels become larger instead of blurred.
Shrink paths use `fast_image_resize`. In the TUI, the initial preview
resolution is derived from the current preview panel area and terminal font
size; use `+`/`-` to change preview resolution and `f` to cycle the fit mode.
Pressing `+` at the source dimensions switches to `max`, which keeps using each
selected file's own maximum preview resolution.
The TUI queries the terminal for native image support and renders through Kitty,
Sixel, or iTerm2 protocols when available, falling back to half-blocks. Decoded
previews are resized with `fast_image_resize` and cached in memory; use
`--preview-cache N` to tune how many previews are kept.

Common subcommand aliases are available: `i` for `image`, `v` for `video`, `p`
for `preview`, `t` for `tui`, `fmt` for `formats`, and `x`/`extract` for
`extract-frame`.

## Codex skill

This repository includes a repo-scoped Codex skill at `.codex/skills/imq-cli/`.
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

## GPU path

With `--features gpu`, `GpuContext::mse_rgba8` dispatches a WGSL compute shader. It compacts strided RGBA8 rows, uploads a `u32` per pixel, reduces per-workgroup squared error on the GPU, then performs the final accumulation on CPU readback.

```rust
use imq::gpu::GpuContext;

let gpu = GpuContext::new()?;
let result = gpu.mse_rgba8(&reference, &distorted)?;
println!("GPU MSE: {}", result.mse);
# Ok::<(), imq::Error>(())
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
- GPU acceleration currently covers RGBA8 MSE. PSNR can be derived from MSE, and more kernels can follow the same layout.
- Neural metrics provide Burn integration and feature-distance scaffolding; model definitions/checkpoints are caller-supplied.
- The `ffmpeg` module uses external executables by path. The core library remains pure Rust and Sans I/O.
