---
name: imq-cli
description: Use the `imq` command-line tool for full-reference image and video quality evaluation. Trigger when Codex needs to compare reference/distorted images or videos, compute PSNR/SSIM/MSE/RMSE/MAE/maxAE, produce JSON metric reports, list supported still-image formats, probe video metadata, extract video frames, or run the TUI.
---

# imq CLI

Use `imq` for image/video quality checks. Prefer the installed `imq` binary when available; if the binary is missing, install from the GitHub repository with Cargo.

## Locate the Command

1. If `imq --help` works, use `imq`.
2. If `imq` is missing, run `cargo install --git https://github.com/Sanzentyo/imq.git --locked` and then use `imq`.
3. For an install with extra crate features, use `cargo install --git https://github.com/Sanzentyo/imq.git --locked --features <feature>`.

Useful environment checks:

```bash
imq --help
ffmpeg -version
ffprobe -version
```

## Compare Automatically

Prefer `compare` for ordinary use; it chooses still-image or video comparison
from file extensions. Use `--stats`/`-s` to add image statistics, color balance,
histograms, and visual tendency labels. With a single image plus `--stats`,
`compare` reports statistics without requiring a distorted image. Use `stats`
(`stat`/`s`) when only image statistics are needed.

```bash
imq compare reference.png distorted.png --metrics psnr,ssim,mse,mae,maxae
imq c reference.png distorted.png -s --format json
imq compare reference.mp4 distorted.mp4 --every 30 --max-frames 120
imq compare image.png -s --format yaml
imq stats image.png --format toml
cat image.png | imq stats - --format json
cat image.rgba | imq stats - --stdin-format raw --raw-width 1920 --raw-height 1080 --raw-pixel-format rgba8
imq pack --tag 1:reference --tag 2:candidate -o pair.imqraw reference.png distorted.png
cat pair.imqraw | imq image - - --stdin-format imqraw --stdin-reference-tag reference --stdin-distorted-tag candidate
imq bundle-info pair.imqraw --format json
```

## Compare Still Images

Use `image` when the caller explicitly wants the still-image path:

```bash
imq image reference.png distorted.png --metrics psnr,ssim,mse,mae,maxae
imq image reference.png distorted.png --metrics psnr:color,mse:all --json
imq image reference.png distorted.png --format yaml --output report.yaml
imq image reference.png distorted.png --format csv --sqlite reports.sqlite
```

Metric domains:

- `luma` or omitted: perceptual luma
- `color` or `rgb`: RGB/color channels, alpha ignored
- `all`: every stored component
- `planeN`: raw plane by index, such as `mse:plane0`

Use `imq formats` to list the image adapter's supported format hints.

Structured output is supported by `compare`, `stats`, `image`, `video`, `probe`,
and `formats` with `--format text|json|yaml|toml|csv`. `--json` is a
compatibility alias for `--format json`. Use `--output PATH` to write the
selected representation to a file, and `--sqlite PATH` to append reports, metric
rows, probe rows, image statistics, and format hints to a SQLite database.

Use `-` as an image input to read encoded image bytes from stdin. For raw packed
stdin bytes, pass `--stdin-format raw` with `--raw-width`, `--raw-height`, and
`--raw-pixel-format rgb8|rgba8|bgr8|bgra8|luma8`.

For multi-image stdin/stdout pipelines, use `imq pack` to create an uncompressed
lossless `imqraw` bundle with labels and tags:

```bash
imq pack --tag 1:reference --tag 2:candidate -o pair.imqraw reference.png distorted.png
imq bundle-info pair.imqraw
cat pair.imqraw | imq stats - --stdin-format imqraw --stdin-tag reference
cat pair.imqraw | imq image - - --stdin-format imqraw --stdin-reference-tag reference --stdin-distorted-tag candidate
```

Use `--stdin-index`/`--stdin-tag` when a command consumes one image from an
`imqraw` bundle. When both image arguments are `-`, use
`--stdin-reference-index`, `--stdin-reference-tag`, `--stdin-distorted-index`,
or `--stdin-distorted-tag`; defaults are index 0 for reference and index 1 for
distorted. Stdin can be used for both image arguments only with
`--stdin-format imqraw`.

## Compare Videos

Video commands require working `ffmpeg` and `ffprobe` executables. Use `video` for frame-by-frame RGBA decoding through ffmpeg:

```bash
imq video reference.mp4 distorted.mp4 --metrics psnr,ssim,mse --every 30 --max-frames 120
imq video reference.mp4 distorted.mp4 --width 1920 --height 1080 --json
imq video reference.mp4 distorted.mp4 --format toml --output video-report.toml
```

Rules:

- Specify `--width` and `--height` together.
- Use `--every N` to sample every Nth decoded frame.
- Use `--max-frames N` to cap runtime.
- Use `--ffmpeg`, `--ffprobe`, and `--stream` when the defaults are wrong.

Probe and extraction:

```bash
imq probe input.mp4 --json
imq extract-frame input.mp4 150 frame-150.png
```

## Preview Files

Use `preview` for terminal previews. Auto mode uses Sixel when the terminal
appears to support it and ANSI color blocks otherwise.

```bash
imq preview image.png
imq p image.png
imq preview --display kitty --cols 2 a.png b.png clip.mp4
imq preview --size 120x60 --fit cover image.png
imq preview -a image.png
imq preview --decode cpu clip.mp4
```

Multiple inputs are arranged as a montage. Use `--rows` or `--cols` to control
the layout. Omit `--size` to derive preview dimensions from terminal size and
display mode, or pass `--size WIDTHxHEIGHT`; known native graphics terminals and
Sixel terminals get a higher pixel-resolution default than ANSI blocks. Auto
mode prefers Kitty graphics protocol for terminals such as Ghostty, then Sixel
for terminals such as Windows Terminal, then iTerm2 inline images, then ANSI
blocks. Over SSH, native graphics are selected when terminal identity hints
such as `TERM`, `TERM_PROGRAM`, or `WT_SESSION` are visible; otherwise ANSI is
used so an image is still drawn. Use `--display kitty`, `--display sixel`,
`--display iterm2`, `IMQ_IMAGE_PROTOCOL=kitty|sixel|iterm2|ansi`, `IMQ_KITTY=1`,
`IMQ_SIXEL=1`, `IMQ_ITERM2=1`, or matching `IMQ_NO_*` variables to override
detection.
`--fit contain|cover|stretch` controls aspect handling. Video thumbnails are
extracted through `ffmpeg`. When a small source needs to fill the terminal-based
display area, preview uses nearest-neighbor enlargement so source pixels become
larger; shrink paths use `fast_image_resize`. Use `-a`/`--actual-size` to render at
the decoded source dimensions instead of fitting the terminal area. In the TUI,
the initial preview resolution is derived from the current preview panel area
and terminal font size. Pressing `a` toggles exact-pixel display. Pressing `+`
at the selected file's source dimensions switches to `max`, and `max` uses each
newly selected file's own maximum preview resolution.
`--decode auto` tries detected hardware decode backends and falls back to CPU
unless the build uses the `cpu-only` feature.
Useful short options include `-m` for metrics, `-j` for JSON output, `preview
-s` for size, `preview -D` for display mode, `preview -a` for exact pixel size,
and `tui -C` for preview cache size.

Subcommand aliases are available: `i` for `image`, `v` for `video`, `p` for
`preview`, `t` for `tui`, `fmt` for `formats`, `raw-pack`/`bundle` for `pack`,
`raw-info` for `bundle-info`, and `x`/`extract` for
`extract-frame`.

## TUI

The `tui` subcommand is enabled by default:

```bash
imq tui reference.png distorted-a.png distorted-b.png
imq tui
imq tui ./images
imq tui --preview-cache 64 ./images
```

The TUI includes a file browser. Use it to move through folders and set
one reference image plus multiple comparison targets without restarting. When
testing non-interactively, run it in a PTY and send `q` or Esc to exit.
Vim-style navigation is supported: `j`/`k` move, `h` goes to the parent
directory, `l` opens/selects, and `g`/`G` jump to the first/last entry. Use `r`
to set the reference, `Space` or `d` to toggle targets, `v` to cycle
current/side-by-side/diff previews, `/` to filter by name, `e` to filter to the
selected extension, `u` to clear filters, `i` to toggle directories, `o` to
toggle non-media files, and `s` to cycle sort order. Use `+`/`-` to adjust
preview resolution, `f` to cycle fit mode, and `a` to toggle exact-pixel preview
display. The TUI shows selected reference/target thumbnails and a multi-target
comparison table. It uses native Kitty/Sixel/iTerm2 image protocols when the
terminal reports support, falls back to half-block rendering otherwise, and
keeps decoded previews in memory according to `--preview-cache`.

## GPU, Benchmark, and NN Features

GPU RGBA8 error stats are exposed as examples, not CLI subcommands. The GPU
path computes MSE, RMSE, PSNR, MAE, and maxAE-compatible stats in one dispatch:

```bash
cargo run --quiet --features gpu --example gpu_mse -- reference.png distorted.png
cargo run --release --example benchmark --features gpu -- --width 3840 --height 2160 --iterations 3
```

The benchmark prints separate CPU error metrics, optimized combined CPU error
metrics, CPU default metrics, and GPU RGBA8 error stats when `gpu` is enabled.

Burn/NN support is library scaffolding. Validate availability with:

```bash
cargo check --features nn-burn-ndarray --lib
cargo check --features nn-burn-wgpu --lib
```

Do not imply bundled pretrained weights exist; applications must supply model definitions/checkpoints.

## imqraw Library Feature

The raw bundle API is always available in the library as `imq::imqraw` and
through `encode_imqraw_bundle` / `decode_imqraw_bundle`. It is Sans I/O and
stores little-endian metadata plus verbatim frame planes, so it is suitable for
cross-platform stdin/stdout exchange without codec artifacts.

For crate-level usage, refer to and run the bundled example:

```bash
cargo run --example imqraw_bundle
```

The optional `imqraw-image` feature enables conversion helpers from common
`image` crate types (`DynamicImage`, `RgbaImage`, `RgbImage`) into imqraw
records. This feature is disabled by default.

## imqraw Browser Distribution

For browser or Three.js capture workflows, prefer the GitHub Pages ESM build:

```js
import { init, encodeRgba8, encodeThreeRenderer } from "https://sanzentyo.github.io/imq/imqraw/v0.1.0/imqraw.js";

await init();
const bytes = encodeRgba8(rgbaBytes, width, height, {
  label: "frame-0001",
  tags: ["threejs", "reference"],
});
```

Use fixed `imqraw/vX.Y.Z/` URLs for reproducible work and `imqraw/latest/` only
for quick experiments. Versioned directories are immutable; release workflows
refresh `latest` and attach the same generated files to GitHub Releases. The
`encodeThreeRenderer(renderer, options)` helper reads RGBA8 pixels from the
renderer's WebGL context, flips WebGL's bottom-left origin, and encodes an
`imqraw` bundle.

## Reporting Results

Report exact commands that passed or failed. If a command depends on missing external tools, say which executable is missing. For metric outputs, summarize the metric names, compared dimensions, frame count, and whether JSON/text output was requested.

## License

This skill is licensed under MIT OR Apache-2.0.
