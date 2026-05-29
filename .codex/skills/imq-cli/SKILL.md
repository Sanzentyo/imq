---
name: imq-cli
description: Use the `imq` command-line tool for full-reference image and video quality evaluation. Trigger when Codex needs to compare reference/distorted images or videos, compute PSNR/SSIM/MSE/RMSE/MAE/maxAE, produce JSON metric reports, list supported still-image formats, probe video metadata, extract video frames, or run the TUI.
---

# imq CLI

Use `imq` for image/video quality checks. Prefer the installed `imq` binary when available; inside the source repo, install it with Cargo if the binary is missing.

## Locate the Command

1. If `imq --help` works, use `imq`.
2. If `imq` is missing and the current directory contains this crate's `Cargo.toml`, run `cargo install --path . --locked` and then use `imq`.
3. If installation is not appropriate, use `cargo run --quiet --bin imq --`.
4. For optional features, use `cargo run --quiet --features <feature> --bin imq --`.

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
`--raw-pixel-format rgb8|rgba8|bgr8|bgra8|luma8`. Stdin can be used for only one
image argument per command.

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
imq preview --decode cpu clip.mp4
```

Multiple inputs are arranged as a montage. Use `--rows` or `--cols` to control
the layout. Omit `--size` to derive preview dimensions from terminal size and
display mode, or pass `--size WIDTHxHEIGHT`; known native graphics terminals and
Sixel terminals get a higher pixel-resolution default than ANSI blocks. Auto
mode prefers Kitty graphics protocol for terminals such as Ghostty, then Sixel,
then ANSI blocks. Use `--display kitty`, `--display sixel`, `IMQ_KITTY=1`,
`IMQ_NO_KITTY=1`, `IMQ_SIXEL=1`, or `IMQ_NO_SIXEL=1` to override detection.
`--fit contain|cover|stretch` controls aspect handling. Video thumbnails are
extracted through `ffmpeg`. Preview generation does not upscale past source
dimensions. In the TUI, pressing `+` at the selected file's source dimensions
switches to `max`, and `max` uses each newly selected file's own maximum preview
resolution. `--decode
auto` tries detected hardware decode backends and falls back to CPU unless the
build uses the `cpu-only` feature.

Subcommand aliases are available: `i` for `image`, `v` for `video`, `p` for
`preview`, `t` for `tui`, `fmt` for `formats`, and `x`/`extract` for
`extract-frame`.

## TUI

The `tui` subcommand is enabled by default:

```bash
imq tui reference.png distorted.png
imq tui
imq tui ./images
```

The TUI includes a file browser. Use it to move through folders and set
reference/distorted images without restarting. When testing non-interactively,
run it in a PTY and send `q` or Esc to exit. Vim-style navigation is supported:
`j`/`k` move, `h` goes to the parent directory, `l` opens/selects, and `g`/`G`
jump to the first/last entry. Use `+`/`-` to adjust preview resolution and `f`
to cycle fit mode.

## GPU and NN Features

GPU MSE is exposed as an example, not a CLI subcommand:

```bash
cargo run --quiet --features gpu --example gpu_mse -- reference.png distorted.png
```

Burn/NN support is library scaffolding. Validate availability with:

```bash
cargo check --features nn-burn-ndarray --lib
cargo check --features nn-burn-wgpu --lib
```

Do not imply bundled pretrained weights exist; applications must supply model definitions/checkpoints.

## Reporting Results

Report exact commands that passed or failed. If a command depends on missing external tools, say which executable is missing. For metric outputs, summarize the metric names, compared dimensions, frame count, and whether JSON/text output was requested.

## License

This skill is licensed under MIT OR Apache-2.0.
