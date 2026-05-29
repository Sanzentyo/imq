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

## Compare Still Images

Use `image` for decoded still images:

```bash
imq image reference.png distorted.png --metrics psnr,ssim,mse,mae,maxae
imq image reference.png distorted.png --metrics psnr:color,mse:all --json
```

Metric domains:

- `luma` or omitted: perceptual luma
- `color` or `rgb`: RGB/color channels, alpha ignored
- `all`: every stored component
- `planeN`: raw plane by index, such as `mse:plane0`

Use `imq formats` to list the image adapter's supported format hints.

## Compare Videos

Video commands require working `ffmpeg` and `ffprobe` executables. Use `video` for frame-by-frame RGBA decoding through ffmpeg:

```bash
imq video reference.mp4 distorted.mp4 --metrics psnr,ssim,mse --every 30 --max-frames 120
imq video reference.mp4 distorted.mp4 --width 1920 --height 1080 --json
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
imq preview --display sixel --cols 2 a.png b.png clip.mp4
imq preview --size 120x60 --fit cover image.png
imq preview --decode cpu clip.mp4
```

Multiple inputs are arranged as a montage. Use `--rows` or `--cols` to control
the layout. Omit `--size` to derive preview dimensions from the terminal size,
or pass `--size WIDTHxHEIGHT`; `--fit contain|cover|stretch` controls aspect
handling. Video thumbnails are extracted through `ffmpeg`; `--decode auto` tries
detected hardware decode backends and falls back to CPU unless the build uses
the `cpu-only` feature.

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
