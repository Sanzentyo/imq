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

## TUI

The `tui` subcommand is enabled by default:

```bash
imq tui reference.png distorted.png
```

When testing non-interactively, run it in a PTY and send `q` or Esc to exit.

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
