---
name: imq-cli
description: Use the `imq` command-line tool for full-reference image and video quality evaluation, and reference imqraw Rust/JavaScript/TypeScript examples when needed. Trigger when Codex needs to compare reference/distorted images or videos, compute PSNR/SSIM/MSE/RMSE/MAE/maxAE, produce JSON metric reports, list supported still-image formats, probe video metadata, extract video frames, or use imqraw from Rust, JS, or TS.
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
`--raw-pixel-format rgb8|rgba8|bgr8|bgra8|luma8`. Bit-packed grayscale and mask
stdin is accepted as `gray1-lsb`, `gray1-msb`, `gray2-lsb`, `gray2-msb`,
`gray4-lsb`, or `gray4-msb`; these inputs are decoded into `Luma8`. Use
`--raw-stride BYTES` when raw rows include padding.

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

## Other Commands

Subcommand aliases are available: `i` for `image`, `v` for `video`, `p` for
`preview`, `t` for `tui`, `fmt` for `formats`, `raw-pack`/`bundle` for `pack`,
`raw-info` for `bundle-info`, and `x`/`extract` for `extract-frame`.

Use `preview` or `tui` only when the user explicitly asks to inspect images in a
terminal UI. For Rust, JavaScript, or TypeScript `imqraw` library usage, read
`references/imqraw-library.md` for concrete examples. For GPU, benchmark, or NN
usage, refer to repository documentation instead of expanding those workflows in
this skill.

## Reporting Results

Report exact commands that passed or failed. If a command depends on missing external tools, say which executable is missing. For metric outputs, summarize the metric names, compared dimensions, frame count, and whether JSON/text output was requested.

## License

This skill is licensed under MIT OR Apache-2.0.
