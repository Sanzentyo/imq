---
name: imq-cli
description: Use the `imq` and supplemental `imq-quality` command-line tools and Rust crate for full-reference image/video quality evaluation, and reference imqraw Rust/JavaScript/TypeScript examples when needed. Trigger when Codex needs to compare reference/distorted images or videos, compute PSNR/SSIM/windowed SSIM/MS-SSIM/MSE/RMSE/MAE/maxAE, produce JSON metric reports, create diff/heatmap images, enforce CI metric gates, list supported still-image formats, probe video metadata, extract video frames, use imq as a Rust library, or use imqraw from Rust, JS, or TS.
---

# imq CLI

Use `imq` for standard image/video quality checks and `imq-quality` for advanced helper workflows such as diff images, CI gates, and external metric command wrapping. Prefer installed binaries when available; if a binary is missing, install from the GitHub repository with Cargo.

## Locate the Commands

1. If `imq --help` works, use `imq`.
2. If `imq-quality --help` works, use it for advanced helpers.
3. If a binary is missing, run `cargo install --git https://github.com/Sanzentyo/imq.git --locked` and then use the installed command.
4. For an install with extra crate features, use `cargo install --git https://github.com/Sanzentyo/imq.git --locked --features <feature>`.

Useful environment checks:

```bash
imq --help
imq-quality --help
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
imq compare reference.png distorted.png --metrics psnr,ssim,wssim,ms-ssim,mse,mae,maxae
imq c reference.png distorted.png -s --format json
imq compare reference.mp4 distorted.mp4 --every 30 --max-frames 120
imq compare ref.mp4 out.mp4 --video-frames 0:1,'(30,31,30)' --format json
imq compare ssh://host/tmp/ref.png ssh://host/tmp/out.png
imq compare ssh://host/tmp/ref.mp4 ssh://host/tmp/out.mp4 --video-frame 120
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
imq image reference.png distorted.png --metrics psnr,ssim,wssim,ms-ssim,mse,mae,maxae
imq image reference.png distorted.png --metrics psnr:color,mse:all --json
imq image reference.imqraw candidate.imqraw --metrics psnr:rgb-visible,psnr:rgb-nonblack-interior2px --selected-metric rgb-visible --fail-under 34 --max-selected-channel-delta 4
imq image reference.png distorted.png --format yaml --output report.yaml
imq image reference.png distorted.png --format csv --sqlite reports.sqlite
```

Metric selection:

- `ssim`: global luma SSIM over the whole frame.
- `wssim`: non-overlapping 8x8 windowed luma SSIM. Use this when local structure changes should count more than a whole-frame aggregate. Aliases are `windowed-ssim`, `windowed_ssim`, `ssim-windowed`, and `ssim_windowed`.
- `ms-ssim`: deterministic luma multi-scale SSIM. Aliases are `ms_ssim` and `msssim`.
- `mse`, `rmse`, `psnr`, `mae`, and `maxae`: error metrics over the selected sample domain.

Metric domains:

- `luma` or omitted: perceptual luma
- `color` or `rgb`: RGB/color channels, alpha ignored
- `all`: every stored component
- `planeN`: raw plane by index, such as `mse:plane0`
- render-parity domains: `rgba`, `rgb-all`, `rgb-visible`, `rgb-opaque`,
  `rgb-nonblack`, `rgb-interiorNpx`, `rgb-visible-interiorNpx`,
  `rgb-nonblack-interiorNpx`; `gray`, `binary`, `hsv`, and `hsva` can use the
  same mask suffixes. Gate with `--selected-metric`, `--fail-under`,
  `--max-selected-channel-delta`, `--max-alpha-delta`,
  `--max-alpha-mismatches`, or `--max-alpha-mismatches-beyond-one-lsb`.

`ssim`, `wssim`, and `ms-ssim` currently operate on luma even when a domain suffix is present.

## Advanced Helpers with imq-quality

Generate diff images or heatmaps:

```bash
imq-quality diff reference.png distorted.png diff.png --mode heatmap
imq-quality diff reference.png distorted.png absolute.png --mode absolute-rgb --scale 4
imq-quality diff reference.png distorted.png signed.png --mode signed-luma --scale 8
```

Run CI threshold gates. The command exits non-zero when any rule fails:

```bash
imq-quality gate reference.png distorted.png --metrics psnr,ssim,wssim,ms-ssim,mae --rule 'psnr>=40' --rule 'wssim>=0.98' --rule 'mae<=0.01'
```

Wrap an external metric command. Use `{reference}` and `{distorted}` placeholders in repeated `--arg` values:

```bash
imq-quality external reference.mp4 distorted.mp4 --program ffmpeg-quality-metrics --arg '{reference}' --arg '{distorted}' --metric-name vmaf --unit unitless --direction higher
```

Use `imq formats` to list the image adapter's supported format hints.

Structured output is supported by `compare`, `stats`, `image`, `video`, `probe`,
and `formats` with `--format text|json|yaml|toml|csv`. `--json` is a
compatibility alias for `--format json`. Use `--output PATH` to write the
selected representation to a file, and `--sqlite PATH` to append reports, metric
rows, probe rows, image statistics, and format hints to a SQLite database.

Use `-` as an image input to read encoded image bytes from stdin. For raw packed
stdin bytes, pass `--stdin-format raw` with `--raw-width`, `--raw-height`, and
`--raw-pixel-format rgb8|rgba8|bgr8|bgra8|luma8|hsv8|hsva8|binary1-lsb|binary1-msb`. Bit-packed grayscale
stdin is accepted as `gray1-lsb`, `gray1-msb`, `gray2-lsb`, `gray2-msb`,
`gray4-lsb`, or `gray4-msb`; these inputs are decoded into `Luma8`. Native
`binary1-*` inputs stay bit-packed in `imqraw`. Use `--raw-stride BYTES` when
raw rows include padding.

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
imq video reference.mp4 distorted.mp4 --metrics psnr,ssim,wssim,ms-ssim,mse --every 30 --max-frames 120
imq video reference.mp4 distorted.mp4 --width 1920 --height 1080 --json
imq video reference.mp4 distorted.mp4 --format toml --output video-report.toml
imq compare reference.mp4 distorted.mp4 --video-frames 0:1,30:31
imq compare reference.mp4 distorted.mp4 --video-frames '(0,1,0),(2,3,2)'
```

Rules:

- Specify `--width` and `--height` together.
- Use `--every N` to sample every Nth decoded frame.
- Use `--max-frames N` to cap runtime.
- Use `--ffmpeg`, `--ffprobe`, and `--stream` when the defaults are wrong.
- Use `--video-frame N` or ordered `--video-frames 0:1,30:31` on `compare`
  when specific frame pairs should be compared. Tuple syntax
  `(ref,dist)` and `(ref,dist,label_frame)` is also accepted.
- Remote inputs use `ssh://host/absolute/path`. Remote images stream over SSH
  by default. Remote videos require `--video-frame` or `--video-frames`; full
  remote video comparison is intentionally not implicit.
- Remote copy never happens as fallback. Use explicit
  `--remote-transfer copy-input|copy-frame|copy-source` only when copying is
  intended.

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
terminal UI. For using `imq` as a Rust crate, read
`references/imq-rust-crate.md`. For Rust, JavaScript, or TypeScript `imqraw`
bundle usage, read `references/imqraw-library.md`. For GPU, benchmark, or NN
usage, refer to repository documentation instead of expanding those workflows in
this skill.

For video aggregation or timestamp-aware pairing in Rust, use
`imq::video_analysis::{aggregate_video_report, pair_frames_by_timestamp}`. Use
timestamp pairing when VFR, duplicated, or dropped frames make decode-order
pairing unreliable.

## Reporting Results

Report exact commands that passed or failed. If a command depends on missing external tools, say which executable is missing. For metric outputs, summarize the metric names, compared dimensions, frame count, threshold rules, and whether JSON/text output was requested. For `wssim`, also mention that it is non-overlapping windowed luma SSIM and include the reported window count when JSON/details are available.

## License

This skill is licensed under MIT OR Apache-2.0.
