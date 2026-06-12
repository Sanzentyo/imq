# imq remote input / ordered video frame-pair design

## Purpose

`imq compare` and `imq preview` can work with local paths and SSH inputs. Remote
copy is never an automatic fallback: the default mode streams bytes over SSH and
returns an error if streaming fails.

Remote full-video comparison is intentionally not enabled by default. Remote
video input requires `--video-frame` or `--video-frames`; those frames are
extracted and compared through the existing image comparison pipeline.

## Inputs

Use SSH URIs for remote inputs:

```bash
imq preview ssh://host/home/me/image.png
imq compare ssh://host/home/me/ref.png ssh://host/home/me/out.png
imq compare ssh://alice@example.com:2222/home/alice/ref.png ./out.png
```

`--ssh HOST` is a shorthand for absolute non-URI paths:

```bash
imq preview --ssh host /home/me/image.png
imq compare --ssh host /home/me/ref.png /home/me/out.png
```

URI form is preferred when inputs may come from different hosts or a mix of
local and remote locations.

## Remote Transfer

```bash
--remote-transfer stream      # default
--remote-transfer copy-input
--remote-transfer copy-frame
--remote-transfer copy-source
```

`stream` uses `ssh` stdout and performs no copy. `copy-input` is for still image
inputs only. `copy-frame` is explicit frame-only transfer for remote video frame
workflows. `copy-source` copies whole remote sources and should be used only
when that cost is intended.

There is no `auto` mode. Stream failure does not fall back to `scp`.

## Video Frames

`--video-frame N` is shorthand for `--video-frames N`.

Recommended syntax:

```bash
imq compare ref.mp4 out.mp4 --video-frame 120
imq compare ref.mp4 out.mp4 --video-frames 0,30,60
imq compare ref.mp4 out.mp4 --video-frames 0:1,30:31
```

Tuple syntax is also accepted:

```bash
imq compare ref.mp4 out.mp4 --video-frames '(0,1),(30,31)'
imq compare ref.mp4 out.mp4 --video-frames '(0,1,0),(2,3,2)'
```

Two-element tuples are `(reference,distorted)`. Three-element tuples are
`(reference,distorted,label_frame_index)`; the third value is preserved in the
report and CSV `frame_index` column but does not affect extraction.

Order and duplicates are preserved:

```bash
imq compare ref.mp4 out.mp4 --video-frames 30:31,0:1,30:31
```

## Remote Video

Remote video frame extraction defaults to PNG pipe:

```bash
imq compare ssh://gpu-a/tmp/ref.mp4 ssh://gpu-a/tmp/out.mp4 \
  --video-frames 0:1,30:31
```

Internally, `imq` runs remote `ffmpeg` over SSH and decodes the returned PNG
locally. Remote video without frame selection is an error:

```text
remote video comparison requires --video-frame or --video-frames
```

## Safety Rules

- Default transfer is `stream`.
- No implicit `scp`.
- Stream failure returns an error.
- Remote video requires explicit frame selection.
- `copy-source` is explicit and warns because it copies full source files.
- Remote paths are POSIX shell-quoted before use in SSH commands.
- Captured remote stdout is bounded by `--remote-max-bytes`.

