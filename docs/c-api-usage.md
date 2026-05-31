# C API Usage

`imqraw-capi` exposes a small C ABI for encoding and decoding `imqraw` bundles.
It is intended for C, C++, Swift, Zig, Python extension modules, and other
foreign-function interfaces that need exact raw image interchange without going
through the CLI.

## Build

```bash
cargo build -p imqraw-capi
```

The package builds both dynamic and static libraries:

- macOS: `target/debug/libimqraw_capi.dylib`
- Linux: `target/debug/libimqraw_capi.so`
- static: `target/debug/libimqraw_capi.a`

The public header is:

```text
crates/imqraw-capi/include/imqraw.h
```

## Ownership

- `ImqrawByteView` and `ImqrawStringView` are borrowed inputs.
- `ImqrawByteBuffer` returned by encode functions is owned by Rust and must be
  freed with `imqraw_buffer_free`.
- `ImqrawBundle *` returned by `imqraw_bundle_decode` is an opaque decoded
  handle and must be freed with `imqraw_bundle_free`.
- Label, tag, and plane views returned from a decoded bundle are borrowed and
  valid only until that bundle handle is freed.
- `imqraw_encode_image` encodes one image. `imqraw_encode_bundle` encodes
  multiple tagged images into one bundle.
- Set `ImqrawEncodeImage.data` for tight packed RGB/RGBA/Luma input. Set
  `planes` and `plane_count` instead when passing custom strides, YUV, or NV12.
- Use the `IMQRAW_PIXEL_FORMAT_*` constants for `pixel_format`. The optional
  `color_space`, `transfer`, and `range` fields accept 0 for format defaults.

## Minimal Example

```c
#include "imqraw.h"

#include <stdint.h>
#include <stdio.h>

int main(void) {
  const uint8_t pixels[] = {
      0, 0, 0, 255,
      255, 255, 255, 255,
  };
  const ImqrawStringView tag = {"reference", 9};
  const ImqrawEncodeImage image = {
      .width = 2,
      .height = 1,
      .pixel_format = IMQRAW_PIXEL_FORMAT_RGBA8,
      .data = {pixels, sizeof pixels},
      .label = {"ref", 3},
      .tags = &tag,
      .tag_count = 1,
  };
  ImqrawStatus status = {0};
  ImqrawByteBuffer bytes = {0};

  if (imqraw_encode_image(&image, &bytes, &status) != IMQRAW_STATUS_OK) {
    fprintf(stderr, "encode failed: %s\n", status.message);
    return 1;
  }

  ImqrawBundle *bundle = 0;
  if (imqraw_bundle_decode((ImqrawByteView){bytes.ptr, bytes.len}, &bundle, &status) !=
      IMQRAW_STATUS_OK) {
    fprintf(stderr, "decode failed: %s\n", status.message);
    imqraw_buffer_free(bytes);
    return 1;
  }

  ImqrawImageInfo info = {0};
  imqraw_bundle_image_info(bundle, 0, &info, &status);
  printf("%ux%u format=%u\n", info.width, info.height, info.pixel_format);

  imqraw_bundle_free(bundle);
  imqraw_buffer_free(bytes);
  return 0;
}
```

## Sample Project

Run the bundled C sample:

```bash
samples/imqraw-c-api/build-and-run.sh
```
