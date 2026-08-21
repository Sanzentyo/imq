# imqraw Library Examples

Use these examples when the user wants to create, read, or pass `imqraw` bundles
from code instead of the `imq` CLI.

## JavaScript

Prefer fixed version URLs for reproducible workflows. Use `latest` only for
quick experiments.

```js
import {
  init,
  encodeRgba8,
  encodeBundle,
  encodeImage,
  decodeBundle,
  decodeImage,
  compareBundle,
  imqraw_image_count,
  PixelFormatCode,
} from "https://sanzentyo.github.io/imq/imqraw/v0.1.0/imqraw.js";

await init();

const reference = new Uint8Array([
  0, 0, 0, 255,
  255, 255, 255, 255,
]);
const distorted = new Uint8Array([
  0, 0, 0, 255,
  250, 250, 250, 255,
]);

const single = encodeRgba8(reference, 2, 1, {
  label: "reference",
  tags: ["ref"],
});

const bundle = encodeBundle([
  { data: reference, width: 2, height: 1, label: "reference", tags: ["ref"] },
  { data: distorted, width: 2, height: 1, label: "distorted", tags: ["dist"] },
]);

const mask = encodeImage(new Uint8Array([0b00000011]), 2, 1, PixelFormatCode.Binary1Lsb, {
  label: "mask",
  tags: ["binary"],
});

const decoded = decodeBundle(bundle);
const byTag = decodeImage(bundle, { tag: "dist" });
const metrics = compareBundle(bundle, { tag: "ref" }, { tag: "dist" }, {
  metrics: ["psnr", "mse", "mae"],
});
console.log(single.byteLength, mask.byteLength, decoded.length, byTag.label, metrics);
```

Three.js/WebGL capture:

```js
import {
  init,
  encodeThreeRenderer,
} from "https://sanzentyo.github.io/imq/imqraw/v0.1.0/imqraw.js";

await init();
renderer.render(scene, camera);

const bytes = encodeThreeRenderer(renderer, {
  label: "frame-0001",
  tags: ["threejs", "reference"],
});
```

For reliable WebGL reads, render to a readable target or use a context whose
drawing buffer is still available when `readPixels` runs.

Browser `Blob`/`ImageBitmap` and WebGPU capture are also available. WebGPU
textures must be RGBA8/BGRA8 and created with `COPY_SRC`; row-padding and BGRA
conversion are handled by the wrapper.

```js
import {
  captureWebGPUTexture,
  compareImages,
} from "https://sanzentyo.github.io/imq/imqraw/v0.1.0/imqraw.js";

const captured = await captureWebGPUTexture(device, texture, {
  width,
  height,
  format: "bgra8unorm",
  label: "candidate",
});
const metrics = compareImages(referenceImage, captured, {
  metrics: ["psnr", "mse"],
});
```

## TypeScript

The distribution includes `imqraw.d.ts`. In Deno or another TS-aware remote-ESM
setup, point TypeScript at the declaration file:

```ts
// @ts-types="https://sanzentyo.github.io/imq/imqraw/v0.1.0/imqraw.d.ts"
import {
  init,
  encodeBundle,
  encodeImage,
  imqraw_image_count,
  PixelFormatCode,
  type Rgba8Image,
} from "https://sanzentyo.github.io/imq/imqraw/v0.1.0/imqraw.js";

await init();

const reference: Rgba8Image = {
  data: new Uint8Array([0, 0, 0, 255]),
  width: 1,
  height: 1,
  label: "reference",
  tags: ["ref"],
};

const candidate: Rgba8Image = {
  data: new Uint8Array([8, 8, 8, 255]),
  width: 1,
  height: 1,
  label: "candidate",
  tags: ["dist"],
};

const bytes: Uint8Array = encodeBundle([reference, candidate]);
const hsvBytes: Uint8Array = encodeImage(
  new Uint8Array([0, 255, 255]),
  1,
  1,
  PixelFormatCode.Hsv8,
  { label: "hsv" },
);
console.log(imqraw_image_count(bytes));
```

Use `https://sanzentyo.github.io/imq/imqraw/latest/imqraw.js` and
`latest/imqraw.d.ts` only when reproducibility is not important.

## Rust

Use the Git repository directly until a crates.io release is published:

```toml
[dependencies]
imq = { git = "https://github.com/Sanzentyo/imq.git", tag = "v0.1.0", default-features = false }
```

Encode and decode a tagged multi-image bundle:

```rust
use imq::{
    Dimensions, FormatSpec, FrameOwned, OwnedPlane, PixelFormat, RawImageBundle, RawImageRecord, Result,
    decode_imqraw_bundle, encode_imqraw_bundle,
};

fn main() -> Result<()> {
    let reference = FrameOwned::packed_tight(
        vec![0, 0, 0, 255, 255, 255, 255, 255],
        2,
        1,
        PixelFormat::Rgba8,
    )?;
    let distorted = FrameOwned::packed_tight(
        vec![0, 0, 0, 255, 250, 250, 250, 255],
        2,
        1,
        PixelFormat::Rgba8,
    )?;
    let mask = FrameOwned::new(
        Dimensions::new(2, 1)?,
        FormatSpec::new(PixelFormat::Binary1Lsb),
        vec![OwnedPlane::new(vec![0b0000_0011], 1)],
    )?;

    let bundle = RawImageBundle::new(vec![
        RawImageRecord::new(
            Some("reference".to_string()),
            vec!["ref".to_string()],
            reference,
        ),
        RawImageRecord::new(
            Some("distorted".to_string()),
            vec!["dist".to_string()],
            distorted,
        ),
        RawImageRecord::new(Some("mask".to_string()), vec!["mask".to_string()], mask),
    ]);

    let bytes = encode_imqraw_bundle(&bundle)?;
    let decoded = decode_imqraw_bundle(&bytes)?;
    assert_eq!(decoded.select_tag("ref")?.label.as_deref(), Some("reference"));
    assert_eq!(decoded.select_tag("dist")?.label.as_deref(), Some("distorted"));
    Ok(())
}
```
