# JavaScript and Browser Usage

`imqraw` is published as static ESM + WebAssembly files on GitHub Pages. This is
the recommended JavaScript distribution path because it works without npm
registry or token setup.

Use fixed version URLs for reproducible workflows:

```js
import { init, encodeRgba8, imqraw_image_count } from "https://sanzentyo.github.io/imq/imqraw/v0.1.0/imqraw.js";

await init();
const bytes = encodeRgba8(new Uint8Array([0, 0, 0, 255]), 1, 1, {
  label: "frame-0001",
  tags: ["reference"],
});
console.log(imqraw_image_count(bytes));
```

Use `latest` only for quick experiments:

```js
import { init, encodeRgba8 } from "https://sanzentyo.github.io/imq/imqraw/latest/imqraw.js";
```

## Three.js / WebGL Capture

The JS wrapper includes helpers for WebGL renderers. `captureWebGLRenderer`
reads the current drawing buffer as RGBA8, flips WebGL's bottom-left origin to
top-left order, and returns a record. `encodeThreeRenderer` encodes that record
as an `imqraw` bundle.

```js
import { init, encodeThreeRenderer } from "https://sanzentyo.github.io/imq/imqraw/v0.1.0/imqraw.js";

await init();
renderer.render(scene, camera);
const bytes = encodeThreeRenderer(renderer, {
  label: "threejs-frame",
  tags: ["threejs", "reference"],
});
```

For WebGL capture to work reliably, render into a context that can be read
after rendering. In Three.js this usually means setting `preserveDrawingBuffer`
or reading from a render target before the buffer is invalidated.

## Browser Sample

Open the sample page through a local static server:

```bash
python3 -m http.server 4173 --directory samples/imqraw-js-browser
```

Then open:

```text
http://localhost:4173/
```

The page imports the published `v0.1.0` ESM build, initializes wasm, encodes a
small RGBA8 image, and reports the encoded bundle size and image count.
