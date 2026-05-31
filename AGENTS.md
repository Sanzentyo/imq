# Repository Instructions

## imqraw Web Distribution

`imqraw` WebAssembly distribution is provided from GitHub only:

- Build browser artifacts from `crates/imqraw-wasm` with `wasm-pack --target web`.
- Publish immutable versioned directories under GitHub Pages:
  - `imqraw/vX.Y.Z/` is immutable after publication.
  - `imqraw/latest/` is replaced by the newest released version.
- Attach the same generated files as a zip to the matching GitHub Release.
- Prefer GitHub Pages + Releases for browser usage. Do not make GitHub Packages npm the primary distribution path unless explicitly requested, because it adds npm registry/token setup for consumers.
- Use fixed `vX.Y.Z` URLs in production examples. Use `latest` only for quick experiments.
- Keep generated `.wasm`/`.js` files out of `main`; publish them to `gh-pages` and Releases.

Expected import shape:

```js
import { init, encodeRgba8 } from "https://sanzentyo.github.io/imq/imqraw/v0.1.0/imqraw.js";

await init();
const bytes = encodeRgba8(rgbaBytes, width, height, {
  label: "frame-0001",
  tags: ["threejs", "reference"],
});
```

For Three.js/WebGL capture, use the JS wrapper in `packages/imqraw-js/imqraw.js`.
It reads RGBA8 pixels from the renderer context, flips WebGL's bottom-left origin
to top-left order, and passes the result to the wasm encoder.
