import { performance } from "node:perf_hooks";
import { readFile } from "node:fs/promises";
import {
  init,
  encodeRgba8,
  encodeBundle,
  imqraw_image_count,
} from "../target/imqraw-pages/imqraw/latest/imqraw.js";

const wasmBytes = await readFile(
  new URL("../target/imqraw-pages/imqraw/latest/imqraw_wasm_bg.wasm", import.meta.url),
);
await init(wasmBytes);

const width = Number.parseInt(process.argv[2] ?? "1920", 10);
const height = Number.parseInt(process.argv[3] ?? "1080", 10);
const iterations = Number.parseInt(process.argv[4] ?? "30", 10);
const data = new Uint8Array(width * height * 4);

for (let i = 0; i < data.length; i += 4) {
  data[i] = i & 0xff;
  data[i + 1] = (i >> 8) & 0xff;
  data[i + 2] = 255 - (i & 0xff);
  data[i + 3] = 255;
}

const warmup = encodeRgba8(data, width, height, {
  label: "warmup",
  tags: ["bench"],
});
if (imqraw_image_count(warmup) !== 1) {
  throw new Error("encoded warmup bundle did not contain exactly one image");
}

const start = performance.now();
let totalBytes = 0;
for (let i = 0; i < iterations; i += 1) {
  const bytes = encodeBundle([
    { data, width, height, label: `frame-${i}`, tags: ["bench", "rgba8"] },
  ]);
  totalBytes += bytes.byteLength;
}
const elapsedMs = performance.now() - start;
const megapixels = (width * height * iterations) / 1_000_000;

console.log(JSON.stringify({
  width,
  height,
  iterations,
  totalBytes,
  elapsedMs,
  millisecondsPerImage: elapsedMs / iterations,
  megapixelsPerSecond: megapixels / (elapsedMs / 1000),
}, null, 2));
