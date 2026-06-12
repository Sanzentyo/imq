import initWasm, {
  encode_imqraw_image,
  encode_imqraw_rgba8,
  encode_imqraw_rgba8_bundle,
  imqraw_image_count,
  imqraw_version,
} from "./imqraw_wasm.js";

export { initWasm, imqraw_image_count, imqraw_version };

export const PixelFormatCode = Object.freeze({
  Luma8: 1,
  Rgb8: 2,
  Rgba8: 3,
  Bgr8: 4,
  Bgra8: 5,
  Luma16Le: 6,
  Rgb16Le: 7,
  Rgba16Le: 8,
  RgbF32: 9,
  RgbaF32: 10,
  Yuv444p8: 11,
  Yuv422p8: 12,
  Yuv420p8: 13,
  Nv12: 14,
  Hsv8: 15,
  Hsva8: 16,
  Binary1Lsb: 17,
  Binary1Msb: 18,
});

export async function init(input) {
  return input === undefined ? initWasm() : initWasm({ module_or_path: input });
}

export function encodeRgba8(data, width, height, options = {}) {
  return encode_imqraw_rgba8(
    data,
    width,
    height,
    options.label ?? "",
    options.tags ?? [],
  );
}

export function encodeBundle(images) {
  return encode_imqraw_rgba8_bundle(
    images.map((image) => ({
      data: image.data,
      width: image.width,
      height: image.height,
      label: image.label ?? "",
      tags: image.tags ?? [],
    })),
  );
}

export function encodeImage(data, width, height, pixelFormat, options = {}) {
  return encode_imqraw_image(
    data,
    width,
    height,
    pixelFormat,
    options.stride ?? 0,
    options.label ?? "",
    options.tags ?? [],
  );
}

export function captureWebGLRenderer(renderer, options = {}) {
  const gl = renderer.getContext();
  const width = options.width ?? gl.drawingBufferWidth;
  const height = options.height ?? gl.drawingBufferHeight;
  const pixels = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  flipRgba8InPlace(pixels, width, height);
  return {
    data: pixels,
    width,
    height,
    label: options.label ?? "",
    tags: options.tags ?? [],
  };
}

export function encodeThreeRenderer(renderer, options = {}) {
  return encodeBundle([captureWebGLRenderer(renderer, options)]);
}

export function flipRgba8InPlace(data, width, height) {
  const rowBytes = width * 4;
  const temp = new Uint8Array(rowBytes);
  for (let y = 0, last = height - 1; y < last; y += 1, last -= 1) {
    const top = y * rowBytes;
    const bottom = last * rowBytes;
    temp.set(data.subarray(top, top + rowBytes));
    data.copyWithin(top, bottom, bottom + rowBytes);
    data.set(temp, bottom);
  }
  return data;
}
