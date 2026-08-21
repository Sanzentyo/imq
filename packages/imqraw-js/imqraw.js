import initWasm, {
  compare_imqraw_images,
  decode_imqraw_image,
  decode_imqraw_images,
  encode_imqraw_image,
  encode_imqraw_rgba8,
  encode_imqraw_rgba8_bundle,
  imqraw_find_tag,
  imqraw_image_count,
  imqraw_version,
} from "./imqraw_wasm.js";

export {
  initWasm,
  imqraw_find_tag,
  imqraw_image_count,
  imqraw_version,
};

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

export function decodeBundle(bytes) {
  return Array.from(decode_imqraw_images(bytes));
}

export function decodeImage(bytes, selector = 0) {
  const index = resolveSelector(bytes, selector);
  return decode_imqraw_image(bytes, index);
}

export function compareBundle(bytes, reference = 0, distorted = 1, options = {}) {
  const referenceIndex = resolveSelector(bytes, reference);
  const distortedIndex = resolveSelector(bytes, distorted);
  const metrics = Array.isArray(options.metrics)
    ? options.metrics.join(",")
    : (options.metrics ?? "");
  return Array.from(
    compare_imqraw_images(bytes, referenceIndex, distortedIndex, metrics),
  );
}

export function compareImages(reference, distorted, options = {}) {
  const bundle = encodeBundle([
    normalizeComparisonImage(reference, "reference"),
    normalizeComparisonImage(distorted, "distorted"),
  ]);
  return compareBundle(bundle, 0, 1, options);
}

function resolveSelector(bytes, selector) {
  if (typeof selector === "number") {
    if (!Number.isSafeInteger(selector) || selector < 0) {
      throw new RangeError("imqraw image index must be a non-negative safe integer");
    }
    return selector;
  }
  const tag = typeof selector === "string" ? selector : selector?.tag;
  if (typeof tag !== "string" || tag.length === 0) {
    throw new TypeError("imqraw selector must be an index, tag string, or { tag }");
  }
  return imqraw_find_tag(bytes, tag);
}

function normalizeComparisonImage(image, defaultTag) {
  if (image?.data instanceof Uint8Array) {
    return {
      ...image,
      tags: image.tags ?? [defaultTag],
    };
  }
  throw new TypeError("comparison image must include Uint8Array data, width, and height");
}

export async function captureImageBitmap(source, options = {}) {
  const sourceIsBitmap = typeof ImageBitmap === "function" && source instanceof ImageBitmap;
  const bitmap = sourceIsBitmap ? source : await createImageBitmap(source);
  try {
    const width = options.width ?? bitmap.width;
    const height = options.height ?? bitmap.height;
    if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0) {
      throw new RangeError("capture dimensions must be positive safe integers");
    }
    const canvas = typeof OffscreenCanvas === "function"
      ? new OffscreenCanvas(width, height)
      : Object.assign(document.createElement("canvas"), { width, height });
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) {
      throw new Error("2D canvas context is unavailable");
    }
    context.drawImage(bitmap, 0, 0, width, height);
    const pixels = context.getImageData(0, 0, width, height).data;
    return {
      data: new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength),
      width,
      height,
      label: options.label ?? "",
      tags: options.tags ?? [],
    };
  } finally {
    if (!sourceIsBitmap && typeof bitmap.close === "function") {
      bitmap.close();
    }
  }
}

export async function encodeImageBitmap(source, options = {}) {
  return encodeBundle([await captureImageBitmap(source, options)]);
}

export async function compareImageSources(reference, distorted, options = {}) {
  const [referenceImage, distortedImage] = await Promise.all([
    captureImageBitmap(reference, {
      ...options.reference,
      tags: options.reference?.tags ?? ["reference"],
    }),
    captureImageBitmap(distorted, {
      ...options.distorted,
      tags: options.distorted?.tags ?? ["distorted"],
    }),
  ]);
  return compareImages(referenceImage, distortedImage, options);
}

export function captureWebGLRenderer(renderer, options = {}) {
  return captureWebGLContext(renderer.getContext(), options);
}

export function captureWebGLContext(gl, options = {}) {
  const width = options.width ?? gl.drawingBufferWidth;
  const height = options.height ?? gl.drawingBufferHeight;
  assertRgbaDimensions(width, height);
  const byteLength = checkedRgbaByteLength(width, height);
  const pixels = new Uint8Array(byteLength);
  gl.readPixels(
    options.x ?? 0,
    options.y ?? 0,
    width,
    height,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    pixels,
  );
  if (options.flipY !== false) {
    flipRgba8InPlace(pixels, width, height);
  }
  return {
    data: pixels,
    width,
    height,
    label: options.label ?? "",
    tags: options.tags ?? [],
  };
}

export async function captureWebGPUTexture(device, texture, options) {
  if (!options || !Number.isSafeInteger(options.width) || options.width <= 0
      || !Number.isSafeInteger(options.height) || options.height <= 0) {
    throw new RangeError("WebGPU capture requires positive integer width and height");
  }
  const { width, height } = options;
  const unpaddedBytesPerRow = checkedRgbaByteLength(width, 1);
  const bytesPerRow = Math.ceil(unpaddedBytesPerRow / 256) * 256;
  const bufferSize = bytesPerRow * height;
  if (!Number.isSafeInteger(bufferSize)) {
    throw new RangeError("WebGPU capture buffer size exceeds the safe integer range");
  }
  const buffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture, origin: options.origin ?? { x: 0, y: 0, z: 0 } },
      { buffer, bytesPerRow, rowsPerImage: height },
      { width, height, depthOrArrayLayers: 1 },
    );
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(buffer.getMappedRange());
    const data = new Uint8Array(unpaddedBytesPerRow * height);
    for (let y = 0; y < height; y += 1) {
      data.set(
        mapped.subarray(y * bytesPerRow, y * bytesPerRow + unpaddedBytesPerRow),
        y * unpaddedBytesPerRow,
      );
    }
    buffer.unmap();
    if (options.format === "bgra8unorm" || options.format === "bgra8unorm-srgb") {
      swapRedBlueInPlace(data);
    } else if (options.format && options.format !== "rgba8unorm" && options.format !== "rgba8unorm-srgb") {
      throw new TypeError(`unsupported WebGPU texture format ${options.format}`);
    }
    if (options.flipY) {
      flipRgba8InPlace(data, width, height);
    }
    return {
      data,
      width,
      height,
      label: options.label ?? "",
      tags: options.tags ?? [],
    };
  } finally {
    buffer.destroy();
  }
}

export async function encodeWebGPUTexture(device, texture, options) {
  return encodeBundle([await captureWebGPUTexture(device, texture, options)]);
}

export function encodeThreeRenderer(renderer, options = {}) {
  return encodeBundle([captureWebGLRenderer(renderer, options)]);
}

export function flipRgba8InPlace(data, width, height) {
  assertRgbaDimensions(width, height);
  const expectedLength = checkedRgbaByteLength(width, height);
  if (data.byteLength !== expectedLength) {
    throw new RangeError(`expected ${expectedLength} RGBA bytes, received ${data.byteLength}`);
  }
  const rowBytes = checkedRgbaByteLength(width, 1);
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


export function swapRedBlueInPlace(data) {
  if (data.byteLength % 4 !== 0) {
    throw new RangeError("RGBA/BGRA byte length must be divisible by four");
  }
  for (let offset = 0; offset < data.byteLength; offset += 4) {
    const red = data[offset];
    data[offset] = data[offset + 2];
    data[offset + 2] = red;
  }
  return data;
}

function assertRgbaDimensions(width, height) {
  if (!Number.isSafeInteger(width) || width <= 0
      || !Number.isSafeInteger(height) || height <= 0) {
    throw new RangeError("RGBA dimensions must be positive safe integers");
  }
}

function checkedRgbaByteLength(width, height) {
  const byteLength = width * height * 4;
  if (!Number.isSafeInteger(byteLength)) {
    throw new RangeError("RGBA byte length exceeds the safe integer range");
  }
  return byteLength;
}
