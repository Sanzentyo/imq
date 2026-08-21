export function init(input?: RequestInfo | URL | Response | BufferSource | WebAssembly.Module): Promise<unknown>;
export function initWasm(input?: RequestInfo | URL | Response | BufferSource | WebAssembly.Module): Promise<unknown>;
export function imqraw_version(): string;
export function imqraw_image_count(bytes: Uint8Array): number;
export function imqraw_find_tag(bytes: Uint8Array, tag: string): number;

export interface Rgba8Image {
  data: Uint8Array;
  width: number;
  height: number;
  label?: string;
  tags?: string[];
}

export const PixelFormatCode: {
  readonly Luma8: 1;
  readonly Rgb8: 2;
  readonly Rgba8: 3;
  readonly Bgr8: 4;
  readonly Bgra8: 5;
  readonly Luma16Le: 6;
  readonly Rgb16Le: 7;
  readonly Rgba16Le: 8;
  readonly RgbF32: 9;
  readonly RgbaF32: 10;
  readonly Yuv444p8: 11;
  readonly Yuv422p8: 12;
  readonly Yuv420p8: 13;
  readonly Nv12: 14;
  readonly Hsv8: 15;
  readonly Hsva8: 16;
  readonly Binary1Lsb: 17;
  readonly Binary1Msb: 18;
};
export type PixelFormatCodeValue = (typeof PixelFormatCode)[keyof typeof PixelFormatCode];

export interface ImageOptions {
  stride?: number;
  label?: string;
  tags?: string[];
}

export interface DecodedPlane {
  data: Uint8Array;
  stride: number;
}

export interface DecodedImage {
  index: number;
  width: number;
  height: number;
  pixelFormat: number;
  colorSpace: number;
  transfer: number;
  range: number;
  label: string | null;
  tags: string[];
  planes: DecodedPlane[];
  data?: Uint8Array;
  stride?: number;
}

export type ImageSelector = number | string | { tag: string };

export interface MetricResult {
  name: string;
  score: number;
  unit: string;
  direction: "higher-is-better" | "lower-is-better" | "neutral";
  details: Record<string, number>;
}

export interface CompareOptions {
  metrics?: string | string[];
}

export interface CaptureOptions {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  flipY?: boolean;
  label?: string;
  tags?: string[];
}

export type BrowserImageSource = ImageBitmapSource | Blob;

export interface ImageSourceCompareOptions extends CompareOptions {
  reference?: CaptureOptions;
  distorted?: CaptureOptions;
}

export interface WebGPUCaptureOptions extends Omit<CaptureOptions, "x" | "y"> {
  width: number;
  height: number;
  origin?: { x?: number; y?: number; z?: number } | [number, number?, number?];
  format?: "rgba8unorm" | "rgba8unorm-srgb" | "bgra8unorm" | "bgra8unorm-srgb";
}

export interface WebGPUBufferLike {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

export interface WebGPUCommandEncoderLike {
  copyTextureToBuffer(source: unknown, destination: unknown, size: unknown): void;
  finish(): unknown;
}

export interface WebGPUDeviceLike {
  createBuffer(descriptor: { size: number; usage: number }): WebGPUBufferLike;
  createCommandEncoder(): WebGPUCommandEncoderLike;
  queue: { submit(commands: unknown[]): void };
}

export function encodeRgba8(
  data: Uint8Array,
  width: number,
  height: number,
  options?: Pick<Rgba8Image, "label" | "tags">,
): Uint8Array;
export function encodeBundle(images: Rgba8Image[]): Uint8Array;
export function encodeImage(
  data: Uint8Array,
  width: number,
  height: number,
  pixelFormat: PixelFormatCodeValue | number,
  options?: ImageOptions,
): Uint8Array;
export function decodeBundle(bytes: Uint8Array): DecodedImage[];
export function decodeImage(bytes: Uint8Array, selector?: ImageSelector): DecodedImage;
export function compareBundle(
  bytes: Uint8Array,
  reference?: ImageSelector,
  distorted?: ImageSelector,
  options?: CompareOptions,
): MetricResult[];
export function compareImages(
  reference: Rgba8Image,
  distorted: Rgba8Image,
  options?: CompareOptions,
): MetricResult[];
export function captureImageBitmap(source: BrowserImageSource, options?: CaptureOptions): Promise<Rgba8Image>;
export function encodeImageBitmap(source: BrowserImageSource, options?: CaptureOptions): Promise<Uint8Array>;
export function compareImageSources(
  reference: BrowserImageSource,
  distorted: BrowserImageSource,
  options?: ImageSourceCompareOptions,
): Promise<MetricResult[]>;
export function captureWebGLRenderer(renderer: { getContext(): WebGLRenderingContext | WebGL2RenderingContext }, options?: CaptureOptions): Rgba8Image;
export function captureWebGLContext(context: WebGLRenderingContext | WebGL2RenderingContext, options?: CaptureOptions): Rgba8Image;
export function encodeThreeRenderer(renderer: { getContext(): WebGLRenderingContext | WebGL2RenderingContext }, options?: CaptureOptions): Uint8Array;
export function captureWebGPUTexture(
  device: WebGPUDeviceLike,
  texture: object,
  options: WebGPUCaptureOptions,
): Promise<Rgba8Image>;
export function encodeWebGPUTexture(
  device: WebGPUDeviceLike,
  texture: object,
  options: WebGPUCaptureOptions,
): Promise<Uint8Array>;
export function flipRgba8InPlace(data: Uint8Array, width: number, height: number): Uint8Array;
export function swapRedBlueInPlace(data: Uint8Array): Uint8Array;
