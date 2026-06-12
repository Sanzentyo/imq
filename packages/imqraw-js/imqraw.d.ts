export function init(input?: RequestInfo | URL | Response | BufferSource | WebAssembly.Module): Promise<unknown>;
export function initWasm(input?: RequestInfo | URL | Response | BufferSource | WebAssembly.Module): Promise<unknown>;
export function imqraw_version(): string;
export function imqraw_image_count(bytes: Uint8Array): number;

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

export interface CaptureOptions {
  width?: number;
  height?: number;
  label?: string;
  tags?: string[];
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
export function captureWebGLRenderer(renderer: { getContext(): WebGLRenderingContext | WebGL2RenderingContext }, options?: CaptureOptions): Rgba8Image;
export function encodeThreeRenderer(renderer: { getContext(): WebGLRenderingContext | WebGL2RenderingContext }, options?: CaptureOptions): Uint8Array;
export function flipRgba8InPlace(data: Uint8Array, width: number, height: number): Uint8Array;
