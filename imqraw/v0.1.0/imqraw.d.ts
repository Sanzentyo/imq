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
export function captureWebGLRenderer(renderer: { getContext(): WebGLRenderingContext | WebGL2RenderingContext }, options?: CaptureOptions): Rgba8Image;
export function encodeThreeRenderer(renderer: { getContext(): WebGLRenderingContext | WebGL2RenderingContext }, options?: CaptureOptions): Uint8Array;
export function flipRgba8InPlace(data: Uint8Array, width: number, height: number): Uint8Array;
