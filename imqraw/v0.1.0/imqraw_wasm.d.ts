/* tslint:disable */
/* eslint-disable */

/**
 * Encodes one tightly packed RGBA8 image as an `imqraw` bundle.
 */
export function encode_imqraw_rgba8(data: Uint8Array, width: number, height: number, label: string, tags: Array<any>): Uint8Array;

/**
 * Encodes an array of `{ data, width, height, label?, tags? }` RGBA8 records.
 */
export function encode_imqraw_rgba8_bundle(images: Array<any>): Uint8Array;

/**
 * Returns the number of images in an encoded `imqraw` bundle.
 */
export function imqraw_image_count(bytes: Uint8Array): number;

/**
 * Returns the wasm package version.
 */
export function imqraw_version(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly encode_imqraw_rgba8: (a: number, b: number, c: number, d: number, e: number, f: number, g: any) => [number, number, number, number];
    readonly encode_imqraw_rgba8_bundle: (a: any) => [number, number, number, number];
    readonly imqraw_image_count: (a: number, b: number) => [number, number, number];
    readonly imqraw_version: () => [number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
