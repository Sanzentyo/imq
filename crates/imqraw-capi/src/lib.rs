//! C ABI for encoding and decoding `imqraw` bundles.

use imq::{
    ColorRange, ColorSpace, Dimensions, FormatSpec, FrameOwned, OwnedPlane, PixelFormat,
    RawImageBundle, RawImageRecord, Transfer, decode_imqraw_bundle, encode_imqraw_bundle,
};
use std::ffi::c_char;
use std::mem::ManuallyDrop;
use std::ptr;
use std::slice;
use std::str;

const STATUS_MESSAGE_CAP: usize = 256;

/// Status code returned by the C API.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImqrawStatusCode {
    /// Operation succeeded.
    Ok = 0,
    /// A required pointer was null.
    NullPointer = 1,
    /// An argument was invalid.
    InvalidArgument = 2,
    /// Bundle decoding failed.
    DecodeError = 3,
    /// Bundle encoding failed.
    EncodeError = 4,
}

/// Status object optionally filled by C API functions.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ImqrawStatus {
    /// Numeric status code.
    pub code: i32,
    /// NUL-terminated UTF-8 message.
    pub message: [c_char; STATUS_MESSAGE_CAP],
}

/// Borrowed byte span.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ImqrawByteView {
    /// Pointer to the first byte.
    pub ptr: *const u8,
    /// Number of bytes.
    pub len: usize,
}

/// Owned byte buffer allocated by Rust.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ImqrawByteBuffer {
    /// Pointer to the first byte.
    pub ptr: *mut u8,
    /// Number of initialized bytes.
    pub len: usize,
    /// Allocation capacity. Pass the complete struct to `imqraw_buffer_free`.
    pub cap: usize,
}

/// Borrowed UTF-8 string span.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ImqrawStringView {
    /// Pointer to UTF-8 bytes. A null pointer with length zero is empty.
    pub ptr: *const c_char,
    /// Number of UTF-8 bytes.
    pub len: usize,
}

/// Borrowed decoded plane view.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ImqrawPlaneView {
    /// Pointer to plane bytes. Valid while the owning bundle handle is alive.
    pub ptr: *const u8,
    /// Number of bytes in the plane.
    pub len: usize,
    /// Row stride in bytes.
    pub stride: usize,
}

/// Basic decoded image metadata.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ImqrawImageInfo {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Stable `imqraw` pixel-format code.
    pub pixel_format: u16,
    /// Stable color-space code.
    pub color_space: u16,
    /// Stable transfer code.
    pub transfer: u16,
    /// Stable range code.
    pub range: u16,
    /// Number of planes.
    pub plane_count: usize,
}

/// Input plane used by generic encode functions.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ImqrawEncodePlane {
    /// Plane bytes.
    pub data: ImqrawByteView,
    /// Row stride in bytes.
    pub stride: usize,
}

/// Input image used by encode functions.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ImqrawEncodeImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Stable `imqraw` pixel-format code. Use `IMQRAW_PIXEL_FORMAT_*`.
    pub pixel_format: u16,
    /// Stable color-space code. Use 0 for the format default.
    pub color_space: u16,
    /// Stable transfer code. Use 0 for the format default.
    pub transfer: u16,
    /// Stable range code. Use 0 for the format default.
    pub range: u16,
    /// Tight pixel bytes for packed single-plane input. Used when
    /// `plane_count == 0`.
    pub data: ImqrawByteView,
    /// Optional explicit plane array. Use for custom stride, YUV, and NV12.
    pub planes: *const ImqrawEncodePlane,
    /// Number of explicit planes.
    pub plane_count: usize,
    /// Optional UTF-8 label.
    pub label: ImqrawStringView,
    /// Optional tag string-view array.
    pub tags: *const ImqrawStringView,
    /// Number of tag views.
    pub tag_count: usize,
}

/// Opaque decoded bundle handle.
pub struct ImqrawBundle {
    bundle: RawImageBundle,
}

/// Returns the C API package version.
#[unsafe(no_mangle)]
pub extern "C" fn imqraw_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Encodes one tightly packed RGBA8 image into an `imqraw` bundle.
///
/// # Safety
///
/// `image` and `out` must be non-null and writable where appropriate.
/// `image.data.ptr` must point to `image.data.len` readable bytes when
/// `image.data.len > 0`. `image.tags` must point to `image.tag_count` readable
/// string views when `image.tag_count > 0`.
/// `out` must be non-null and writable. The returned buffer must be released
/// with `imqraw_buffer_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_encode_rgba8(
    image: *const ImqrawEncodeImage,
    out: *mut ImqrawByteBuffer,
    status: *mut ImqrawStatus,
) -> i32 {
    encode_single_image(image, out, PixelFormat::Rgba8, status)
}

/// Encodes one tightly packed Luma8 image into an `imqraw` bundle.
///
/// # Safety
///
/// `image` and `out` must be non-null and writable where appropriate.
/// `image.data.ptr` must point to `image.data.len` readable bytes when
/// `image.data.len > 0`. `image.tags` must point to `image.tag_count` readable
/// string views when `image.tag_count > 0`.
/// `out` must be non-null and writable. The returned buffer must be released
/// with `imqraw_buffer_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_encode_luma8(
    image: *const ImqrawEncodeImage,
    out: *mut ImqrawByteBuffer,
    status: *mut ImqrawStatus,
) -> i32 {
    encode_single_image(image, out, PixelFormat::Luma8, status)
}

/// Encodes one image into an `imqraw` bundle using `image.pixel_format`.
///
/// # Safety
///
/// `image` and `out` must be non-null and writable where appropriate. Borrowed
/// data, plane, and tag pointers must address readable memory for their stated
/// lengths. The returned buffer must be released with `imqraw_buffer_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_encode_image(
    image: *const ImqrawEncodeImage,
    out: *mut ImqrawByteBuffer,
    status: *mut ImqrawStatus,
) -> i32 {
    encode_images(image, 1, out, None, status)
}

/// Encodes multiple images into one `imqraw` bundle.
///
/// # Safety
///
/// `images` must point to `image_count` readable image descriptors when
/// `image_count > 0`. `out` must be non-null and writable. Borrowed data,
/// plane, and tag pointers inside each image must address readable memory for
/// their stated lengths. The returned buffer must be released with
/// `imqraw_buffer_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_encode_bundle(
    images: *const ImqrawEncodeImage,
    image_count: usize,
    out: *mut ImqrawByteBuffer,
    status: *mut ImqrawStatus,
) -> i32 {
    encode_images(images, image_count, out, None, status)
}

/// Frees a byte buffer returned by this API.
///
/// # Safety
///
/// `buffer` must either be the zero/null buffer or an exact buffer previously
/// returned by this library and not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_buffer_free(buffer: ImqrawByteBuffer) {
    if buffer.ptr.is_null() {
        return;
    }
    // SAFETY: The contract requires buffers to come from this library with the
    // original pointer, length, and capacity.
    unsafe {
        drop(Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.cap));
    }
}

/// Decodes an `imqraw` bundle into an opaque handle.
///
/// # Safety
///
/// `bytes.ptr` must point to `bytes.len` readable bytes when `bytes.len > 0`.
/// `out` must be non-null and writable. The returned handle must be released
/// with `imqraw_bundle_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_bundle_decode(
    bytes: ImqrawByteView,
    out: *mut *mut ImqrawBundle,
    status: *mut ImqrawStatus,
) -> i32 {
    with_status(status, || {
        if out.is_null() {
            return Err(CApiError::new(
                ImqrawStatusCode::NullPointer,
                "out bundle pointer is null",
            ));
        }
        let bytes = byte_slice(bytes)?;
        let bundle = decode_imqraw_bundle(bytes)
            .map_err(|error| CApiError::new(ImqrawStatusCode::DecodeError, error.to_string()))?;
        let handle = Box::new(ImqrawBundle { bundle });
        // SAFETY: `out` was checked non-null and points to writable caller
        // storage by function contract.
        unsafe {
            *out = Box::into_raw(handle);
        }
        Ok(())
    })
}

/// Frees a decoded bundle handle.
///
/// # Safety
///
/// `bundle` must be null or a handle previously returned by
/// `imqraw_bundle_decode` and not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_bundle_free(bundle: *mut ImqrawBundle) {
    if bundle.is_null() {
        return;
    }
    // SAFETY: The contract requires a valid handle allocated by this library.
    unsafe {
        drop(Box::from_raw(bundle));
    }
}

/// Returns the number of images in a decoded bundle.
///
/// # Safety
///
/// `bundle` must be a valid decoded bundle handle. `out` must be non-null and
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_bundle_image_count(
    bundle: *const ImqrawBundle,
    out: *mut usize,
    status: *mut ImqrawStatus,
) -> i32 {
    with_status(status, || {
        let bundle = bundle_ref(bundle)?;
        if out.is_null() {
            return Err(CApiError::new(
                ImqrawStatusCode::NullPointer,
                "out count pointer is null",
            ));
        }
        // SAFETY: `out` was checked non-null and points to writable caller
        // storage by function contract.
        unsafe {
            *out = bundle.bundle.records.len();
        }
        Ok(())
    })
}

/// Returns metadata for one image in a decoded bundle.
///
/// # Safety
///
/// `bundle` must be a valid decoded bundle handle. `out` must be non-null and
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_bundle_image_info(
    bundle: *const ImqrawBundle,
    image_index: usize,
    out: *mut ImqrawImageInfo,
    status: *mut ImqrawStatus,
) -> i32 {
    with_status(status, || {
        let record = image_record(bundle, image_index)?;
        if out.is_null() {
            return Err(CApiError::new(
                ImqrawStatusCode::NullPointer,
                "out info pointer is null",
            ));
        }
        let dims = record.frame.dimensions();
        let format = record.frame.format();
        let info = ImqrawImageInfo {
            width: dims.width,
            height: dims.height,
            pixel_format: pixel_format_code(format.pixel_format),
            color_space: color_space_code(format.color_space),
            transfer: transfer_code(format.transfer),
            range: color_range_code(format.range),
            plane_count: record.frame.owned_planes().len(),
        };
        // SAFETY: `out` was checked non-null and points to writable caller
        // storage by function contract.
        unsafe {
            *out = info;
        }
        Ok(())
    })
}

/// Returns a borrowed UTF-8 label view for one image.
///
/// # Safety
///
/// `bundle` must be valid and `out` must be non-null and writable. The returned
/// string view is valid until the bundle handle is freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_bundle_image_label(
    bundle: *const ImqrawBundle,
    image_index: usize,
    out: *mut ImqrawStringView,
    status: *mut ImqrawStatus,
) -> i32 {
    with_status(status, || {
        let record = image_record(bundle, image_index)?;
        if out.is_null() {
            return Err(CApiError::new(
                ImqrawStatusCode::NullPointer,
                "out label pointer is null",
            ));
        }
        let view = record
            .label
            .as_deref()
            .map_or(empty_string_view(), string_view_from_str);
        // SAFETY: `out` was checked non-null and points to writable caller
        // storage by function contract.
        unsafe {
            *out = view;
        }
        Ok(())
    })
}

/// Returns the number of tags on one image.
///
/// # Safety
///
/// `bundle` must be valid and `out` must be non-null and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_bundle_image_tag_count(
    bundle: *const ImqrawBundle,
    image_index: usize,
    out: *mut usize,
    status: *mut ImqrawStatus,
) -> i32 {
    with_status(status, || {
        let record = image_record(bundle, image_index)?;
        if out.is_null() {
            return Err(CApiError::new(
                ImqrawStatusCode::NullPointer,
                "out tag count pointer is null",
            ));
        }
        // SAFETY: `out` was checked non-null and points to writable caller
        // storage by function contract.
        unsafe {
            *out = record.tags.len();
        }
        Ok(())
    })
}

/// Returns one borrowed UTF-8 tag view.
///
/// # Safety
///
/// `bundle` must be valid and `out` must be non-null and writable. The returned
/// string view is valid until the bundle handle is freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_bundle_image_tag(
    bundle: *const ImqrawBundle,
    image_index: usize,
    tag_index: usize,
    out: *mut ImqrawStringView,
    status: *mut ImqrawStatus,
) -> i32 {
    with_status(status, || {
        let record = image_record(bundle, image_index)?;
        if out.is_null() {
            return Err(CApiError::new(
                ImqrawStatusCode::NullPointer,
                "out tag pointer is null",
            ));
        }
        let tag = record.tags.get(tag_index).ok_or_else(|| {
            CApiError::new(
                ImqrawStatusCode::InvalidArgument,
                format!("tag index {tag_index} is out of range"),
            )
        })?;
        // SAFETY: `out` was checked non-null and points to writable caller
        // storage by function contract.
        unsafe {
            *out = string_view_from_str(tag);
        }
        Ok(())
    })
}

/// Returns one borrowed plane view.
///
/// # Safety
///
/// `bundle` must be valid and `out` must be non-null and writable. The returned
/// plane bytes are valid until the bundle handle is freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imqraw_bundle_image_plane(
    bundle: *const ImqrawBundle,
    image_index: usize,
    plane_index: usize,
    out: *mut ImqrawPlaneView,
    status: *mut ImqrawStatus,
) -> i32 {
    with_status(status, || {
        let record = image_record(bundle, image_index)?;
        if out.is_null() {
            return Err(CApiError::new(
                ImqrawStatusCode::NullPointer,
                "out plane pointer is null",
            ));
        }
        let plane = record
            .frame
            .owned_planes()
            .get(plane_index)
            .ok_or_else(|| {
                CApiError::new(
                    ImqrawStatusCode::InvalidArgument,
                    format!("plane index {plane_index} is out of range"),
                )
            })?;
        // SAFETY: `out` was checked non-null and points to writable caller
        // storage by function contract.
        unsafe {
            *out = ImqrawPlaneView {
                ptr: plane.data.as_ptr(),
                len: plane.data.len(),
                stride: plane.stride,
            };
        }
        Ok(())
    })
}

fn encode_single_image(
    image: *const ImqrawEncodeImage,
    out: *mut ImqrawByteBuffer,
    pixel_format: PixelFormat,
    status: *mut ImqrawStatus,
) -> i32 {
    encode_images(image, 1, out, Some(pixel_format), status)
}

fn encode_images(
    images: *const ImqrawEncodeImage,
    image_count: usize,
    out: *mut ImqrawByteBuffer,
    override_pixel_format: Option<PixelFormat>,
    status: *mut ImqrawStatus,
) -> i32 {
    with_status(status, || {
        if images.is_null() {
            return Err(CApiError::new(
                ImqrawStatusCode::NullPointer,
                "image array pointer is null",
            ));
        }
        if image_count == 0 {
            return Err(CApiError::new(
                ImqrawStatusCode::InvalidArgument,
                "image count must be greater than zero",
            ));
        }
        if out.is_null() {
            return Err(CApiError::new(
                ImqrawStatusCode::NullPointer,
                "out buffer pointer is null",
            ));
        }
        // SAFETY: `images` was checked non-null and the function contract
        // requires it to point to `image_count` readable input structs.
        let images = unsafe { slice::from_raw_parts(images, image_count) };
        let records = images
            .iter()
            .map(|image| image_record_for_encode(image, override_pixel_format))
            .collect::<Result<Vec<_>, _>>()?;
        let bundle = RawImageBundle::new(records);
        let bytes = encode_imqraw_bundle(&bundle)
            .map_err(|error| CApiError::new(ImqrawStatusCode::EncodeError, error.to_string()))?;
        // SAFETY: `out` was checked non-null and points to writable caller
        // storage by function contract.
        unsafe {
            *out = owned_buffer(bytes);
        }
        Ok(())
    })
}

fn image_record_for_encode(
    image: &ImqrawEncodeImage,
    override_pixel_format: Option<PixelFormat>,
) -> Result<RawImageRecord, CApiError> {
    let pixel_format = match override_pixel_format {
        Some(pixel_format) => pixel_format,
        None => pixel_format_from_code(image.pixel_format)?,
    };
    let format =
        format_spec_from_codes(pixel_format, image.color_space, image.transfer, image.range)?;
    let frame = frame_from_image(image, format)?;
    let label = optional_string(image.label)?;
    let tags = string_views(image.tags, image.tag_count)?;
    Ok(RawImageRecord::new(label, tags, frame))
}

fn frame_from_image(
    image: &ImqrawEncodeImage,
    format: FormatSpec,
) -> Result<FrameOwned, CApiError> {
    if image.plane_count == 0 {
        let data = byte_slice(image.data)?.to_vec();
        return FrameOwned::packed_tight(data, image.width, image.height, format.pixel_format)
            .map_err(|error| CApiError::new(ImqrawStatusCode::EncodeError, error.to_string()));
    }
    if image.planes.is_null() {
        return Err(CApiError::new(
            ImqrawStatusCode::NullPointer,
            "plane array pointer is null",
        ));
    }
    // SAFETY: The C API contract requires `planes` to address `plane_count`
    // readable plane descriptors.
    let planes = unsafe { slice::from_raw_parts(image.planes, image.plane_count) }
        .iter()
        .map(|plane| {
            Ok(OwnedPlane::new(
                byte_slice(plane.data)?.to_vec(),
                plane.stride,
            ))
        })
        .collect::<Result<Vec<_>, CApiError>>()?;
    FrameOwned::new(
        Dimensions::new(image.width, image.height)
            .map_err(|error| CApiError::new(ImqrawStatusCode::EncodeError, error.to_string()))?,
        format,
        planes,
    )
    .map_err(|error| CApiError::new(ImqrawStatusCode::EncodeError, error.to_string()))
}

fn with_status(status: *mut ImqrawStatus, f: impl FnOnce() -> Result<(), CApiError>) -> i32 {
    match f() {
        Ok(()) => {
            write_status(status, ImqrawStatusCode::Ok, "ok");
            ImqrawStatusCode::Ok as i32
        }
        Err(error) => {
            write_status(status, error.code, &error.message);
            error.code as i32
        }
    }
}

fn write_status(status: *mut ImqrawStatus, code: ImqrawStatusCode, message: &str) {
    if status.is_null() {
        return;
    }
    let mut out = ImqrawStatus {
        code: code as i32,
        message: [0; STATUS_MESSAGE_CAP],
    };
    message
        .as_bytes()
        .iter()
        .copied()
        .take(STATUS_MESSAGE_CAP - 1)
        .enumerate()
        .for_each(|(index, byte)| {
            out.message[index] = byte as c_char;
        });
    // SAFETY: The caller supplied an optional writable status pointer. Null was
    // handled above.
    unsafe {
        *status = out;
    }
}

fn owned_buffer(bytes: Vec<u8>) -> ImqrawByteBuffer {
    let mut bytes = ManuallyDrop::new(bytes);
    ImqrawByteBuffer {
        ptr: bytes.as_mut_ptr(),
        len: bytes.len(),
        cap: bytes.capacity(),
    }
}

fn byte_slice(view: ImqrawByteView) -> Result<&'static [u8], CApiError> {
    if view.ptr.is_null() {
        if view.len == 0 {
            return Ok(&[]);
        }
        return Err(CApiError::new(
            ImqrawStatusCode::NullPointer,
            "byte view pointer is null",
        ));
    }
    // SAFETY: The C API contract requires the pointer to be valid for `len`
    // readable bytes.
    Ok(unsafe { slice::from_raw_parts(view.ptr, view.len) })
}

fn string_view(view: ImqrawStringView) -> Result<&'static str, CApiError> {
    if view.ptr.is_null() {
        if view.len == 0 {
            return Ok("");
        }
        return Err(CApiError::new(
            ImqrawStatusCode::NullPointer,
            "string view pointer is null",
        ));
    }
    // SAFETY: The C API contract requires the pointer to be valid for `len`
    // readable bytes.
    let bytes = unsafe { slice::from_raw_parts(view.ptr.cast::<u8>(), view.len) };
    str::from_utf8(bytes).map_err(|_| {
        CApiError::new(
            ImqrawStatusCode::InvalidArgument,
            "string view is not valid UTF-8",
        )
    })
}

fn optional_string(view: ImqrawStringView) -> Result<Option<String>, CApiError> {
    let value = string_view(view)?;
    Ok((!value.is_empty()).then(|| value.to_string()))
}

fn string_views(ptr: *const ImqrawStringView, len: usize) -> Result<Vec<String>, CApiError> {
    if ptr.is_null() {
        if len == 0 {
            return Ok(Vec::new());
        }
        return Err(CApiError::new(
            ImqrawStatusCode::NullPointer,
            "tag array pointer is null",
        ));
    }
    // SAFETY: The C API contract requires `ptr` to address `len` readable string
    // views.
    let views = unsafe { slice::from_raw_parts(ptr, len) };
    views
        .iter()
        .map(|view| string_view(*view).map(str::to_string))
        .collect()
}

fn bundle_ref(ptr: *const ImqrawBundle) -> Result<&'static ImqrawBundle, CApiError> {
    if ptr.is_null() {
        return Err(CApiError::new(
            ImqrawStatusCode::NullPointer,
            "bundle pointer is null",
        ));
    }
    // SAFETY: The C API contract requires a valid handle from this library.
    Ok(unsafe { &*ptr })
}

fn image_record(
    bundle: *const ImqrawBundle,
    image_index: usize,
) -> Result<&'static imq::RawImageRecord, CApiError> {
    let bundle = bundle_ref(bundle)?;
    bundle.bundle.records.get(image_index).ok_or_else(|| {
        CApiError::new(
            ImqrawStatusCode::InvalidArgument,
            format!("image index {image_index} is out of range"),
        )
    })
}

fn empty_string_view() -> ImqrawStringView {
    ImqrawStringView {
        ptr: ptr::null(),
        len: 0,
    }
}

fn string_view_from_str(value: &str) -> ImqrawStringView {
    ImqrawStringView {
        ptr: value.as_ptr().cast(),
        len: value.len(),
    }
}

fn pixel_format_code(value: PixelFormat) -> u16 {
    match value {
        PixelFormat::Luma8 => 1,
        PixelFormat::Rgb8 => 2,
        PixelFormat::Rgba8 => 3,
        PixelFormat::Bgr8 => 4,
        PixelFormat::Bgra8 => 5,
        PixelFormat::Luma16Le => 6,
        PixelFormat::Rgb16Le => 7,
        PixelFormat::Rgba16Le => 8,
        PixelFormat::RgbF32 => 9,
        PixelFormat::RgbaF32 => 10,
        PixelFormat::Yuv444p8 => 11,
        PixelFormat::Yuv422p8 => 12,
        PixelFormat::Yuv420p8 => 13,
        PixelFormat::Nv12 => 14,
        _ => 65535,
    }
}

fn pixel_format_from_code(value: u16) -> Result<PixelFormat, CApiError> {
    match value {
        1 => Ok(PixelFormat::Luma8),
        2 => Ok(PixelFormat::Rgb8),
        3 => Ok(PixelFormat::Rgba8),
        4 => Ok(PixelFormat::Bgr8),
        5 => Ok(PixelFormat::Bgra8),
        6 => Ok(PixelFormat::Luma16Le),
        7 => Ok(PixelFormat::Rgb16Le),
        8 => Ok(PixelFormat::Rgba16Le),
        9 => Ok(PixelFormat::RgbF32),
        10 => Ok(PixelFormat::RgbaF32),
        11 => Ok(PixelFormat::Yuv444p8),
        12 => Ok(PixelFormat::Yuv422p8),
        13 => Ok(PixelFormat::Yuv420p8),
        14 => Ok(PixelFormat::Nv12),
        _ => Err(CApiError::new(
            ImqrawStatusCode::InvalidArgument,
            format!("unsupported pixel format code {value}"),
        )),
    }
}

fn color_space_code(value: ColorSpace) -> u16 {
    match value {
        ColorSpace::Srgb => 1,
        ColorSpace::LinearRgb => 2,
        ColorSpace::Bt601 => 3,
        ColorSpace::Bt709 => 4,
        ColorSpace::Bt2020 => 5,
        ColorSpace::Unknown => 65535,
    }
}

fn color_space_from_code(value: u16) -> Result<Option<ColorSpace>, CApiError> {
    match value {
        0 => Ok(None),
        1 => Ok(Some(ColorSpace::Srgb)),
        2 => Ok(Some(ColorSpace::LinearRgb)),
        3 => Ok(Some(ColorSpace::Bt601)),
        4 => Ok(Some(ColorSpace::Bt709)),
        5 => Ok(Some(ColorSpace::Bt2020)),
        65535 => Ok(Some(ColorSpace::Unknown)),
        _ => Err(CApiError::new(
            ImqrawStatusCode::InvalidArgument,
            format!("unsupported color-space code {value}"),
        )),
    }
}

fn transfer_code(value: Transfer) -> u16 {
    match value {
        Transfer::Srgb => 1,
        Transfer::Linear => 2,
        Transfer::Bt1886 => 3,
        Transfer::Hlg => 4,
        Transfer::Pq => 5,
        Transfer::Unknown => 65535,
    }
}

fn transfer_from_code(value: u16) -> Result<Option<Transfer>, CApiError> {
    match value {
        0 => Ok(None),
        1 => Ok(Some(Transfer::Srgb)),
        2 => Ok(Some(Transfer::Linear)),
        3 => Ok(Some(Transfer::Bt1886)),
        4 => Ok(Some(Transfer::Hlg)),
        5 => Ok(Some(Transfer::Pq)),
        65535 => Ok(Some(Transfer::Unknown)),
        _ => Err(CApiError::new(
            ImqrawStatusCode::InvalidArgument,
            format!("unsupported transfer code {value}"),
        )),
    }
}

fn color_range_code(value: ColorRange) -> u16 {
    match value {
        ColorRange::Full => 1,
        ColorRange::Limited => 2,
    }
}

fn color_range_from_code(value: u16) -> Result<Option<ColorRange>, CApiError> {
    match value {
        0 => Ok(None),
        1 => Ok(Some(ColorRange::Full)),
        2 => Ok(Some(ColorRange::Limited)),
        _ => Err(CApiError::new(
            ImqrawStatusCode::InvalidArgument,
            format!("unsupported color-range code {value}"),
        )),
    }
}

fn format_spec_from_codes(
    pixel_format: PixelFormat,
    color_space: u16,
    transfer: u16,
    range: u16,
) -> Result<FormatSpec, CApiError> {
    let default = FormatSpec::new(pixel_format);
    Ok(FormatSpec::with_metadata(
        pixel_format,
        color_space_from_code(color_space)?.unwrap_or(default.color_space),
        transfer_from_code(transfer)?.unwrap_or(default.transfer),
        color_range_from_code(range)?.unwrap_or(default.range),
    ))
}

#[derive(Debug)]
struct CApiError {
    code: ImqrawStatusCode,
    message: String,
}

impl CApiError {
    fn new(code: ImqrawStatusCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn str_view(value: &str) -> ImqrawStringView {
        ImqrawStringView {
            ptr: value.as_ptr().cast(),
            len: value.len(),
        }
    }

    #[test]
    fn encodes_and_decodes_rgba8_bundle() {
        let pixels = [0, 0, 0, 255, 255, 255, 255, 255];
        let tags = [str_view("reference")];
        let image = ImqrawEncodeImage {
            width: 2,
            height: 1,
            pixel_format: 3,
            color_space: 0,
            transfer: 0,
            range: 0,
            data: ImqrawByteView {
                ptr: pixels.as_ptr(),
                len: pixels.len(),
            },
            planes: ptr::null(),
            plane_count: 0,
            label: str_view("ref"),
            tags: tags.as_ptr(),
            tag_count: tags.len(),
        };
        let mut status = ImqrawStatus {
            code: -1,
            message: [0; STATUS_MESSAGE_CAP],
        };
        let mut encoded = ImqrawByteBuffer {
            ptr: ptr::null_mut(),
            len: 0,
            cap: 0,
        };

        let code = unsafe { imqraw_encode_rgba8(&image, &mut encoded, &mut status) };
        assert_eq!(code, ImqrawStatusCode::Ok as i32);
        assert!(!encoded.ptr.is_null());

        let mut bundle = ptr::null_mut();
        let code = unsafe {
            imqraw_bundle_decode(
                ImqrawByteView {
                    ptr: encoded.ptr,
                    len: encoded.len,
                },
                &mut bundle,
                &mut status,
            )
        };
        assert_eq!(code, ImqrawStatusCode::Ok as i32);

        let mut count = 0;
        let code = unsafe { imqraw_bundle_image_count(bundle, &mut count, &mut status) };
        assert_eq!(code, ImqrawStatusCode::Ok as i32);
        assert_eq!(count, 1);

        let mut info = ImqrawImageInfo {
            width: 0,
            height: 0,
            pixel_format: 0,
            color_space: 0,
            transfer: 0,
            range: 0,
            plane_count: 0,
        };
        let code = unsafe { imqraw_bundle_image_info(bundle, 0, &mut info, &mut status) };
        assert_eq!(code, ImqrawStatusCode::Ok as i32);
        assert_eq!(info.width, 2);
        assert_eq!(info.height, 1);
        assert_eq!(info.pixel_format, 3);
        assert_eq!(info.plane_count, 1);

        let mut plane = ImqrawPlaneView {
            ptr: ptr::null(),
            len: 0,
            stride: 0,
        };
        let code = unsafe { imqraw_bundle_image_plane(bundle, 0, 0, &mut plane, &mut status) };
        assert_eq!(code, ImqrawStatusCode::Ok as i32);
        assert_eq!(plane.len, pixels.len());

        unsafe {
            imqraw_bundle_free(bundle);
            imqraw_buffer_free(encoded);
        }
    }

    #[test]
    fn encodes_multiple_images_with_generic_api() {
        let rgba = [0, 0, 0, 255, 255, 255, 255, 255];
        let luma = [10, 20];
        let images = [
            ImqrawEncodeImage {
                width: 2,
                height: 1,
                pixel_format: 3,
                color_space: 0,
                transfer: 0,
                range: 0,
                data: ImqrawByteView {
                    ptr: rgba.as_ptr(),
                    len: rgba.len(),
                },
                planes: ptr::null(),
                plane_count: 0,
                label: str_view("rgba"),
                tags: ptr::null(),
                tag_count: 0,
            },
            ImqrawEncodeImage {
                width: 2,
                height: 1,
                pixel_format: 1,
                color_space: 0,
                transfer: 0,
                range: 0,
                data: ImqrawByteView {
                    ptr: luma.as_ptr(),
                    len: luma.len(),
                },
                planes: ptr::null(),
                plane_count: 0,
                label: str_view("luma"),
                tags: ptr::null(),
                tag_count: 0,
            },
        ];
        let mut status = ImqrawStatus {
            code: -1,
            message: [0; STATUS_MESSAGE_CAP],
        };
        let mut encoded = ImqrawByteBuffer {
            ptr: ptr::null_mut(),
            len: 0,
            cap: 0,
        };

        let code = unsafe {
            imqraw_encode_bundle(images.as_ptr(), images.len(), &mut encoded, &mut status)
        };
        assert_eq!(code, ImqrawStatusCode::Ok as i32);

        let mut bundle = ptr::null_mut();
        let code = unsafe {
            imqraw_bundle_decode(
                ImqrawByteView {
                    ptr: encoded.ptr,
                    len: encoded.len,
                },
                &mut bundle,
                &mut status,
            )
        };
        assert_eq!(code, ImqrawStatusCode::Ok as i32);

        let mut count = 0;
        let code = unsafe { imqraw_bundle_image_count(bundle, &mut count, &mut status) };
        assert_eq!(code, ImqrawStatusCode::Ok as i32);
        assert_eq!(count, 2);

        unsafe {
            imqraw_bundle_free(bundle);
            imqraw_buffer_free(encoded);
        }
    }
}
