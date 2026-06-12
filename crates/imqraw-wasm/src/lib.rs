//! WebAssembly bindings for `imqraw` browser and Node pipelines.

use imq::{
    Dimensions, FormatSpec, FrameOwned, OwnedPlane, PixelFormat, RawImageBundle, RawImageRecord,
    decode_imqraw_bundle, encode_imqraw_bundle,
};
use js_sys::{Array, Reflect, Uint8Array};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

/// Returns the wasm package version.
#[wasm_bindgen]
pub fn imqraw_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Encodes one tightly packed RGBA8 image as an `imqraw` bundle.
#[wasm_bindgen]
pub fn encode_imqraw_rgba8(
    data: &[u8],
    width: u32,
    height: u32,
    label: String,
    tags: Array,
) -> Result<Vec<u8>, JsValue> {
    let record = rgba8_record(data.to_vec(), width, height, label, parse_tags(&tags)?)?;
    encode_imqraw_bundle(&RawImageBundle::new(vec![record])).map_err(js_error)
}

/// Encodes an array of `{ data, width, height, label?, tags? }` RGBA8 records.
#[wasm_bindgen]
pub fn encode_imqraw_rgba8_bundle(images: Array) -> Result<Vec<u8>, JsValue> {
    let records = images
        .iter()
        .map(|value| {
            let data = get_uint8_array(&value, "data")?.to_vec();
            let width = get_u32(&value, "width")?;
            let height = get_u32(&value, "height")?;
            let label = get_optional_string(&value, "label")?.unwrap_or_default();
            let tags = get_optional_tags(&value, "tags")?;
            rgba8_record(data, width, height, label, tags)
        })
        .collect::<Result<Vec<_>, JsValue>>()?;
    encode_imqraw_bundle(&RawImageBundle::new(records)).map_err(js_error)
}

/// Encodes one tightly packed image as an `imqraw` bundle using a stable pixel-format code.
#[wasm_bindgen]
pub fn encode_imqraw_image(
    data: &[u8],
    width: u32,
    height: u32,
    pixel_format: u16,
    stride: usize,
    label: String,
    tags: Array,
) -> Result<Vec<u8>, JsValue> {
    let pixel_format = pixel_format_from_code(pixel_format)?;
    let record = image_record(
        data.to_vec(),
        width,
        height,
        pixel_format,
        (stride != 0).then_some(stride),
        label,
        parse_tags(&tags)?,
    )?;
    encode_imqraw_bundle(&RawImageBundle::new(vec![record])).map_err(js_error)
}

/// Returns the number of images in an encoded `imqraw` bundle.
#[wasm_bindgen]
pub fn imqraw_image_count(bytes: &[u8]) -> Result<usize, JsValue> {
    decode_imqraw_bundle(bytes)
        .map(|bundle| bundle.records.len())
        .map_err(js_error)
}

fn rgba8_record(
    data: Vec<u8>,
    width: u32,
    height: u32,
    label: String,
    tags: Vec<String>,
) -> Result<RawImageRecord, JsValue> {
    let frame =
        FrameOwned::packed_tight(data, width, height, PixelFormat::Rgba8).map_err(js_error)?;
    Ok(RawImageRecord::new(non_empty(label), tags, frame))
}

fn image_record(
    data: Vec<u8>,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    stride: Option<usize>,
    label: String,
    tags: Vec<String>,
) -> Result<RawImageRecord, JsValue> {
    let frame = if pixel_format.is_bit_packed() {
        let row_bytes = usize::try_from(width)
            .map_err(|_| JsValue::from_str("width overflows usize"))?
            .div_ceil(8);
        let stride = stride.unwrap_or(row_bytes);
        FrameOwned::new(
            Dimensions::new(width, height).map_err(js_error)?,
            FormatSpec::new(pixel_format),
            vec![OwnedPlane::new(data, stride)],
        )
    } else if let Some(stride) = stride {
        FrameOwned::packed(data, width, height, pixel_format, stride)
    } else {
        FrameOwned::packed_tight(data, width, height, pixel_format)
    }
    .map_err(js_error)?;
    Ok(RawImageRecord::new(non_empty(label), tags, frame))
}

fn pixel_format_from_code(code: u16) -> Result<PixelFormat, JsValue> {
    match code {
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
        15 => Ok(PixelFormat::Hsv8),
        16 => Ok(PixelFormat::Hsva8),
        17 => Ok(PixelFormat::Binary1Lsb),
        18 => Ok(PixelFormat::Binary1Msb),
        _ => Err(JsValue::from_str(&format!(
            "unsupported imqraw pixel format code {code}"
        ))),
    }
}

fn non_empty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn get_property(value: &JsValue, name: &str) -> Result<JsValue, JsValue> {
    Reflect::get(value, &JsValue::from_str(name))
}

fn get_uint8_array(value: &JsValue, name: &str) -> Result<Uint8Array, JsValue> {
    get_property(value, name)?
        .dyn_into::<Uint8Array>()
        .map_err(|_| JsValue::from_str(&format!("property `{name}` must be a Uint8Array")))
}

fn get_u32(value: &JsValue, name: &str) -> Result<u32, JsValue> {
    let number = get_property(value, name)?
        .as_f64()
        .ok_or_else(|| JsValue::from_str(&format!("property `{name}` must be a number")))?;
    if !number.is_finite()
        || number.fract() != 0.0
        || !(1.0..=f64::from(u32::MAX)).contains(&number)
    {
        return Err(JsValue::from_str(&format!(
            "property `{name}` must be an integer in 1..=u32::MAX"
        )));
    }
    Ok(number as u32)
}

fn get_optional_string(value: &JsValue, name: &str) -> Result<Option<String>, JsValue> {
    let property = get_property(value, name)?;
    if property.is_undefined() || property.is_null() {
        return Ok(None);
    }
    property
        .as_string()
        .map(Some)
        .ok_or_else(|| JsValue::from_str(&format!("property `{name}` must be a string")))
}

fn get_optional_tags(value: &JsValue, name: &str) -> Result<Vec<String>, JsValue> {
    let property = get_property(value, name)?;
    if property.is_undefined() || property.is_null() {
        return Ok(Vec::new());
    }
    if !Array::is_array(&property) {
        return Err(JsValue::from_str(&format!(
            "property `{name}` must be an array of strings"
        )));
    }
    parse_tags(&Array::from(&property))
}

fn parse_tags(tags: &Array) -> Result<Vec<String>, JsValue> {
    tags.iter()
        .map(|tag| {
            tag.as_string()
                .ok_or_else(|| JsValue::from_str("tags must be strings"))
        })
        .collect()
}

fn js_error(error: impl ToString) -> JsValue {
    JsValue::from_str(&error.to_string())
}
