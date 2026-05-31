//! WebAssembly bindings for `imqraw` browser and Node pipelines.

use imq::{
    FrameOwned, PixelFormat, RawImageBundle, RawImageRecord, decode_imqraw_bundle,
    encode_imqraw_bundle,
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
