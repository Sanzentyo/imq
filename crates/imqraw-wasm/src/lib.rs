//! WebAssembly bindings for `imqraw` browser and Node pipelines.

use imq::{
    ColorRange, ColorSpace, Dimensions, Direction, FormatSpec, FrameOwned, MetricSet, OwnedPlane,
    PixelFormat, RawImageBundle, RawImageRecord, Transfer, decode_imqraw_bundle as decode_bundle,
    decode_imqraw_record, encode_imqraw_bundle, imqraw_find_tag as find_imqraw_tag,
    imqraw_record_count,
};
use js_sys::{Array, Object, Reflect, Uint8Array};
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
    imqraw_record_count(bytes).map_err(js_error)
}

/// Returns the index of the first image carrying `tag`.
#[wasm_bindgen]
pub fn imqraw_find_tag(bytes: &[u8], tag: &str) -> Result<usize, JsValue> {
    find_imqraw_tag(bytes, tag).map_err(js_error)
}

/// Decodes all images into JavaScript records with copied `Uint8Array` planes.
#[wasm_bindgen]
pub fn decode_imqraw_images(bytes: &[u8]) -> Result<Array, JsValue> {
    let bundle = decode_bundle(bytes).map_err(js_error)?;
    let records = Array::new();
    for (index, record) in bundle.records.iter().enumerate() {
        records.push(&record_to_js(record, index)?);
    }
    Ok(records)
}

/// Decodes one image by zero-based index into a JavaScript record.
#[wasm_bindgen]
pub fn decode_imqraw_image(bytes: &[u8], index: usize) -> Result<JsValue, JsValue> {
    let record = decode_imqraw_record(bytes, index).map_err(js_error)?;
    record_to_js(&record, index)
}

/// Compares two records in one bundle and returns metric result objects.
#[wasm_bindgen]
pub fn compare_imqraw_images(
    bytes: &[u8],
    reference_index: usize,
    distorted_index: usize,
    metrics: &str,
) -> Result<Array, JsValue> {
    let reference = decode_imqraw_record(bytes, reference_index).map_err(js_error)?;
    let distorted = decode_imqraw_record(bytes, distorted_index).map_err(js_error)?;
    let metric_set = if metrics.trim().is_empty() {
        MetricSet::defaults()
    } else {
        MetricSet::from_csv(metrics).map_err(js_error)?
    };
    let outputs = metric_set
        .compare(&reference.frame.as_view(), &distorted.frame.as_view())
        .map_err(js_error)?;
    let results = Array::new();
    for output in outputs {
        let object = Object::new();
        set(&object, "name", &JsValue::from_str(&output.name))?;
        set(&object, "score", &JsValue::from_f64(output.score))?;
        set(&object, "unit", &JsValue::from_str(&output.unit))?;
        let direction = match output.direction {
            Direction::HigherIsBetter => "higher-is-better",
            Direction::LowerIsBetter => "lower-is-better",
            Direction::Neutral => "neutral",
        };
        set(&object, "direction", &JsValue::from_str(direction))?;
        let details = Object::new();
        for (name, value) in output.details {
            set(&details, &name, &JsValue::from_f64(value))?;
        }
        set(&object, "details", &details)?;
        results.push(&object);
    }
    Ok(results)
}

fn record_to_js(record: &RawImageRecord, index: usize) -> Result<JsValue, JsValue> {
    let object = Object::new();
    let dimensions = record.frame.dimensions();
    let format = record.frame.format();
    set(&object, "index", &JsValue::from_f64(index as f64))?;
    set(
        &object,
        "width",
        &JsValue::from_f64(f64::from(dimensions.width)),
    )?;
    set(
        &object,
        "height",
        &JsValue::from_f64(f64::from(dimensions.height)),
    )?;
    set(
        &object,
        "pixelFormat",
        &JsValue::from_f64(f64::from(pixel_format_code(format.pixel_format))),
    )?;
    set(
        &object,
        "colorSpace",
        &JsValue::from_f64(f64::from(color_space_code(format.color_space))),
    )?;
    set(
        &object,
        "transfer",
        &JsValue::from_f64(f64::from(transfer_code(format.transfer))),
    )?;
    set(
        &object,
        "range",
        &JsValue::from_f64(f64::from(color_range_code(format.range))),
    )?;
    match &record.label {
        Some(label) => set(&object, "label", &JsValue::from_str(label))?,
        None => set(&object, "label", &JsValue::NULL)?,
    }
    let tags = Array::new();
    for tag in &record.tags {
        tags.push(&JsValue::from_str(tag));
    }
    set(&object, "tags", &tags)?;
    let planes = Array::new();
    for plane in record.frame.owned_planes() {
        let plane_object = Object::new();
        let data = Uint8Array::from(plane.data.as_slice());
        set(&plane_object, "data", &data)?;
        set(
            &plane_object,
            "stride",
            &JsValue::from_f64(plane.stride as f64),
        )?;
        planes.push(&plane_object);
    }
    set(&object, "planes", &planes)?;
    if record.frame.owned_planes().len() == 1 {
        let plane = &record.frame.owned_planes()[0];
        set(&object, "data", &Uint8Array::from(plane.data.as_slice()))?;
        set(&object, "stride", &JsValue::from_f64(plane.stride as f64))?;
    }
    Ok(object.into())
}

fn set(object: &Object, name: &str, value: &JsValue) -> Result<(), JsValue> {
    Reflect::set(object, &JsValue::from_str(name), value).map(|_| ())
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
        PixelFormat::Hsv8 => 15,
        PixelFormat::Hsva8 => 16,
        PixelFormat::Binary1Lsb => 17,
        PixelFormat::Binary1Msb => 18,
        _ => u16::MAX,
    }
}

fn color_space_code(value: ColorSpace) -> u16 {
    match value {
        ColorSpace::Srgb => 1,
        ColorSpace::LinearRgb => 2,
        ColorSpace::Bt601 => 3,
        ColorSpace::Bt709 => 4,
        ColorSpace::Bt2020 => 5,
        ColorSpace::Unknown => u16::MAX,
    }
}

fn transfer_code(value: Transfer) -> u16 {
    match value {
        Transfer::Srgb => 1,
        Transfer::Linear => 2,
        Transfer::Bt1886 => 3,
        Transfer::Hlg => 4,
        Transfer::Pq => 5,
        Transfer::Unknown => u16::MAX,
    }
}

fn color_range_code(value: ColorRange) -> u16 {
    match value {
        ColorRange::Full => 1,
        ColorRange::Limited => 2,
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
