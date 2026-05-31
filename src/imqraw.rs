//! Lossless raw image bundle encoding for cross-platform stdin/stdout pipelines.
//!
//! `imqraw` is intentionally small and dependency-free. It stores validated
//! [`FrameOwned`] planes verbatim with little-endian metadata, optional labels,
//! and arbitrary tags. It is meant for local pipes, tests, and tool-to-tool
//! interchange where codec artifacts and platform-specific image containers are
//! undesirable.

use crate::frame::{
    ColorRange, ColorSpace, Dimensions, FormatSpec, FrameOwned, OwnedPlane, PixelFormat, Transfer,
};
use crate::{Error, Result};

/// Magic bytes at the beginning of an `imqraw` bundle.
pub const MAGIC: &[u8; 8] = b"IMQRAW1\n";

/// A decoded `imqraw` bundle.
#[derive(Debug, Clone)]
pub struct RawImageBundle {
    /// Images in stream order.
    pub records: Vec<RawImageRecord>,
}

impl RawImageBundle {
    /// Creates a bundle from image records.
    pub fn new(records: Vec<RawImageRecord>) -> Self {
        Self { records }
    }

    /// Returns the record at a zero-based index.
    pub fn select_index(&self, index: usize) -> Result<&RawImageRecord> {
        self.records.get(index).ok_or_else(|| {
            Error::unsupported(format!(
                "imqraw image index {index} is out of range; bundle contains {} image(s)",
                self.records.len()
            ))
        })
    }

    /// Returns the first record containing the requested tag.
    pub fn select_tag(&self, tag: &str) -> Result<&RawImageRecord> {
        self.records
            .iter()
            .find(|record| record.tags.iter().any(|candidate| candidate == tag))
            .ok_or_else(|| Error::unsupported(format!("imqraw tag `{tag}` was not found")))
    }

    /// Returns the record selected by tag or index.
    pub fn select(&self, selector: &RawImageSelector) -> Result<&RawImageRecord> {
        match selector {
            RawImageSelector::Index(index) => self.select_index(*index),
            RawImageSelector::Tag(tag) => self.select_tag(tag),
        }
    }
}

/// One image inside an `imqraw` bundle.
#[derive(Debug, Clone)]
pub struct RawImageRecord {
    /// Optional human-readable label, commonly the source path.
    pub label: Option<String>,
    /// Caller-defined tags used for selection and grouping.
    pub tags: Vec<String>,
    /// Validated image frame.
    pub frame: FrameOwned,
}

impl RawImageRecord {
    /// Creates a record from a validated frame.
    pub fn new(label: Option<String>, tags: Vec<String>, frame: FrameOwned) -> Self {
        Self { label, tags, frame }
    }
}

/// A selector for an image in a bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawImageSelector {
    /// Select by zero-based image index.
    Index(usize),
    /// Select the first image containing this tag.
    Tag(String),
}

/// Encodes a raw image bundle into bytes.
pub fn encode_bundle(bundle: &RawImageBundle) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    output.extend_from_slice(MAGIC);
    push_u32(
        &mut output,
        checked_u32(bundle.records.len(), "record count")?,
    );
    for record in &bundle.records {
        encode_record(&mut output, record)?;
    }
    Ok(output)
}

/// Decodes a raw image bundle from bytes.
pub fn decode_bundle(bytes: &[u8]) -> Result<RawImageBundle> {
    let mut reader = Reader::new(bytes);
    let magic = reader.read_exact(MAGIC.len())?;
    if magic != MAGIC {
        return Err(Error::unsupported("input is not an imqraw bundle"));
    }
    let record_count = checked_usize(reader.read_u32()?, "record count")?;
    let records = (0..record_count)
        .map(|_| decode_record(&mut reader))
        .collect::<Result<Vec<_>>>()?;
    if !reader.is_empty() {
        return Err(Error::unsupported("imqraw bundle has trailing bytes"));
    }
    Ok(RawImageBundle { records })
}

#[cfg(feature = "imqraw-image")]
#[cfg_attr(docsrs, doc(cfg(feature = "imqraw-image")))]
/// Creates an `imqraw` record from an `image::DynamicImage`, normalized to RGBA8.
pub fn record_from_dynamic_image(
    label: Option<String>,
    tags: Vec<String>,
    image: &image::DynamicImage,
) -> Result<RawImageRecord> {
    let frame = crate::adapters::image_crate::from_dynamic_image_rgba8(image)?;
    Ok(RawImageRecord::new(label, tags, frame))
}

#[cfg(feature = "imqraw-image")]
#[cfg_attr(docsrs, doc(cfg(feature = "imqraw-image")))]
/// Creates an `imqraw` record from an `image::RgbaImage` without an extra pixel copy.
pub fn record_from_rgba_image(
    label: Option<String>,
    tags: Vec<String>,
    image: image::RgbaImage,
) -> Result<RawImageRecord> {
    let frame = crate::adapters::image_crate::from_rgba_image(image)?;
    Ok(RawImageRecord::new(label, tags, frame))
}

#[cfg(feature = "imqraw-image")]
#[cfg_attr(docsrs, doc(cfg(feature = "imqraw-image")))]
/// Creates an `imqraw` record from an `image::RgbImage` without an extra pixel copy.
pub fn record_from_rgb_image(
    label: Option<String>,
    tags: Vec<String>,
    image: image::RgbImage,
) -> Result<RawImageRecord> {
    let frame = crate::adapters::image_crate::from_rgb_image(image)?;
    Ok(RawImageRecord::new(label, tags, frame))
}

fn encode_record(output: &mut Vec<u8>, record: &RawImageRecord) -> Result<()> {
    let dims = record.frame.dimensions();
    let format = record.frame.format();
    push_u32(output, dims.width);
    push_u32(output, dims.height);
    push_u16(output, pixel_format_code(format.pixel_format));
    push_u16(output, color_space_code(format.color_space));
    push_u16(output, transfer_code(format.transfer));
    push_u16(output, color_range_code(format.range));
    push_string(output, record.label.as_deref())?;
    push_u32(output, checked_u32(record.tags.len(), "tag count")?);
    push_u32(
        output,
        checked_u32(record.frame.owned_planes().len(), "plane count")?,
    );
    for tag in &record.tags {
        push_required_string(output, tag)?;
    }
    for plane in record.frame.owned_planes() {
        push_u64(output, checked_u64(plane.stride, "plane stride")?);
        push_u64(output, checked_u64(plane.data.len(), "plane data length")?);
        output.extend_from_slice(&plane.data);
    }
    Ok(())
}

fn decode_record(reader: &mut Reader<'_>) -> Result<RawImageRecord> {
    let width = reader.read_u32()?;
    let height = reader.read_u32()?;
    let pixel_format = pixel_format_from_code(reader.read_u16()?)?;
    let color_space = color_space_from_code(reader.read_u16()?)?;
    let transfer = transfer_from_code(reader.read_u16()?)?;
    let range = color_range_from_code(reader.read_u16()?)?;
    let label = reader.read_optional_string()?;
    let tag_count = checked_usize(reader.read_u32()?, "tag count")?;
    let plane_count = checked_usize(reader.read_u32()?, "plane count")?;
    let tags = (0..tag_count)
        .map(|_| reader.read_required_string())
        .collect::<Result<Vec<_>>>()?;
    let planes = (0..plane_count)
        .map(|_| {
            let stride = checked_usize(reader.read_u64()?, "plane stride")?;
            let data_len = checked_usize(reader.read_u64()?, "plane data length")?;
            let data = reader.read_exact(data_len)?.to_vec();
            Ok(OwnedPlane::new(data, stride))
        })
        .collect::<Result<Vec<_>>>()?;
    let frame = FrameOwned::new(
        Dimensions::new(width, height)?,
        FormatSpec::with_metadata(pixel_format, color_space, transfer, range),
        planes,
    )?;
    Ok(RawImageRecord { label, tags, frame })
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
    }
}

fn pixel_format_from_code(code: u16) -> Result<PixelFormat> {
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
        _ => Err(Error::unsupported(format!(
            "unknown imqraw pixel format code {code}"
        ))),
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

fn color_space_from_code(code: u16) -> Result<ColorSpace> {
    match code {
        1 => Ok(ColorSpace::Srgb),
        2 => Ok(ColorSpace::LinearRgb),
        3 => Ok(ColorSpace::Bt601),
        4 => Ok(ColorSpace::Bt709),
        5 => Ok(ColorSpace::Bt2020),
        65535 => Ok(ColorSpace::Unknown),
        _ => Err(Error::unsupported(format!(
            "unknown imqraw color space code {code}"
        ))),
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

fn transfer_from_code(code: u16) -> Result<Transfer> {
    match code {
        1 => Ok(Transfer::Srgb),
        2 => Ok(Transfer::Linear),
        3 => Ok(Transfer::Bt1886),
        4 => Ok(Transfer::Hlg),
        5 => Ok(Transfer::Pq),
        65535 => Ok(Transfer::Unknown),
        _ => Err(Error::unsupported(format!(
            "unknown imqraw transfer code {code}"
        ))),
    }
}

fn color_range_code(value: ColorRange) -> u16 {
    match value {
        ColorRange::Full => 1,
        ColorRange::Limited => 2,
    }
}

fn color_range_from_code(code: u16) -> Result<ColorRange> {
    match code {
        1 => Ok(ColorRange::Full),
        2 => Ok(ColorRange::Limited),
        _ => Err(Error::unsupported(format!(
            "unknown imqraw color range code {code}"
        ))),
    }
}

fn push_string(output: &mut Vec<u8>, value: Option<&str>) -> Result<()> {
    match value {
        Some(value) => push_required_string(output, value),
        None => {
            push_u32(output, u32::MAX);
            Ok(())
        }
    }
}

fn push_required_string(output: &mut Vec<u8>, value: &str) -> Result<()> {
    push_u32(output, checked_u32(value.len(), "string length")?);
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn checked_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::unsupported(format!("imqraw {label} exceeds u32")))
}

fn checked_u64(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::unsupported(format!("imqraw {label} exceeds u64")))
}

fn checked_usize<T>(value: T, label: &str) -> Result<usize>
where
    usize: TryFrom<T>,
{
    usize::try_from(value).map_err(|_| Error::unsupported(format!("imqraw {label} exceeds usize")))
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| Error::unsupported("imqraw offset overflow"))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| Error::unsupported("truncated imqraw bundle"))?;
        self.offset = end;
        Ok(slice)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_optional_string(&mut self) -> Result<Option<String>> {
        let len = self.read_u32()?;
        if len == u32::MAX {
            return Ok(None);
        }
        self.read_string(checked_usize(len, "string length")?)
            .map(Some)
    }

    fn read_required_string(&mut self) -> Result<String> {
        let len = checked_usize(self.read_u32()?, "string length")?;
        self.read_string(len)
    }

    fn read_string(&mut self, len: usize) -> Result<String> {
        let bytes = self.read_exact(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| Error::unsupported("imqraw string is not valid UTF-8"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_record(label: &str, tag: &str, pixel: [u8; 4]) -> RawImageRecord {
        RawImageRecord::new(
            Some(label.to_string()),
            vec![tag.to_string()],
            FrameOwned::packed_tight(pixel.to_vec(), 1, 1, PixelFormat::Rgba8).unwrap(),
        )
    }

    #[test]
    fn roundtrips_single_rgba_record() {
        let bundle = RawImageBundle::new(vec![tiny_record("ref", "reference", [1, 2, 3, 255])]);
        let decoded = decode_bundle(&encode_bundle(&bundle).unwrap()).unwrap();

        assert_eq!(decoded.records.len(), 1);
        assert_eq!(decoded.records[0].label.as_deref(), Some("ref"));
        assert_eq!(decoded.records[0].tags, ["reference"]);
        assert_eq!(
            decoded.records[0].frame.owned_planes()[0].data,
            [1, 2, 3, 255]
        );
    }

    #[test]
    fn roundtrips_multiple_records_and_tag_selection() {
        let bundle = RawImageBundle::new(vec![
            tiny_record("ref", "reference", [0, 0, 0, 255]),
            tiny_record("dist", "candidate", [255, 0, 0, 255]),
        ]);
        let decoded = decode_bundle(&encode_bundle(&bundle).unwrap()).unwrap();

        assert_eq!(
            decoded.select_index(1).unwrap().label.as_deref(),
            Some("dist")
        );
        assert_eq!(
            decoded
                .select_tag("candidate")
                .unwrap()
                .frame
                .owned_planes()[0]
                .data,
            [255, 0, 0, 255]
        );
    }

    #[test]
    fn rejects_bad_magic() {
        let error = decode_bundle(b"not imqraw").unwrap_err();
        assert!(error.to_string().contains("not an imqraw"));
    }
}
