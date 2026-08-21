//! Zero-copy frame representations, pixel-format metadata, and typestate validation.

use crate::{Error, Result};
use std::marker::PhantomData;

/// A width/height pair in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dimensions {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Dimensions {
    /// Creates non-zero dimensions.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid_frame("width and height must be non-zero"));
        }
        Ok(Self { width, height })
    }

    /// Number of pixels, checked into `usize`.
    pub fn pixels(self) -> Result<usize> {
        let px = u64::from(self.width) * u64::from(self.height);
        usize::try_from(px).map_err(|_| Error::invalid_frame("pixel count overflows usize"))
    }

    /// Returns `(width, height)` as `usize`.
    pub fn as_usize(self) -> Result<(usize, usize)> {
        Ok((
            usize::try_from(self.width)
                .map_err(|_| Error::invalid_frame("width overflows usize"))?,
            usize::try_from(self.height)
                .map_err(|_| Error::invalid_frame("height overflows usize"))?,
        ))
    }
}

/// RGB/YUV color space metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColorSpace {
    /// sRGB primaries and transfer.
    #[default]
    Srgb,
    /// Linear RGB with Rec.709/sRGB primaries.
    LinearRgb,
    /// ITU-R BT.601 YCbCr.
    Bt601,
    /// ITU-R BT.709 YCbCr / HD video.
    Bt709,
    /// ITU-R BT.2020 YCbCr / UHD video.
    Bt2020,
    /// Unknown or caller-managed color space.
    Unknown,
}

/// Transfer function metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Transfer {
    /// sRGB transfer curve.
    #[default]
    Srgb,
    /// Linear samples.
    Linear,
    /// BT.1886/BT.709-ish video gamma.
    Bt1886,
    /// Hybrid log-gamma.
    Hlg,
    /// Perceptual quantizer.
    Pq,
    /// Unknown/caller-managed transfer.
    Unknown,
}

/// Marker trait for range metadata.
pub trait RangeMarker: sealed::Sealed + Copy + Clone + Default + std::fmt::Debug + 'static {
    /// Concrete enum value for runtime APIs.
    const VALUE: ColorRange;
}

/// Full-range samples, e.g. image RGB 0..255.
#[derive(Debug, Clone, Copy, Default)]
pub struct FullRange;

/// Studio/video limited-range samples.
#[derive(Debug, Clone, Copy, Default)]
pub struct LimitedRange;

impl RangeMarker for FullRange {
    const VALUE: ColorRange = ColorRange::Full;
}

impl RangeMarker for LimitedRange {
    const VALUE: ColorRange = ColorRange::Limited;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::FullRange {}
    impl Sealed for super::LimitedRange {}
}

/// Runtime color-range enum.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColorRange {
    /// Full-range samples.
    #[default]
    Full,
    /// Limited/video-range samples.
    Limited,
}

/// YUV chroma sampling layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChromaSampling {
    /// 4:4:4, no chroma subsampling.
    Cs444,
    /// 4:2:2, horizontal chroma subsampling.
    Cs422,
    /// 4:2:0, horizontal and vertical chroma subsampling.
    Cs420,
    /// NV12: Y plane + interleaved UV 4:2:0 plane.
    Nv12,
}

/// Supported in-memory pixel formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum PixelFormat {
    /// 8-bit luma, packed single channel.
    Luma8,
    /// 8-bit RGB, packed.
    Rgb8,
    /// 8-bit RGBA, packed.
    Rgba8,
    /// 8-bit BGR, packed.
    Bgr8,
    /// 8-bit BGRA, packed.
    Bgra8,
    /// 16-bit luma, little-endian, packed single channel.
    Luma16Le,
    /// 16-bit RGB, little-endian, packed.
    Rgb16Le,
    /// 16-bit RGBA, little-endian, packed.
    Rgba16Le,
    /// 32-bit float RGB, native-endian, packed.
    RgbF32,
    /// 32-bit float RGBA, native-endian, packed.
    RgbaF32,
    /// Planar YUV 4:4:4, 8-bit.
    Yuv444p8,
    /// Planar YUV 4:2:2, 8-bit.
    Yuv422p8,
    /// Planar YUV 4:2:0, 8-bit.
    Yuv420p8,
    /// NV12, 8-bit: Y plane + interleaved UV plane.
    Nv12,
    /// 8-bit HSV, packed. Hue, saturation, and value are normalized over 0..255.
    Hsv8,
    /// 8-bit HSVA, packed. Hue, saturation, value, and alpha are normalized over 0..255.
    Hsva8,
    /// 1-bit binary mask, one row at a time, least-significant bit first.
    Binary1Lsb,
    /// 1-bit binary mask, one row at a time, most-significant bit first.
    Binary1Msb,
}

impl PixelFormat {
    /// Returns `true` for packed single-plane layouts.
    pub fn is_packed(self) -> bool {
        matches!(
            self,
            Self::Luma8
                | Self::Rgb8
                | Self::Rgba8
                | Self::Bgr8
                | Self::Bgra8
                | Self::Luma16Le
                | Self::Rgb16Le
                | Self::Rgba16Le
                | Self::RgbF32
                | Self::RgbaF32
                | Self::Hsv8
                | Self::Hsva8
        )
    }

    /// Returns `true` for 1-bit packed mask layouts.
    pub fn is_bit_packed(self) -> bool {
        matches!(self, Self::Binary1Lsb | Self::Binary1Msb)
    }

    /// Returns `true` for planar/interleaved YUV layouts.
    pub fn is_yuv(self) -> bool {
        matches!(
            self,
            Self::Yuv444p8 | Self::Yuv422p8 | Self::Yuv420p8 | Self::Nv12
        )
    }

    /// Number of planes.
    pub fn plane_count(self) -> usize {
        match self {
            Self::Yuv444p8 | Self::Yuv422p8 | Self::Yuv420p8 => 3,
            Self::Nv12 => 2,
            _ => 1,
        }
    }

    /// Bytes per pixel for packed layouts.
    pub fn bytes_per_pixel(self) -> Option<usize> {
        match self {
            Self::Luma8 => Some(1),
            Self::Rgb8 | Self::Bgr8 | Self::Hsv8 => Some(3),
            Self::Rgba8 | Self::Bgra8 | Self::Hsva8 => Some(4),
            Self::Luma16Le => Some(2),
            Self::Rgb16Le => Some(6),
            Self::Rgba16Le => Some(8),
            Self::RgbF32 => Some(12),
            Self::RgbaF32 => Some(16),
            _ => None,
        }
    }

    /// Logical channels included by the format.
    pub fn channel_count(self) -> usize {
        match self {
            Self::Luma8 | Self::Luma16Le => 1,
            Self::Rgb8 | Self::Bgr8 | Self::Rgb16Le | Self::RgbF32 | Self::Hsv8 => 3,
            Self::Rgba8 | Self::Bgra8 | Self::Rgba16Le | Self::RgbaF32 | Self::Hsva8 => 4,
            Self::Yuv444p8 | Self::Yuv422p8 | Self::Yuv420p8 => 3,
            Self::Nv12 => 3,
            Self::Binary1Lsb | Self::Binary1Msb => 1,
        }
    }

    /// Maximum code value used to normalize integer samples.
    pub fn max_code_value(self) -> f64 {
        match self {
            Self::Luma16Le | Self::Rgb16Le | Self::Rgba16Le => 65_535.0,
            Self::RgbF32 | Self::RgbaF32 => 1.0,
            Self::Binary1Lsb | Self::Binary1Msb => 1.0,
            _ => 255.0,
        }
    }

    /// Suggested color space for the format.
    pub fn default_color_space(self) -> ColorSpace {
        if self.is_yuv() {
            ColorSpace::Bt709
        } else {
            ColorSpace::Srgb
        }
    }

    /// Returns the plane dimensions for a plane index.
    pub fn plane_dimensions(self, dims: Dimensions, plane: usize) -> Result<Dimensions> {
        let (w, h) = dims.as_usize()?;
        let (pw, ph) = match self {
            Self::Yuv420p8 => match plane {
                0 => (w, h),
                1 | 2 => (w.div_ceil(2), h.div_ceil(2)),
                _ => return Err(Error::invalid_frame("YUV420p has exactly three planes")),
            },
            Self::Yuv422p8 => match plane {
                0 => (w, h),
                1 | 2 => (w.div_ceil(2), h),
                _ => return Err(Error::invalid_frame("YUV422p has exactly three planes")),
            },
            Self::Yuv444p8 => match plane {
                0..=2 => (w, h),
                _ => return Err(Error::invalid_frame("YUV444p has exactly three planes")),
            },
            Self::Nv12 => match plane {
                0 => (w, h),
                1 => (w.div_ceil(2) * 2, h.div_ceil(2)),
                _ => return Err(Error::invalid_frame("NV12 has exactly two planes")),
            },
            Self::Binary1Lsb | Self::Binary1Msb => match plane {
                0 => (w.div_ceil(8), h),
                _ => {
                    return Err(Error::invalid_frame(
                        "binary packed formats have exactly one plane",
                    ));
                }
            },
            _ => match plane {
                0 => (w * self.bytes_per_pixel().unwrap_or(1), h),
                _ => {
                    return Err(Error::invalid_frame(
                        "packed formats have exactly one plane",
                    ));
                }
            },
        };
        Dimensions::new(
            u32::try_from(pw).map_err(|_| Error::invalid_frame("plane width overflows u32"))?,
            u32::try_from(ph).map_err(|_| Error::invalid_frame("plane height overflows u32"))?,
        )
    }

    /// Minimum useful bytes in one row for the requested plane.
    pub fn plane_row_bytes(self, dims: Dimensions, plane: usize) -> Result<usize> {
        if plane >= self.plane_count() {
            return Err(Error::invalid_frame(format!(
                "plane {plane} does not exist for {self:?}"
            )));
        }
        if self.is_packed() {
            let width = usize::try_from(dims.width)
                .map_err(|_| Error::invalid_frame("width overflows usize"))?;
            width
                .checked_mul(self.bytes_per_pixel().expect("packed format has bpp"))
                .ok_or_else(|| Error::invalid_frame("row byte count overflows usize"))
        } else if self.is_bit_packed() {
            usize::try_from(dims.width)
                .map_err(|_| Error::invalid_frame("width overflows usize"))
                .map(|width| width.div_ceil(8))
        } else {
            usize::try_from(self.plane_dimensions(dims, plane)?.width)
                .map_err(|_| Error::invalid_frame("plane row bytes overflow usize"))
        }
    }

    /// Minimum valid byte length for a plane using the supplied row stride.
    pub fn plane_buffer_len(self, dims: Dimensions, plane: usize, stride: usize) -> Result<usize> {
        let row_bytes = self.plane_row_bytes(dims, plane)?;
        if stride < row_bytes {
            return Err(Error::invalid_frame(format!(
                "plane {plane} stride {stride} is smaller than required row bytes {row_bytes}"
            )));
        }
        let height = if self.is_packed() || self.is_bit_packed() {
            usize::try_from(dims.height)
                .map_err(|_| Error::invalid_frame("height overflows usize"))?
        } else {
            usize::try_from(self.plane_dimensions(dims, plane)?.height)
                .map_err(|_| Error::invalid_frame("plane height overflows usize"))?
        };
        stride
            .checked_mul(height.saturating_sub(1))
            .and_then(|length| length.checked_add(row_bytes))
            .ok_or_else(|| Error::invalid_frame("plane buffer length overflows usize"))
    }
}

/// Pixel-format metadata attached to a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FormatSpec {
    /// Pixel layout.
    pub pixel_format: PixelFormat,
    /// Color-space metadata.
    pub color_space: ColorSpace,
    /// Transfer-function metadata.
    pub transfer: Transfer,
    /// Full or limited range.
    pub range: ColorRange,
}

impl FormatSpec {
    /// Creates a full-range format spec using the pixel format's default color space.
    pub fn new(pixel_format: PixelFormat) -> Self {
        Self {
            pixel_format,
            color_space: pixel_format.default_color_space(),
            transfer: if pixel_format.is_yuv() {
                Transfer::Bt1886
            } else {
                Transfer::Srgb
            },
            range: ColorRange::Full,
        }
    }

    /// Creates a format spec with explicit metadata.
    pub fn with_metadata(
        pixel_format: PixelFormat,
        color_space: ColorSpace,
        transfer: Transfer,
        range: ColorRange,
    ) -> Self {
        Self {
            pixel_format,
            color_space,
            transfer,
            range,
        }
    }

    /// Returns the same format with a different range marker.
    pub fn typed_range<R: RangeMarker>(mut self) -> Self {
        self.range = R::VALUE;
        self
    }
}

/// Marker: the frame has not been validated.
#[derive(Debug, Clone, Copy)]
pub enum Unchecked {}

/// Marker: dimensions, plane count, strides, and buffer lengths were validated.
#[derive(Debug, Clone, Copy)]
pub enum Validated {}

/// A borrowed plane view.
#[derive(Debug, Clone, Copy)]
pub struct PlaneView<'a> {
    /// Plane bytes.
    pub data: &'a [u8],
    /// Bytes between two consecutive rows in this plane.
    pub stride: usize,
}

impl<'a> PlaneView<'a> {
    /// Creates a plane view.
    pub fn new(data: &'a [u8], stride: usize) -> Self {
        Self { data, stride }
    }

    /// Returns a row slice with the requested useful byte length.
    pub fn row(&self, y: usize, row_bytes: usize) -> Result<&'a [u8]> {
        let start = y
            .checked_mul(self.stride)
            .ok_or_else(|| Error::invalid_frame("row offset overflows usize"))?;
        let end = start
            .checked_add(row_bytes)
            .ok_or_else(|| Error::invalid_frame("row end overflows usize"))?;
        self.data
            .get(start..end)
            .ok_or_else(|| Error::invalid_frame("plane row is out of bounds"))
    }
}

/// Owned plane buffer.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OwnedPlane {
    /// Plane bytes.
    pub data: Vec<u8>,
    /// Bytes between rows.
    pub stride: usize,
}

impl OwnedPlane {
    /// Creates an owned plane.
    pub fn new(data: Vec<u8>, stride: usize) -> Self {
        Self { data, stride }
    }

    /// Borrows this plane.
    pub fn as_view(&self) -> PlaneView<'_> {
        PlaneView::new(&self.data, self.stride)
    }
}

/// A zero-copy borrowed frame view.
#[derive(Debug, Clone)]
pub struct FrameView<'a, State = Unchecked> {
    dims: Dimensions,
    format: FormatSpec,
    planes: Vec<PlaneView<'a>>,
    _state: PhantomData<State>,
}

impl<'a> FrameView<'a, Unchecked> {
    /// Creates an unchecked borrowed frame from explicit planes.
    pub fn from_planes(dims: Dimensions, format: FormatSpec, planes: Vec<PlaneView<'a>>) -> Self {
        Self {
            dims,
            format,
            planes,
            _state: PhantomData,
        }
    }

    /// Creates an unchecked packed single-plane frame.
    pub fn packed(
        data: &'a [u8],
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        stride: usize,
    ) -> Result<Self> {
        if !pixel_format.is_packed() {
            return Err(Error::invalid_frame(
                "FrameView::packed requires a packed pixel format",
            ));
        }
        Ok(Self::from_planes(
            Dimensions::new(width, height)?,
            FormatSpec::new(pixel_format),
            vec![PlaneView::new(data, stride)],
        ))
    }

    /// Creates an unchecked planar YUV frame from three planes.
    pub fn yuv_planar(
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        y: PlaneView<'a>,
        u: PlaneView<'a>,
        v: PlaneView<'a>,
    ) -> Result<Self> {
        if !matches!(
            pixel_format,
            PixelFormat::Yuv444p8 | PixelFormat::Yuv422p8 | PixelFormat::Yuv420p8
        ) {
            return Err(Error::invalid_frame(
                "yuv_planar requires Yuv444p8/Yuv422p8/Yuv420p8",
            ));
        }
        Ok(Self::from_planes(
            Dimensions::new(width, height)?,
            FormatSpec::new(pixel_format),
            vec![y, u, v],
        ))
    }

    /// Creates an unchecked NV12 frame from Y and interleaved UV planes.
    pub fn nv12(width: u32, height: u32, y: PlaneView<'a>, uv: PlaneView<'a>) -> Result<Self> {
        Ok(Self::from_planes(
            Dimensions::new(width, height)?,
            FormatSpec::new(PixelFormat::Nv12),
            vec![y, uv],
        ))
    }

    /// Validates plane count, stride, and accessible row byte ranges.
    pub fn validate(self) -> Result<FrameView<'a, Validated>> {
        validate_parts(self.dims, self.format, &self.planes)?;
        Ok(FrameView {
            dims: self.dims,
            format: self.format,
            planes: self.planes,
            _state: PhantomData,
        })
    }
}

impl<'a, S> FrameView<'a, S> {
    /// Frame dimensions.
    pub fn dimensions(&self) -> Dimensions {
        self.dims
    }

    /// Format spec.
    pub fn format(&self) -> FormatSpec {
        self.format
    }

    /// Pixel format.
    pub fn pixel_format(&self) -> PixelFormat {
        self.format.pixel_format
    }

    /// Borrowed plane list.
    pub fn planes(&self) -> &[PlaneView<'a>] {
        &self.planes
    }

    /// Returns one plane.
    pub fn plane(&self, index: usize) -> Result<PlaneView<'a>> {
        self.planes
            .get(index)
            .copied()
            .ok_or_else(|| Error::invalid_frame(format!("plane {index} does not exist")))
    }
}

impl<'a> FrameView<'a, Validated> {
    /// Re-borrows this validated view.
    pub fn as_ref(&self) -> FrameView<'_, Validated> {
        FrameView {
            dims: self.dims,
            format: self.format,
            planes: self.planes.clone(),
            _state: PhantomData,
        }
    }

    /// Returns `true` if the frame is RGBA8 packed with a tight or strided plane.
    pub fn is_rgba8(&self) -> bool {
        self.format.pixel_format == PixelFormat::Rgba8
    }
}

/// Owned frame storage.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameOwned {
    dims: Dimensions,
    format: FormatSpec,
    planes: Vec<OwnedPlane>,
}

impl FrameOwned {
    /// Creates an owned frame after validating its planes.
    pub fn new(dims: Dimensions, format: FormatSpec, planes: Vec<OwnedPlane>) -> Result<Self> {
        let views: Vec<_> = planes.iter().map(OwnedPlane::as_view).collect();
        validate_parts(dims, format, &views)?;
        Ok(Self {
            dims,
            format,
            planes,
        })
    }

    /// Creates an owned packed frame.
    pub fn packed(
        data: Vec<u8>,
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        stride: usize,
    ) -> Result<Self> {
        if !pixel_format.is_packed() {
            return Err(Error::invalid_frame(
                "FrameOwned::packed requires a packed pixel format",
            ));
        }
        Self::new(
            Dimensions::new(width, height)?,
            FormatSpec::new(pixel_format),
            vec![OwnedPlane::new(data, stride)],
        )
    }

    /// Creates an owned tightly packed frame.
    pub fn packed_tight(
        data: Vec<u8>,
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
    ) -> Result<Self> {
        let bpp = pixel_format
            .bytes_per_pixel()
            .ok_or_else(|| Error::invalid_frame("packed_tight requires a packed pixel format"))?;
        let stride = usize::try_from(width)
            .map_err(|_| Error::invalid_frame("width overflows usize"))?
            .checked_mul(bpp)
            .ok_or_else(|| Error::invalid_frame("stride overflows usize"))?;
        Self::packed(data, width, height, pixel_format, stride)
    }

    /// Frame dimensions.
    pub fn dimensions(&self) -> Dimensions {
        self.dims
    }

    /// Frame format.
    pub fn format(&self) -> FormatSpec {
        self.format
    }

    /// Owned plane list.
    pub fn owned_planes(&self) -> &[OwnedPlane] {
        &self.planes
    }

    /// Borrows as a validated zero-copy frame view.
    pub fn as_view(&self) -> FrameView<'_, Validated> {
        FrameView {
            dims: self.dims,
            format: self.format,
            planes: self.planes.iter().map(OwnedPlane::as_view).collect(),
            _state: PhantomData,
        }
    }

    /// Consumes the frame and returns its raw pieces.
    pub fn into_parts(self) -> (Dimensions, FormatSpec, Vec<OwnedPlane>) {
        (self.dims, self.format, self.planes)
    }

    /// Returns a tightly packed copy with per-row padding removed.
    ///
    /// The frame is cloned unchanged when every plane is already tight.
    pub fn to_tightly_packed(&self) -> Result<Self> {
        let planes =
            self.planes
                .iter()
                .enumerate()
                .map(|(index, plane)| {
                    let row_bytes = self.format.pixel_format.plane_row_bytes(self.dims, index)?;
                    let height = if self.format.pixel_format.is_packed()
                        || self.format.pixel_format.is_bit_packed()
                    {
                        usize::try_from(self.dims.height)
                            .map_err(|_| Error::invalid_frame("height overflows usize"))?
                    } else {
                        usize::try_from(
                            self.format
                                .pixel_format
                                .plane_dimensions(self.dims, index)?
                                .height,
                        )
                        .map_err(|_| Error::invalid_frame("plane height overflows usize"))?
                    };
                    let mut data =
                        Vec::with_capacity(row_bytes.checked_mul(height).ok_or_else(|| {
                            Error::invalid_frame("tight plane size overflows usize")
                        })?);
                    let view = plane.as_view();
                    for y in 0..height {
                        data.extend_from_slice(view.row(y, row_bytes)?);
                    }
                    Ok(OwnedPlane::new(data, row_bytes))
                })
                .collect::<Result<Vec<_>>>()?;
        Self::new(self.dims, self.format, planes)
    }
}

fn validate_parts(dims: Dimensions, format: FormatSpec, planes: &[PlaneView<'_>]) -> Result<()> {
    if planes.len() != format.pixel_format.plane_count() {
        return Err(Error::invalid_frame(format!(
            "expected {} plane(s), got {}",
            format.pixel_format.plane_count(),
            planes.len()
        )));
    }

    for (i, plane) in planes.iter().copied().enumerate() {
        let plane_dims = format.pixel_format.plane_dimensions(dims, i)?;
        let logical_row_bytes = format.pixel_format.plane_row_bytes(dims, i)?;
        let plane_h = if format.pixel_format.is_packed() || format.pixel_format.is_bit_packed() {
            dims.as_usize()?.1
        } else {
            plane_dims.as_usize()?.1
        };

        if plane.stride < logical_row_bytes {
            return Err(Error::invalid_frame(format!(
                "plane {i} stride {} is smaller than required row bytes {logical_row_bytes}",
                plane.stride
            )));
        }
        if plane_h == 0 {
            return Err(Error::invalid_frame(format!("plane {i} has zero height")));
        }
        let min_len = format
            .pixel_format
            .plane_buffer_len(dims, i, plane.stride)?;
        if plane.data.len() < min_len {
            return Err(Error::invalid_frame(format!(
                "plane {i} data length {} is smaller than required {min_len}",
                plane.data.len()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_tight_rgba() {
        let data = vec![0u8; 4 * 2 * 3];
        let frame = FrameOwned::packed_tight(data, 2, 3, PixelFormat::Rgba8).unwrap();
        assert_eq!(frame.dimensions().pixels().unwrap(), 6);
    }

    #[test]
    fn rejects_short_plane() {
        let err = FrameOwned::packed_tight(vec![0u8; 3], 2, 1, PixelFormat::Rgba8).unwrap_err();
        assert!(matches!(err, Error::InvalidFrame(_)));
    }

    #[test]
    fn computes_minimum_plane_layouts() {
        let dims = Dimensions::new(5, 3).unwrap();
        assert_eq!(PixelFormat::Rgba8.plane_row_bytes(dims, 0).unwrap(), 20);
        assert_eq!(PixelFormat::Yuv420p8.plane_row_bytes(dims, 1).unwrap(), 3);
        assert_eq!(PixelFormat::Nv12.plane_row_bytes(dims, 1).unwrap(), 6);
        assert_eq!(PixelFormat::Binary1Lsb.plane_row_bytes(dims, 0).unwrap(), 1);
        assert_eq!(
            PixelFormat::Rgba8.plane_buffer_len(dims, 0, 24).unwrap(),
            68
        );
    }

    #[test]
    fn strips_row_padding_without_changing_pixels() {
        let frame = FrameOwned::packed(
            vec![1, 2, 3, 4, 99, 99, 5, 6, 7, 8],
            1,
            2,
            PixelFormat::Rgba8,
            6,
        )
        .unwrap();
        let tight = frame.to_tightly_packed().unwrap();
        assert_eq!(tight.owned_planes()[0].stride, 4);
        assert_eq!(tight.owned_planes()[0].data, [1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
