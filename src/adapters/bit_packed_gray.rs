//! Bit-packed grayscale and mask adapters.

use crate::frame::{FrameOwned, PixelFormat};
use crate::{Error, Result};

/// Bit order inside each packed byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOrder {
    /// Pixel 0 is stored in the least-significant bits.
    LsbFirst,
    /// Pixel 0 is stored in the most-significant bits.
    MsbFirst,
}

/// Supported bit-packed grayscale layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitPackedGrayFormat {
    /// Bits per pixel. Supported values are 1, 2, and 4.
    pub bits_per_pixel: u8,
    /// Bit order inside each byte.
    pub bit_order: BitOrder,
}

impl BitPackedGrayFormat {
    /// Creates a packed grayscale format.
    pub fn new(bits_per_pixel: u8, bit_order: BitOrder) -> Result<Self> {
        if !matches!(bits_per_pixel, 1 | 2 | 4) {
            return Err(Error::unsupported(
                "bit-packed gray supports only 1, 2, or 4 bits per pixel",
            ));
        }
        Ok(Self {
            bits_per_pixel,
            bit_order,
        })
    }
}

/// Decodes bit-packed grayscale rows into tightly packed `Luma8`.
///
/// `input_stride` is the byte distance between rows in the packed input. When
/// omitted, the adapter uses the tight packed row size for `width`.
pub fn decode_bit_packed_gray(
    data: &[u8],
    width: u32,
    height: u32,
    format: BitPackedGrayFormat,
    input_stride: Option<usize>,
) -> Result<FrameOwned> {
    let width =
        usize::try_from(width).map_err(|_| Error::invalid_frame("width overflows usize"))?;
    let height =
        usize::try_from(height).map_err(|_| Error::invalid_frame("height overflows usize"))?;
    let tight_stride = packed_row_bytes(width, format.bits_per_pixel)?;
    let input_stride = input_stride.unwrap_or(tight_stride);
    if input_stride < tight_stride {
        return Err(Error::invalid_frame(format!(
            "raw stride {input_stride} is smaller than tight packed row size {tight_stride}"
        )));
    }
    let required = input_stride
        .checked_mul(height.saturating_sub(1))
        .and_then(|offset| offset.checked_add(tight_stride))
        .ok_or_else(|| Error::invalid_frame("bit-packed input length overflows usize"))?;
    if data.len() < required {
        return Err(Error::invalid_frame(format!(
            "bit-packed input is too short: need {required} bytes, got {}",
            data.len()
        )));
    }

    let output_len = width
        .checked_mul(height)
        .ok_or_else(|| Error::invalid_frame("decoded luma length overflows usize"))?;
    let mut output = vec![0u8; output_len];
    for y in 0..height {
        let row_start = y
            .checked_mul(input_stride)
            .ok_or_else(|| Error::invalid_frame("row offset overflows usize"))?;
        let row = &data[row_start..row_start + tight_stride];
        let out_row = &mut output[y * width..(y + 1) * width];
        decode_row(row, out_row, format);
    }

    FrameOwned::packed_tight(
        output,
        u32::try_from(width).map_err(|_| Error::invalid_frame("width overflows u32"))?,
        u32::try_from(height).map_err(|_| Error::invalid_frame("height overflows u32"))?,
        PixelFormat::Luma8,
    )
}

fn packed_row_bytes(width: usize, bits_per_pixel: u8) -> Result<usize> {
    width
        .checked_mul(usize::from(bits_per_pixel))
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
        .ok_or_else(|| Error::invalid_frame("packed row size overflows usize"))
}

fn decode_row(row: &[u8], output: &mut [u8], format: BitPackedGrayFormat) {
    let mask = (1u8 << format.bits_per_pixel) - 1;
    let max = u16::from(mask);
    output.iter_mut().enumerate().for_each(|(x, out)| {
        let bit_index = x * usize::from(format.bits_per_pixel);
        let byte = row[bit_index / 8];
        let shift = match format.bit_order {
            BitOrder::LsbFirst => bit_index % 8,
            BitOrder::MsbFirst => 8 - usize::from(format.bits_per_pixel) - (bit_index % 8),
        };
        let value = (byte >> shift) & mask;
        *out = u8::try_from((u16::from(value) * 255) / max).expect("scaled luma fits u8");
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded_plane(frame: &FrameOwned) -> &[u8] {
        &frame.owned_planes()[0].data
    }

    #[test]
    fn decodes_gray1_lsb_bit_order() {
        let format = BitPackedGrayFormat::new(1, BitOrder::LsbFirst).unwrap();
        let frame = decode_bit_packed_gray(&[0b0000_1011], 4, 1, format, None).unwrap();
        assert_eq!(decoded_plane(&frame), [255, 255, 0, 255]);
    }

    #[test]
    fn decodes_gray1_msb_bit_order() {
        let format = BitPackedGrayFormat::new(1, BitOrder::MsbFirst).unwrap();
        let frame = decode_bit_packed_gray(&[0b1011_0000], 4, 1, format, None).unwrap();
        assert_eq!(decoded_plane(&frame), [255, 0, 255, 255]);
    }

    #[test]
    fn decodes_gray2_lsb_partial_byte() {
        let format = BitPackedGrayFormat::new(2, BitOrder::LsbFirst).unwrap();
        let frame = decode_bit_packed_gray(&[0b0011_1001], 3, 1, format, None).unwrap();
        assert_eq!(decoded_plane(&frame), [85, 170, 255]);
    }

    #[test]
    fn decodes_gray2_msb_partial_byte() {
        let format = BitPackedGrayFormat::new(2, BitOrder::MsbFirst).unwrap();
        let frame = decode_bit_packed_gray(&[0b0110_1100], 3, 1, format, None).unwrap();
        assert_eq!(decoded_plane(&frame), [85, 170, 255]);
    }

    #[test]
    fn decodes_gray4_lsb_with_row_stride() {
        let format = BitPackedGrayFormat::new(4, BitOrder::LsbFirst).unwrap();
        let frame =
            decode_bit_packed_gray(&[0x21, 0xff, 0x43, 0xee], 2, 2, format, Some(2)).unwrap();
        assert_eq!(decoded_plane(&frame), [17, 34, 51, 68]);
    }

    #[test]
    fn decodes_gray4_msb_with_row_stride() {
        let format = BitPackedGrayFormat::new(4, BitOrder::MsbFirst).unwrap();
        let frame =
            decode_bit_packed_gray(&[0x12, 0xff, 0x34, 0xee], 2, 2, format, Some(2)).unwrap();
        assert_eq!(decoded_plane(&frame), [17, 34, 51, 68]);
    }
}
