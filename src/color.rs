//! Color-range and transfer-function helpers for quality metrics.
//!
//! The existing frame metadata already records color space, transfer, and range.
//! This module provides small, deterministic helpers that metric implementations
//! and applications can use when they need explicit SDR/HDR behavior instead of
//! treating normalized code values as display-linear samples.

use crate::frame::{ColorRange, ColorSpace, Transfer};

/// Options used when converting RGB or YUV code values into luma-like samples.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorTransformOptions {
    /// RGB/YUV matrix family.
    pub color_space: ColorSpace,
    /// Transfer function used by the stored code values.
    pub transfer: Transfer,
    /// Whether the stored code values use full or video-limited range.
    pub range: ColorRange,
    /// Decode RGB transfer functions before computing luma.
    pub linearize_rgb: bool,
}

impl ColorTransformOptions {
    /// Creates transform options from frame metadata.
    pub fn new(color_space: ColorSpace, transfer: Transfer, range: ColorRange) -> Self {
        Self {
            color_space,
            transfer,
            range,
            linearize_rgb: false,
        }
    }

    /// Enables or disables RGB transfer-function decoding before luma computation.
    pub fn with_linearize_rgb(mut self, linearize_rgb: bool) -> Self {
        self.linearize_rgb = linearize_rgb;
        self
    }
}

impl Default for ColorTransformOptions {
    fn default() -> Self {
        Self::new(ColorSpace::Srgb, Transfer::Srgb, ColorRange::Full)
    }
}

/// Returns RGB luma weights for the requested color-space family.
pub fn luma_weights(color_space: ColorSpace) -> (f64, f64, f64) {
    match color_space {
        ColorSpace::Bt601 => (0.299, 0.587, 0.114),
        ColorSpace::Bt2020 => (0.2627, 0.6780, 0.0593),
        _ => (0.2126, 0.7152, 0.0722),
    }
}

/// Expands a normalized luma code value according to full or limited range.
pub fn expand_luma_range(value: f64, range: ColorRange) -> f64 {
    match range {
        ColorRange::Full => value.clamp(0.0, 1.0),
        ColorRange::Limited => ((value * 255.0 - 16.0) / 219.0).clamp(0.0, 1.0),
    }
}

/// Expands a normalized chroma code value to the full 0..1 chroma-code range.
pub fn expand_chroma_range(value: f64, range: ColorRange) -> f64 {
    match range {
        ColorRange::Full => value.clamp(0.0, 1.0),
        ColorRange::Limited => ((value * 255.0 - 16.0) / 224.0).clamp(0.0, 1.0),
    }
}

/// Decodes a normalized code value according to a transfer function.
pub fn decode_transfer(value: f64, transfer: Transfer) -> f64 {
    let v = value.clamp(0.0, 1.0);
    match transfer {
        Transfer::Linear => v,
        Transfer::Srgb => {
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        }
        Transfer::Bt1886 => v.powf(2.4),
        Transfer::Hlg => decode_hlg(v),
        Transfer::Pq => decode_pq_st2084(v),
        Transfer::Unknown => v,
    }
}

/// Encodes a normalized linear-light value according to a transfer function.
pub fn encode_transfer(value: f64, transfer: Transfer) -> f64 {
    let v = value.clamp(0.0, 1.0);
    match transfer {
        Transfer::Linear => v,
        Transfer::Srgb => {
            if v <= 0.003_130_8 {
                12.92 * v
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            }
        }
        Transfer::Bt1886 => v.powf(1.0 / 2.4),
        Transfer::Hlg => encode_hlg(v),
        Transfer::Pq => encode_pq_st2084(v),
        Transfer::Unknown => v,
    }
}

/// Converts normalized RGB code values to a luma sample using explicit options.
pub fn rgb_to_luma(rgb: [f64; 3], options: ColorTransformOptions) -> f64 {
    let [mut r, mut g, mut b] = rgb;
    if options.range == ColorRange::Limited {
        r = expand_luma_range(r, options.range);
        g = expand_luma_range(g, options.range);
        b = expand_luma_range(b, options.range);
    }
    if options.linearize_rgb {
        r = decode_transfer(r, options.transfer);
        g = decode_transfer(g, options.transfer);
        b = decode_transfer(b, options.transfer);
    }
    let (kr, kg, kb) = luma_weights(options.color_space);
    (r * kr + g * kg + b * kb).clamp(0.0, 1.0)
}

/// Converts normalized YUV code values to RGB using the selected matrix and range.
pub fn yuv_to_rgb(yuv: [f64; 3], color_space: ColorSpace, range: ColorRange) -> [f64; 3] {
    let y = expand_luma_range(yuv[0], range);
    let u = expand_chroma_range(yuv[1], range) - 0.5;
    let v = expand_chroma_range(yuv[2], range) - 0.5;
    let (rv, gu, gv, bu) = match color_space {
        ColorSpace::Bt601 => (1.402, 0.344_136, 0.714_136, 1.772),
        ColorSpace::Bt2020 => (1.4746, 0.164_553, 0.571_353, 1.8814),
        _ => (1.5748, 0.187_324, 0.468_124, 1.8556),
    };
    [
        (y + rv * v).clamp(0.0, 1.0),
        (y - gu * u - gv * v).clamp(0.0, 1.0),
        (y + bu * u).clamp(0.0, 1.0),
    ]
}

fn decode_hlg(value: f64) -> f64 {
    let a: f64 = 0.178_832_77;
    let b: f64 = 0.284_668_92;
    let c: f64 = 0.559_910_73;
    if value <= 0.5 {
        (value * value) / 3.0
    } else {
        ((value - c).exp() + b) / 12.0 / a.exp()
    }
}

fn encode_hlg(value: f64) -> f64 {
    let a: f64 = 0.178_832_77;
    let b: f64 = 0.284_668_92;
    let c: f64 = 0.559_910_73;
    if value <= 1.0 / 12.0 {
        (3.0 * value).sqrt()
    } else {
        a * (12.0 * value - b).ln() + c
    }
}

fn decode_pq_st2084(value: f64) -> f64 {
    let m1 = 2610.0 / 16_384.0;
    let m2 = 2523.0 / 32.0;
    let c1 = 3424.0 / 4096.0;
    let c2 = 2413.0 / 128.0;
    let c3 = 2392.0 / 128.0;
    let v = value.powf(1.0 / m2);
    ((v - c1).max(0.0) / (c2 - c3 * v)).powf(1.0 / m1)
}

fn encode_pq_st2084(value: f64) -> f64 {
    let m1 = 2610.0 / 16_384.0;
    let m2 = 2523.0 / 32.0;
    let c1 = 3424.0 / 4096.0;
    let c2 = 2413.0 / 128.0;
    let c3 = 2392.0 / 128.0;
    let vp = value.powf(m1);
    ((c1 + c2 * vp) / (1.0 + c3 * vp)).powf(m2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_round_trip_is_close() {
        let linear = decode_transfer(0.5, Transfer::Srgb);
        let encoded = encode_transfer(linear, Transfer::Srgb);
        assert!((encoded - 0.5).abs() < 1e-12);
    }

    #[test]
    fn limited_luma_expands_black_and_white() {
        assert_eq!(expand_luma_range(16.0 / 255.0, ColorRange::Limited), 0.0);
        assert!((expand_luma_range(235.0 / 255.0, ColorRange::Limited) - 1.0).abs() < 1e-12);
    }
}
