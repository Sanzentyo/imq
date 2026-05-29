//! Burn tensor adapters and feature-distance helpers.

use crate::frame::{ColorSpace, FrameView, PixelFormat, Validated};
use crate::metrics::{Direction, MetricOutput};
use crate::{Error, Result};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use std::collections::BTreeMap;
use std::marker::PhantomData;

/// Per-channel RGB normalization applied before creating a tensor.
///
/// Values are interpreted in normalized 0..1 RGB space. Use
/// [`NormalizeRgb::imagenet`] for VGG/LPIPS-style feature extractors.
#[derive(Debug, Clone, Copy)]
pub struct NormalizeRgb {
    /// Channel means in RGB order.
    pub mean: [f32; 3],
    /// Channel standard deviations in RGB order.
    pub std: [f32; 3],
}

impl NormalizeRgb {
    /// No normalization; leaves values in 0..1.
    pub const fn identity() -> Self {
        Self {
            mean: [0.0; 3],
            std: [1.0; 3],
        }
    }

    /// ImageNet-style normalization commonly used by VGG/LPIPS backbones.
    pub const fn imagenet() -> Self {
        Self {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    fn apply(self, rgb: [f32; 3]) -> [f32; 3] {
        [
            (rgb[0] - self.mean[0]) / self.std[0],
            (rgb[1] - self.mean[1]) / self.std[1],
            (rgb[2] - self.mean[2]) / self.std[2],
        ]
    }
}

impl Default for NormalizeRgb {
    fn default() -> Self {
        Self::identity()
    }
}

/// Converts an `imq` frame into an NCHW `[1, 3, H, W]` Burn tensor.
///
/// The conversion accepts common packed RGB/RGBA/BGR/BGRA/luma formats and
/// 8-bit YUV/NV12 formats. It performs exactly one CPU copy into a contiguous
/// `Vec<f32>`, which is the representation Burn needs for tensor creation.
pub fn frame_to_nchw_rgb_tensor<B: Backend>(
    frame: &FrameView<'_, Validated>,
    device: &B::Device,
    normalize: NormalizeRgb,
) -> Result<Tensor<B, 4>> {
    let (w, h) = frame.dimensions().as_usize()?;
    let plane_len = w
        .checked_mul(h)
        .ok_or_else(|| Error::invalid_frame("tensor plane size overflow"))?;
    let total = plane_len
        .checked_mul(3)
        .ok_or_else(|| Error::invalid_frame("tensor size overflow"))?;
    let mut data = vec![0.0f32; total];

    for y in 0..h {
        for x in 0..w {
            let rgb = normalize.apply(read_rgb01(frame, x, y)?);
            let idx = y * w + x;
            data[idx] = rgb[0];
            data[plane_len + idx] = rgb[1];
            data[plane_len * 2 + idx] = rgb[2];
        }
    }

    Ok(Tensor::<B, 4>::from_data(
        TensorData::new(data, [1, 3, h, w]),
        device,
    ))
}

/// Reads a scalar Burn tensor as `f64`.
///
/// Burn backends normally use `f32` float tensors, but this accepts `f64` too.
pub fn scalar_tensor_to_f64<B: Backend>(tensor: Tensor<B, 1>) -> Result<f64> {
    let data = tensor.into_data();
    if let Ok(values) = data.to_vec::<f32>() {
        return values
            .first()
            .map(|v| f64::from(*v))
            .ok_or_else(|| Error::Neural("scalar tensor is empty".to_string()));
    }
    if let Ok(values) = data.to_vec::<f64>() {
        return values
            .first()
            .copied()
            .ok_or_else(|| Error::Neural("scalar tensor is empty".to_string()));
    }
    Err(Error::Neural(
        "scalar tensor has an unsupported float element type".to_string(),
    ))
}

/// A simple Burn-based L2/MSE metric over normalized RGB tensors.
pub struct BurnL2Metric<B: Backend> {
    device: B::Device,
    normalize: NormalizeRgb,
    _backend: PhantomData<B>,
}

impl<B: Backend> BurnL2Metric<B> {
    /// Creates the metric for a Burn backend/device.
    pub fn new(device: B::Device) -> Self {
        Self {
            device,
            normalize: NormalizeRgb::identity(),
            _backend: PhantomData,
        }
    }

    /// Sets RGB normalization.
    pub fn with_normalize(mut self, normalize: NormalizeRgb) -> Self {
        self.normalize = normalize;
        self
    }

    /// Compares frames by tensor-space MSE.
    pub fn compare(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<MetricOutput> {
        if reference.dimensions() != distorted.dimensions() {
            return Err(Error::incompatible(format!(
                "dimension mismatch: {:?} vs {:?}",
                reference.dimensions(),
                distorted.dimensions()
            )));
        }
        let a = frame_to_nchw_rgb_tensor::<B>(reference, &self.device, self.normalize)?;
        let b = frame_to_nchw_rgb_tensor::<B>(distorted, &self.device, self.normalize)?;
        let diff = a - b;
        let mse = scalar_tensor_to_f64::<B>((diff.clone() * diff).mean())?;
        let mut details = BTreeMap::new();
        details.insert(
            "height".to_string(),
            f64::from(reference.dimensions().height),
        );
        details.insert("width".to_string(), f64::from(reference.dimensions().width));
        Ok(MetricOutput {
            name: "burn_l2_rgb".to_string(),
            score: mse,
            direction: Direction::LowerIsBetter,
            unit: "normalized_tensor_mse".to_string(),
            details,
        })
    }
}

/// Feature extractor abstraction used by LPIPS/DISTS-like metrics.
///
/// Implement this trait for a Burn module wrapping a pretrained network. The
/// extractor should return corresponding feature maps for each input image.
pub trait BurnFeatureExtractor<B: Backend> {
    /// Extracts an ordered set of feature tensors from a `[1, 3, H, W]` input.
    fn extract(&self, image: Tensor<B, 4>) -> Vec<Tensor<B, 4>>;
}

/// LPIPS/DISTS-style feature-distance metric.
///
/// This type computes the mean squared distance of corresponding feature maps.
/// A production LPIPS implementation can add learned per-channel weights inside
/// the extractor or by wrapping this type.
pub struct LpipsLikeMetric<B: Backend, E> {
    device: B::Device,
    extractor: E,
    normalize: NormalizeRgb,
    _backend: PhantomData<B>,
}

impl<B: Backend, E> LpipsLikeMetric<B, E>
where
    E: BurnFeatureExtractor<B>,
{
    /// Creates a feature-distance metric with ImageNet normalization.
    pub fn new(device: B::Device, extractor: E) -> Self {
        Self {
            device,
            extractor,
            normalize: NormalizeRgb::imagenet(),
            _backend: PhantomData,
        }
    }

    /// Overrides the normalization used before feeding the extractor.
    pub fn with_normalize(mut self, normalize: NormalizeRgb) -> Self {
        self.normalize = normalize;
        self
    }

    /// Compares frames with the feature extractor.
    pub fn compare(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<MetricOutput> {
        if reference.dimensions() != distorted.dimensions() {
            return Err(Error::incompatible(format!(
                "dimension mismatch: {:?} vs {:?}",
                reference.dimensions(),
                distorted.dimensions()
            )));
        }
        let a = frame_to_nchw_rgb_tensor::<B>(reference, &self.device, self.normalize)?;
        let b = frame_to_nchw_rgb_tensor::<B>(distorted, &self.device, self.normalize)?;
        let fa = self.extractor.extract(a);
        let fb = self.extractor.extract(b);
        if fa.len() != fb.len() {
            return Err(Error::Neural(format!(
                "feature extractor returned different feature counts: {} vs {}",
                fa.len(),
                fb.len()
            )));
        }

        let mut score = 0.0;
        let mut details = BTreeMap::new();
        for (index, (ra, rb)) in fa.into_iter().zip(fb).enumerate() {
            if ra.dims() != rb.dims() {
                return Err(Error::Neural(format!(
                    "feature map {index} shape mismatch: {:?} vs {:?}",
                    ra.dims(),
                    rb.dims()
                )));
            }
            let diff = ra - rb;
            let layer = scalar_tensor_to_f64::<B>((diff.clone() * diff).mean())?;
            details.insert(format!("layer_{index}"), layer);
            score += layer;
        }
        Ok(MetricOutput {
            name: "burn_lpips_like".to_string(),
            score,
            direction: Direction::LowerIsBetter,
            unit: "feature_distance".to_string(),
            details,
        })
    }
}

fn read_rgb01(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<[f32; 3]> {
    match frame.pixel_format() {
        PixelFormat::Luma8 => {
            let l = packed_u8(frame, x, y, 0, 1)?;
            Ok([l, l, l])
        }
        PixelFormat::Rgb8 => Ok([
            packed_u8(frame, x, y, 0, 3)?,
            packed_u8(frame, x, y, 1, 3)?,
            packed_u8(frame, x, y, 2, 3)?,
        ]),
        PixelFormat::Rgba8 => Ok([
            packed_u8(frame, x, y, 0, 4)?,
            packed_u8(frame, x, y, 1, 4)?,
            packed_u8(frame, x, y, 2, 4)?,
        ]),
        PixelFormat::Bgr8 => Ok([
            packed_u8(frame, x, y, 2, 3)?,
            packed_u8(frame, x, y, 1, 3)?,
            packed_u8(frame, x, y, 0, 3)?,
        ]),
        PixelFormat::Bgra8 => Ok([
            packed_u8(frame, x, y, 2, 4)?,
            packed_u8(frame, x, y, 1, 4)?,
            packed_u8(frame, x, y, 0, 4)?,
        ]),
        PixelFormat::Yuv444p8
        | PixelFormat::Yuv422p8
        | PixelFormat::Yuv420p8
        | PixelFormat::Nv12 => Ok(yuv_to_rgb(
            read_yuv01(frame, x, y)?,
            frame.format().color_space,
        )),
        other => Err(Error::unsupported(format!(
            "Burn tensor adapter does not yet support {other:?}"
        ))),
    }
}

fn packed_u8(
    frame: &FrameView<'_, Validated>,
    x: usize,
    y: usize,
    channel: usize,
    bpp: usize,
) -> Result<f32> {
    let plane = frame.plane(0)?;
    let (w, _) = frame.dimensions().as_usize()?;
    let row_bytes = w
        .checked_mul(bpp)
        .ok_or_else(|| Error::invalid_frame("row byte size overflow"))?;
    let row = plane.row(y, row_bytes)?;
    let off = x
        .checked_mul(bpp)
        .and_then(|v| v.checked_add(channel))
        .ok_or_else(|| Error::invalid_frame("pixel offset overflow"))?;
    row.get(off)
        .map(|v| f32::from(*v) / 255.0)
        .ok_or_else(|| Error::invalid_frame("pixel channel out of bounds"))
}

fn read_yuv01(frame: &FrameView<'_, Validated>, x: usize, y: usize) -> Result<[f32; 3]> {
    let y_plane = frame.plane(0)?;
    let (w, _) = frame.dimensions().as_usize()?;
    let y_row = y_plane.row(y, w)?;
    let yv = f32::from(y_row[x]) / 255.0;
    let chroma = match frame.pixel_format() {
        PixelFormat::Yuv444p8 => (x, y),
        PixelFormat::Yuv422p8 => (x / 2, y),
        PixelFormat::Yuv420p8 | PixelFormat::Nv12 => (x / 2, y / 2),
        _ => return Err(Error::unsupported("read_yuv01 requires a YUV format")),
    };
    match frame.pixel_format() {
        PixelFormat::Yuv444p8 | PixelFormat::Yuv422p8 | PixelFormat::Yuv420p8 => {
            let u_plane = frame.plane(1)?;
            let v_plane = frame.plane(2)?;
            let row_bytes = frame
                .pixel_format()
                .plane_dimensions(frame.dimensions(), 1)?
                .as_usize()?
                .0;
            Ok([
                yv,
                f32::from(u_plane.row(chroma.1, row_bytes)?[chroma.0]) / 255.0,
                f32::from(v_plane.row(chroma.1, row_bytes)?[chroma.0]) / 255.0,
            ])
        }
        PixelFormat::Nv12 => {
            let uv_plane = frame.plane(1)?;
            let row_bytes = frame
                .pixel_format()
                .plane_dimensions(frame.dimensions(), 1)?
                .as_usize()?
                .0;
            let row = uv_plane.row(chroma.1, row_bytes)?;
            let off = chroma.0 * 2;
            Ok([
                yv,
                f32::from(row[off]) / 255.0,
                f32::from(row[off + 1]) / 255.0,
            ])
        }
        _ => unreachable!(),
    }
}

fn yuv_to_rgb(yuv: [f32; 3], cs: ColorSpace) -> [f32; 3] {
    let y = yuv[0];
    let u = yuv[1] - 0.5;
    let v = yuv[2] - 0.5;
    let (rv, gu, gv, bu) = match cs {
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
