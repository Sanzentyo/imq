//! Input adapters for raw memory, `image` crate images, and camera/video buffers.
//!
//! The constructors here do not compute metrics. They only translate external
//! buffer layouts into [`crate::FrameView`] or [`crate::FrameOwned`].

use crate::Result;
use crate::frame::{FrameView, PixelFormat, PlaneView, Unchecked, Validated};

#[cfg(feature = "image-codecs")]
#[cfg_attr(docsrs, doc(cfg(feature = "image-codecs")))]
pub mod image_crate;

/// Creates a validated borrowed RGB8 frame.
pub fn rgb8_view(
    data: &[u8],
    width: u32,
    height: u32,
    stride: usize,
) -> Result<FrameView<'_, Validated>> {
    FrameView::<Unchecked>::packed(data, width, height, PixelFormat::Rgb8, stride)?.validate()
}

/// Creates a validated borrowed RGBA8 frame.
pub fn rgba8_view(
    data: &[u8],
    width: u32,
    height: u32,
    stride: usize,
) -> Result<FrameView<'_, Validated>> {
    FrameView::<Unchecked>::packed(data, width, height, PixelFormat::Rgba8, stride)?.validate()
}

/// Creates a validated borrowed BGR8 frame.
pub fn bgr8_view(
    data: &[u8],
    width: u32,
    height: u32,
    stride: usize,
) -> Result<FrameView<'_, Validated>> {
    FrameView::<Unchecked>::packed(data, width, height, PixelFormat::Bgr8, stride)?.validate()
}

/// Creates a validated borrowed BGRA8 frame.
pub fn bgra8_view(
    data: &[u8],
    width: u32,
    height: u32,
    stride: usize,
) -> Result<FrameView<'_, Validated>> {
    FrameView::<Unchecked>::packed(data, width, height, PixelFormat::Bgra8, stride)?.validate()
}

/// Creates a validated borrowed YUV420p8 frame from separate Y/U/V planes.
#[allow(clippy::too_many_arguments)]
pub fn yuv420p8_view<'a>(
    width: u32,
    height: u32,
    y: &'a [u8],
    y_stride: usize,
    u: &'a [u8],
    u_stride: usize,
    v: &'a [u8],
    v_stride: usize,
) -> Result<FrameView<'a, Validated>> {
    FrameView::<Unchecked>::yuv_planar(
        width,
        height,
        PixelFormat::Yuv420p8,
        PlaneView::new(y, y_stride),
        PlaneView::new(u, u_stride),
        PlaneView::new(v, v_stride),
    )?
    .validate()
}

/// Creates a validated borrowed NV12 frame.
pub fn nv12_view<'a>(
    width: u32,
    height: u32,
    y: &'a [u8],
    y_stride: usize,
    uv: &'a [u8],
    uv_stride: usize,
) -> Result<FrameView<'a, Validated>> {
    FrameView::<Unchecked>::nv12(
        width,
        height,
        PlaneView::new(y, y_stride),
        PlaneView::new(uv, uv_stride),
    )?
    .validate()
}
