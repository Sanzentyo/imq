//! Optional neural/perceptual metric layer backed by Burn.
//!
//! This module provides the hard part that should be shared by LPIPS, DISTS,
//! MUSIQ, CLIP-IQA-like, and custom research metrics: converting `imq` frames
//! into backend-generic Burn tensors and comparing feature tensors. Pretrained
//! model definitions/weights are intentionally caller-provided so applications
//! can choose licensing, backend, and checkpoint provenance explicitly.

#[cfg(feature = "nn-burn")]
mod burn_impl;

#[cfg(feature = "nn-burn")]
pub use burn_impl::{
    BurnFeatureExtractor, BurnL2Metric, LpipsLikeMetric, NormalizeRgb, frame_to_nchw_rgb_tensor,
    scalar_tensor_to_f64,
};
