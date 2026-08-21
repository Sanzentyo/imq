//! Optional GPU acceleration layer.
//!
//! The public API stays separate from the Sans-I/O core: callers pass validated
//! borrowed frames, and the implementation uploads only the exact bytes required
//! for the selected kernel.

#[cfg(feature = "gpu")]
mod wgpu_impl;

#[cfg(feature = "gpu")]
pub use wgpu_impl::{
    GpuCapabilities, GpuComparator, GpuComparatorOptions, GpuContext, GpuContextOptions,
    GpuErrorDomain, GpuErrorMetric, GpuExecution, GpuFallbackPolicy, GpuMse, GpuPowerPreference,
    GpuRgba8Comparison, GpuRgba8ErrorStats, GpuWindowedSsimComparison, GpuWindowedSsimOptions,
};
