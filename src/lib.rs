//! `imq` is a Sans-I/O-first image and video quality evaluation crate.
//!
//! The core layer accepts borrowed frame views (`&[u8]`, `Vec<u8>` via
//! [`FrameOwned`], camera buffers, YUV planes, or `image` crate buffers) and
//! computes metrics without doing filesystem, terminal, process, or network I/O.
//!
//! Optional layers add:
//! - `image-codecs`: decoding through the `image` crate.
//! - `ffmpeg`: video/raw-frame piping through external `ffmpeg`/`ffprobe`.
//! - `gpu`: wgpu compute kernels for RGBA8-heavy workloads.
//! - `nn-burn`: Burn tensor adapters for NN/perceptual metrics.
//! - `preview`: terminal-friendly still/video thumbnails.
//! - `cli` / `tui`: command-line and terminal UI frontends.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

pub mod adapters;
pub mod error;
pub mod frame;
pub mod metrics;
pub mod report;
pub mod stats;

#[cfg(feature = "ffmpeg")]
#[cfg_attr(docsrs, doc(cfg(feature = "ffmpeg")))]
pub mod video;

#[cfg(feature = "gpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "gpu")))]
pub mod gpu;

#[cfg(feature = "nn-burn")]
#[cfg_attr(docsrs, doc(cfg(feature = "nn-burn")))]
pub mod nn;

#[cfg(feature = "preview")]
#[cfg_attr(docsrs, doc(cfg(feature = "preview")))]
pub mod preview;

pub use error::{Error, Result};
pub use frame::{
    ChromaSampling, ColorRange, ColorSpace, Dimensions, FormatSpec, FrameOwned, FrameView,
    FullRange, LimitedRange, OwnedPlane, PixelFormat, PlaneView, Transfer, Unchecked, Validated,
};
pub use metrics::{
    Direction, Metric, MetricOutput, MetricSet, MetricSpec, SampleDomain, compare_with_defaults,
};
pub use report::{ComparisonReport, FrameReport, VideoReport};
pub use stats::{ImageStatistics, ImageStatisticsOptions, image_statistics};
