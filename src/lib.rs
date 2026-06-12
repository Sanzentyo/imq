//! `imq` is a Sans-I/O-first image and video quality evaluation crate.
//!
//! The core layer accepts borrowed frame views (`&[u8]`, `Vec<u8>` via
//! [`FrameOwned`], camera buffers, YUV planes, or `image` crate buffers) and
//! computes metrics without doing filesystem, terminal, process, or network I/O.
//!
//! Optional layers add:
//! - `image-codecs`: decoding through the `image` crate.
//! - `imqraw-image`: conversion helpers from common Rust image crate types.
//! - `ffmpeg`: video/raw-frame piping through external `ffmpeg`/`ffprobe`.
//! - `gpu`: wgpu compute kernels for RGBA8-heavy workloads.
//! - `nn-burn`: Burn tensor adapters for NN/perceptual metrics.
//! - `preview`: terminal-friendly still/video thumbnails.
//! - `cli` / `tui`: command-line and terminal UI frontends.
//! - diff/heatmap, CI gate, external metric, color, and video aggregation helpers.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

pub mod adapters;
pub mod color;
pub mod diff;
pub mod error;
pub mod external;
pub mod frame;
pub mod gate;
pub mod imqraw;
pub mod metrics;
pub mod reference_vectors;
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub mod remote;
pub mod report;
pub mod stats;
pub mod video_analysis;

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

pub use color::{
    ColorTransformOptions, decode_transfer, encode_transfer, expand_chroma_range,
    expand_luma_range, luma_weights, rgb_to_luma, yuv_to_rgb,
};
pub use diff::{DiffImageMode, DiffImageOptions, RgbaImageData, diff_image};
pub use error::{Error, Result};
pub use external::{ExternalMetricCommand, ExternalMetricParser, ExternalMetricRun};
pub use frame::{
    ChromaSampling, ColorRange, ColorSpace, Dimensions, FormatSpec, FrameOwned, FrameView,
    FullRange, LimitedRange, OwnedPlane, PixelFormat, PlaneView, Transfer, Unchecked, Validated,
};
pub use gate::{
    GateEvaluation, GateOperator, MetricThresholdCheck, MetricThresholdRule, evaluate_thresholds,
};
pub use imqraw::{
    RawImageBundle, RawImageRecord, RawImageSelector, decode_bundle as decode_imqraw_bundle,
    encode_bundle as encode_imqraw_bundle,
};
pub use metrics::{
    AlphaBucketCounts, AlphaDiagnostics, Direction, Metric, MetricOutput, MetricSet, MetricSpec,
    MsSsim, RenderChannels, RenderDomain, RenderMask, SampleDomain, Ssim, WindowedSsim,
    alpha_diagnostics, compare_with_defaults,
};
pub use reference_vectors::{
    ReferenceMetricExpectations, ReferenceMetricVector, assert_reference_vectors,
    reference_metric_vectors,
};
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub use remote::{
    InputSpec, RemoteFrameFormat, RemoteOptions, RemoteTransferMode, SshInput, VideoFramePair,
    parse_video_frame_pairs, shell_quote_posix,
};
pub use report::{
    ComparisonGateReport, ComparisonInput, ComparisonReport, ComparisonThresholds, FramePairReport,
    FrameReport, VideoFramePairComparisonReport, VideoReport,
};
pub use stats::{ImageStatistics, ImageStatisticsOptions, image_statistics};
pub use video_analysis::{
    MetricSeriesAggregate, PercentileValue, TimestampFramePair, TimestampPairingOptions,
    TimestampedFrame, VideoAggregationPolicy, aggregate_video_report, pair_frames_by_timestamp,
};
