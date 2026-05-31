//! Error types used by `imq`.

use std::io;
use thiserror::Error;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced by the core, adapter, CLI, ffmpeg, GPU, and NN layers.
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid dimensions, strides, planes, or incompatible pixel formats.
    #[error("invalid frame: {0}")]
    InvalidFrame(String),

    /// Two frames cannot be compared by the requested metric.
    #[error("incompatible frames: {0}")]
    IncompatibleFrames(String),

    /// A requested metric name is not registered.
    #[error("unknown metric `{0}`")]
    UnknownMetric(String),

    /// The requested metric cannot be computed for this format/domain yet.
    #[error("unsupported metric input: {0}")]
    UnsupportedInput(String),

    /// Input/output error in optional I/O layers.
    #[error(transparent)]
    Io(#[from] io::Error),

    /// Input path does not exist.
    #[error("input file not found: {path}")]
    InputNotFound {
        /// Missing input path.
        path: String,
    },

    /// Error returned by the `image` crate.
    #[cfg(feature = "image-codecs")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image-codecs")))]
    #[error(transparent)]
    Image(#[from] image::ImageError),

    /// JSON parsing/serialization error used by reports and ffprobe parsing.
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// An external command failed.
    #[error("process `{program}` failed with status {status}: {stderr}")]
    ProcessFailed {
        /// Program name/path.
        program: String,
        /// Exit status as text.
        status: String,
        /// Captured stderr, lossy UTF-8.
        stderr: String,
    },

    /// A required external command could not be found.
    #[error("external tool `{program}` not found; install it or pass `{option} PATH`")]
    ExternalToolNotFound {
        /// Program name/path.
        program: String,
        /// CLI option that can override the program path.
        option: String,
    },

    /// A required external command could not be started.
    #[error("failed to start external tool `{program}`: {source}")]
    ExternalToolStartFailed {
        /// Program name/path.
        program: String,
        /// Start failure.
        #[source]
        source: io::Error,
    },

    /// GPU initialization, validation, dispatch, or readback failed.
    #[error("gpu error: {0}")]
    Gpu(String),

    /// Burn/tensor/perceptual metric error.
    #[error("nn metric error: {0}")]
    Neural(String),
}

impl Error {
    /// Helper for frame validation failures.
    pub fn invalid_frame(msg: impl Into<String>) -> Self {
        Self::InvalidFrame(msg.into())
    }

    /// Helper for incompatible-frame failures.
    pub fn incompatible(msg: impl Into<String>) -> Self {
        Self::IncompatibleFrames(msg.into())
    }

    /// Helper for unsupported inputs.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::UnsupportedInput(msg.into())
    }

    /// Helper for missing input files.
    pub fn input_not_found(path: impl Into<String>) -> Self {
        Self::InputNotFound { path: path.into() }
    }

    /// Helper for external command startup failures.
    pub fn external_tool_start_failed(
        program: impl Into<String>,
        option: impl Into<String>,
        source: io::Error,
    ) -> Self {
        let program = program.into();
        if source.kind() == io::ErrorKind::NotFound {
            Self::ExternalToolNotFound {
                program,
                option: option.into(),
            }
        } else {
            Self::ExternalToolStartFailed { program, source }
        }
    }
}
