//! Video support through external `ffmpeg`/`ffprobe` rawvideo pipes.

pub mod ffmpeg;

pub use ffmpeg::{
    FfmpegFrameIter, FfmpegOptions, VideoCompareOptions, VideoInfo, compare_videos,
    decode_single_frame, probe_video,
};
