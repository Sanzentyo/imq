//! Video support through external `ffmpeg`/`ffprobe` rawvideo pipes.

pub mod ffmpeg;

pub use ffmpeg::{
    DecodedVideoFrame, FfmpegFrameIter, FfmpegOptions, FfmpegTimestampedFrameIter,
    TimestampVideoCompareOptions, TimestampVideoComparison, VideoCompareOptions, VideoFrameInfo,
    VideoInfo, VideoStreamControl, VideoStreamSummary, compare_videos, compare_videos_by_timestamp,
    compare_videos_streaming, decode_single_frame, probe_video, probe_video_frames,
};
