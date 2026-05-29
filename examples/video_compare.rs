use imq::Result;
use imq::metrics::MetricSet;
use imq::video::{FfmpegOptions, VideoCompareOptions, compare_videos};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    let reference = args
        .next()
        .expect("usage: video_compare <reference> <distorted>");
    let distorted = args
        .next()
        .expect("usage: video_compare <reference> <distorted>");

    let metrics = MetricSet::from_csv("psnr,ssim,mse")?;
    let options = VideoCompareOptions {
        every: 30,
        max_frames: Some(120),
    };
    let report = compare_videos(
        reference,
        distorted,
        &FfmpegOptions::default(),
        &options,
        &metrics,
    )?;

    #[cfg(feature = "serde")]
    println!("{}", report.to_json_pretty()?);

    #[cfg(not(feature = "serde"))]
    for metric in report.mean_metrics {
        println!("{} = {} {}", metric.name, metric.score, metric.unit);
    }

    Ok(())
}
