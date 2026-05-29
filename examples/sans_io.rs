use imq::{FrameView, MetricSet, PixelFormat, Result};

fn main() -> Result<()> {
    let reference_rgba = [0, 0, 0, 255, 255, 255, 255, 255];
    let distorted_rgba = [0, 0, 0, 255, 250, 250, 250, 255];

    let reference = FrameView::packed(&reference_rgba, 2, 1, PixelFormat::Rgba8, 8)?.validate()?;
    let distorted = FrameView::packed(&distorted_rgba, 2, 1, PixelFormat::Rgba8, 8)?.validate()?;

    let metrics = MetricSet::from_csv("psnr:color,mse:color,mae:color")?;
    for metric in metrics.compare(&reference, &distorted)? {
        println!("{} = {} {}", metric.name, metric.score, metric.unit);
    }
    Ok(())
}
