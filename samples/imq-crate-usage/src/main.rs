use imq::{
    FrameOwned, MetricSet, PixelFormat, RawImageBundle, RawImageRecord, Result,
    decode_imqraw_bundle, encode_imqraw_bundle,
};

fn main() -> Result<()> {
    let reference = FrameOwned::packed_tight(
        vec![0, 0, 0, 255, 255, 255, 255, 255],
        2,
        1,
        PixelFormat::Rgba8,
    )?;
    let distorted = FrameOwned::packed_tight(
        vec![0, 0, 0, 255, 250, 250, 250, 255],
        2,
        1,
        PixelFormat::Rgba8,
    )?;

    let metrics = MetricSet::from_csv("psnr:color,mse:color,mae:color")?;
    for metric in metrics.compare(&reference.as_view(), &distorted.as_view())? {
        println!("{} = {} {}", metric.name, metric.score, metric.unit);
    }

    let bundle = RawImageBundle::new(vec![
        RawImageRecord::new(
            Some("reference".to_string()),
            vec!["ref".to_string()],
            reference,
        ),
        RawImageRecord::new(
            Some("distorted".to_string()),
            vec!["dist".to_string()],
            distorted,
        ),
    ]);
    let bytes = encode_imqraw_bundle(&bundle)?;
    let decoded = decode_imqraw_bundle(&bytes)?;
    println!(
        "imqraw bytes={} images={}",
        bytes.len(),
        decoded.records.len()
    );

    Ok(())
}
