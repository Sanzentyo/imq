use imq::{
    FrameOwned, MetricSet, PixelFormat, RawImageBundle, RawImageRecord, Result,
    decode_imqraw_bundle, encode_imqraw_bundle,
};

fn main() -> Result<()> {
    let reference = RawImageRecord::new(
        Some("reference".to_string()),
        vec!["ref".to_string()],
        FrameOwned::packed_tight(
            vec![0, 0, 0, 255, 255, 255, 255, 255],
            2,
            1,
            PixelFormat::Rgba8,
        )?,
    );
    let candidate = RawImageRecord::new(
        Some("candidate".to_string()),
        vec!["dist".to_string()],
        FrameOwned::packed_tight(
            vec![0, 0, 0, 255, 250, 250, 250, 255],
            2,
            1,
            PixelFormat::Rgba8,
        )?,
    );

    let bundle = RawImageBundle::new(vec![reference, candidate]);
    let bytes = encode_imqraw_bundle(&bundle)?;
    let decoded = decode_imqraw_bundle(&bytes)?;

    let reference = &decoded.select_tag("ref")?.frame;
    let candidate = &decoded.select_tag("dist")?.frame;
    let metrics = MetricSet::from_csv("psnr:color,mse:color,mae:color")?;

    for metric in metrics.compare(&reference.as_view(), &candidate.as_view())? {
        println!("{} = {} {}", metric.name, metric.score, metric.unit);
    }

    Ok(())
}
