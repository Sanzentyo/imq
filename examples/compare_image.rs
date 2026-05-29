use imq::Result;
use imq::adapters::image_crate;
use imq::metrics::MetricSet;
use imq::report::ComparisonReport;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    let reference_path = args
        .next()
        .expect("usage: compare_image <reference> <distorted>");
    let distorted_path = args
        .next()
        .expect("usage: compare_image <reference> <distorted>");

    let reference = image_crate::load_image_path(&reference_path)?;
    let distorted = image_crate::load_image_path(&distorted_path)?;
    let metrics = MetricSet::defaults();
    let outputs = metrics.compare(&reference.as_view(), &distorted.as_view())?;
    let report = ComparisonReport::new(
        reference.dimensions(),
        reference.format(),
        distorted.format(),
        outputs,
    )
    .with_labels(
        reference_path.display().to_string(),
        distorted_path.display().to_string(),
    );

    #[cfg(feature = "serde")]
    println!("{}", report.to_json_pretty()?);

    #[cfg(not(feature = "serde"))]
    for metric in report.metrics {
        println!("{} = {} {}", metric.name, metric.score, metric.unit);
    }

    Ok(())
}
