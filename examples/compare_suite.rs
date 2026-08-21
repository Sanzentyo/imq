use imq::{
    BaselineThresholdRule, ComparisonCandidate, ComparisonSuiteOptions, FrameOwned, MetricSet,
    PixelFormat, compare_candidate_suite,
};

fn main() -> imq::Result<()> {
    let reference = luma(&[0, 64, 128, 255])?;
    let baseline = luma(&[0, 66, 128, 252])?;
    let candidate_a = luma(&[0, 65, 128, 254])?;
    let candidate_b = luma(&[8, 70, 140, 240])?;
    let candidates = [
        ComparisonCandidate::new("baseline", baseline.as_view()),
        ComparisonCandidate::new("candidate-a", candidate_a.as_view()),
        ComparisonCandidate::new("candidate-b", candidate_b.as_view()),
    ];
    let metrics = MetricSet::from_csv("psnr,ssim,wssim,mse,mae,maxae")?;
    let options = ComparisonSuiteOptions {
        primary_metric: Some("psnr".to_string()),
        baseline_candidate: Some("baseline".to_string()),
        baseline_rules: vec![BaselineThresholdRule::parse("psnr>=baseline-1")?],
        ..ComparisonSuiteOptions::default()
    };
    let report = compare_candidate_suite(
        Some("reference".to_string()),
        &reference.as_view(),
        &candidates,
        &metrics,
        &options,
    )?;

    println!("{}", report.to_json_pretty()?);
    Ok(())
}

fn luma(samples: &[u8]) -> imq::Result<FrameOwned> {
    FrameOwned::packed_tight(samples.to_vec(), 2, 2, PixelFormat::Luma8)
}
