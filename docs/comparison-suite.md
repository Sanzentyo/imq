# Multi-candidate comparison suites

Pairwise metrics are easy to misuse when candidates are run with different
settings or copied into unrelated reports. `imq::suite` applies one metric set
to every candidate, preserves failures, and ranks the resulting rows without
combining unlike score units.

## CLI

```bash
imq suite reference.png baseline.png codec-a.png codec-b.png \
  --metrics psnr,ssim,wssim,ms-ssim,mse,mae,mae:color,maxae \
  --primary-metric psnr \
  --weight psnr=2 --weight mse=0 \
  --baseline baseline.png \
  --rule 'psnr>=35' \
  --rule 'mae:color.abs_error_p99<=0.1' \
  --rule 'psnr>=baseline-0.5' \
  --rule 'mae<=baseline*1.1' \
  --jobs 4 \
  --format json
```

Absolute rules are evaluated independently for each candidate.
Baseline-relative rules use the named baseline candidate and accept these
right-hand-side forms:

- `baseline`
- `baseline+OFFSET` or `baseline-OFFSET`
- `baseline*SCALE`
- `baseline/DIVISOR`

The process exits non-zero when a candidate cannot be compared or any gate
fails. The default continue policy records comparison failures and finishes the
remaining work; `--fail-fast` returns the first failure in input order.

## Ranking semantics

Each directional metric is sorted in its declared quality direction. Rank 1 is
best and ties use competition ranking (`1, 1, 3`). Absolute and relative score
tolerances prevent insignificant floating-point differences from splitting a
tie.

When `--primary-metric` is set, that metric controls the top-level rank.
Otherwise, the suite averages ordinal ranks across all directional metrics.
Ordinal ranks are averaged instead of raw scores, so dB, unitless similarities,
and normalized errors are never added together. Repeated `--weight
METRIC=WEIGHT` options weight the ordinal mean; a zero weight removes a metric
from consensus ranking without removing it from the report or Pareto analysis.

The report also marks the Pareto frontier. A candidate is Pareto-optimal when no
other successful candidate is at least as good on every shared directional
metric and strictly better on one. This exposes quality tradeoffs that a single
winner would hide.

## Rust API

```rust
use imq::{
    BaselineThresholdRule, ComparisonCandidate, ComparisonSuiteOptions,
    MetricSet, compare_candidate_suite,
};

let candidates = [
    ComparisonCandidate::new("baseline", baseline.as_view()),
    ComparisonCandidate::new("candidate-a", candidate_a.as_view()),
    ComparisonCandidate::new("candidate-b", candidate_b.as_view()),
];
let metrics = MetricSet::from_csv("psnr,ssim,wssim,mse,mae")?;
let options = ComparisonSuiteOptions {
    primary_metric: Some("psnr".into()),
    baseline_candidate: Some("baseline".into()),
    baseline_rules: vec![
        BaselineThresholdRule::parse("psnr>=baseline-0.5")?,
    ],
    ..ComparisonSuiteOptions::default()
};
let report = compare_candidate_suite(
    Some("reference".into()),
    &reference.as_view(),
    &candidates,
    &metrics,
    &options,
)?;

assert_eq!(report.winner().map(|item| item.overall_rank), Some(Some(1)));
# Ok::<(), imq::Error>(())
```

`ComparisonCandidate` borrows validated frames, so the core adds no decode or
filesystem policy. `max_threads = 1` forces sequential work; parallel execution
is deterministic in report and ranking order.
