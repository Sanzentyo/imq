use imq::Result;
use imq::adapters::image_crate;
use imq::gpu::{GpuComparator, GpuErrorMetric};
use std::io;
use std::path::PathBuf;

fn main() -> Result<()> {
    let paths = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if paths.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: gpu_mse <reference> <distorted> [<distorted> ...]",
        )
        .into());
    }

    let comparator = GpuComparator::new()?;
    if let Some(capabilities) = comparator.capabilities() {
        eprintln!(
            "adapter={} backend={} device_type={} max_error_pixels={}",
            capabilities.adapter_name,
            capabilities.backend,
            capabilities.device_type,
            capabilities.max_error_stats_pixels
        );
    } else if let Some(error) = comparator.initialization_error() {
        eprintln!("wgpu unavailable; using CPU fallback: {error}");
    }

    let reference = image_crate::load_image_path(&paths[0])?;
    let candidates = paths[1..]
        .iter()
        .map(image_crate::load_image_path)
        .collect::<Result<Vec<_>>>()?;
    let candidate_views = candidates
        .iter()
        .map(|frame| frame.as_view())
        .collect::<Vec<_>>();
    let candidate_refs = candidate_views.iter().collect::<Vec<_>>();
    let results = comparator.compare_many_rgba8(
        &reference.as_view(),
        &candidate_refs,
        &GpuErrorMetric::ALL,
    )?;
    for (index, (distorted_path, result)) in paths[1..].iter().zip(results).enumerate() {
        println!(
            "pair[{index}] reference={} distorted={} execution={:?}",
            paths[0].display(),
            distorted_path.display(),
            result.execution
        );
        for metric in result.metrics {
            println!("  {} = {} {}", metric.name, metric.score, metric.unit);
        }
    }
    Ok(())
}
