use imq::Result;
use imq::adapters::image_crate;
use imq::gpu::GpuContext;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    let reference_path = args.next().expect("usage: gpu_mse <reference> <distorted>");
    let distorted_path = args.next().expect("usage: gpu_mse <reference> <distorted>");

    let reference = image_crate::load_image_path(reference_path)?;
    let distorted = image_crate::load_image_path(distorted_path)?;
    let gpu = GpuContext::new()?;
    let result = gpu.error_stats_rgba8(&reference.as_view(), &distorted.as_view())?;
    println!("gpu_mse_rgba8 = {}", result.mse);
    println!("gpu_rmse_rgba8 = {}", result.rmse);
    println!("gpu_mae_rgba8 = {}", result.mae);
    println!("gpu_maxae_rgba8 = {}", result.max_abs);
    Ok(())
}
