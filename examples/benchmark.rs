use imq::{FrameOwned, MetricSet, PixelFormat, Result};
use std::hint::black_box;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
struct Config {
    width: u32,
    height: u32,
    iterations: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            width: 3840,
            height: 2160,
            iterations: 10,
        }
    }
}

fn main() -> Result<()> {
    let config = parse_config();
    let reference = synthetic_frame(config.width, config.height, 0)?;
    let distorted = synthetic_frame(config.width, config.height, 17)?;
    let pixels = u64::from(config.width) * u64::from(config.height);

    println!(
        "benchmark: {}x{} pixels={} iterations={}",
        config.width, config.height, pixels, config.iterations
    );

    let error_metrics = MetricSet::from_csv("psnr,mse,rmse,mae,maxae")?;
    let separate_error_metrics = [
        MetricSet::from_csv("psnr")?,
        MetricSet::from_csv("mse")?,
        MetricSet::from_csv("rmse")?,
        MetricSet::from_csv("mae")?,
        MetricSet::from_csv("maxae")?,
    ];
    let cpu_separate_error = time_iterations(config.iterations, || {
        for metric in &separate_error_metrics {
            let outputs = metric.compare(&reference.as_view(), &distorted.as_view())?;
            black_box(outputs);
        }
        Ok(())
    })?;
    print_result(
        "cpu-error-metrics-separate",
        cpu_separate_error,
        pixels,
        config.iterations,
    );

    let cpu_error = time_iterations(config.iterations, || {
        let outputs = error_metrics.compare(&reference.as_view(), &distorted.as_view())?;
        black_box(outputs);
        Ok(())
    })?;
    print_result("cpu-error-metrics", cpu_error, pixels, config.iterations);

    let default_metrics = MetricSet::defaults();
    let cpu_default = time_iterations(config.iterations, || {
        let outputs = default_metrics.compare(&reference.as_view(), &distorted.as_view())?;
        black_box(outputs);
        Ok(())
    })?;
    print_result(
        "cpu-default-metrics",
        cpu_default,
        pixels,
        config.iterations,
    );

    #[cfg(feature = "gpu")]
    {
        let gpu = imq::gpu::GpuContext::new()?;
        let gpu_time = time_iterations(config.iterations, || {
            let stats = gpu.error_stats_rgba8(&reference.as_view(), &distorted.as_view())?;
            black_box(stats);
            Ok(())
        })?;
        print_result("gpu-rgba8-error-stats", gpu_time, pixels, config.iterations);
    }

    Ok(())
}

fn parse_config() -> Config {
    let mut config = Config::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--width" | "-w" => {
                config.width = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(config.width);
            }
            "--height" | "-h" => {
                config.height = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(config.height);
            }
            "--iterations" | "-n" => {
                config.iterations = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(config.iterations);
            }
            _ => {}
        }
    }
    config.iterations = config.iterations.max(1);
    config
}

fn synthetic_frame(width: u32, height: u32, phase: u8) -> Result<FrameOwned> {
    let pixels = usize::try_from(u64::from(width) * u64::from(height))
        .map_err(|_| imq::Error::invalid_frame("synthetic frame is too large"))?;
    let mut data = Vec::with_capacity(pixels * 4);
    for index in 0..pixels {
        let x = (index % width as usize) as u8;
        let y = (index / width as usize) as u8;
        data.push(x.wrapping_add(phase));
        data.push(y.wrapping_mul(3).wrapping_add(phase));
        data.push(x.wrapping_add(y).wrapping_add(phase));
        data.push(255);
    }
    FrameOwned::packed_tight(data, width, height, PixelFormat::Rgba8)
}

fn time_iterations(mut iterations: usize, mut f: impl FnMut() -> Result<()>) -> Result<Duration> {
    f()?;
    let start = Instant::now();
    while iterations > 0 {
        f()?;
        iterations -= 1;
    }
    Ok(start.elapsed())
}

fn print_result(label: &str, duration: Duration, pixels: u64, iterations: usize) {
    let seconds = duration.as_secs_f64();
    let mpix_per_second = pixels as f64 * iterations as f64 / seconds / 1_000_000.0;
    println!(
        "{label}: {:.3} ms total, {:.2} MPix/s",
        seconds * 1000.0,
        mpix_per_second
    );
}
