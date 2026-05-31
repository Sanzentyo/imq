//! wgpu compute kernels for image-quality primitives.
//!
//! The first kernel is intentionally narrow: RGBA8 full-reference error stats.
//! That is the most common interchange format when frames come from `image` or
//! `ffmpeg -pix_fmt rgba`, and it maps cleanly to a single `u32` per pixel on
//! the GPU.

use crate::frame::{FrameView, PixelFormat, Validated};
use crate::metrics::{Direction, MetricOutput};
use crate::{Error, Result};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;
const SHADER: &str = include_str!("shaders/mse_rgba8.wgsl");

/// GPU-computed RGBA8 error statistics.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuRgba8ErrorStats {
    /// Mean squared error over R, G, B, and A channels in code-value units.
    pub mse: f64,
    /// Per-channel MSE in RGBA order.
    pub channel_mse: [f64; 4],
    /// Root mean squared error over R, G, B, and A channels in code-value units.
    pub rmse: f64,
    /// Mean absolute error over R, G, B, and A channels in code-value units.
    pub mae: f64,
    /// Per-channel MAE in RGBA order.
    pub channel_mae: [f64; 4],
    /// Maximum absolute error in code-value units.
    pub max_abs: f64,
    /// Per-channel maximum absolute error in RGBA order.
    pub channel_max_abs: [f64; 4],
    /// Pixel count used by the kernel.
    pub pixels: u64,
}

/// Backward-compatible alias for the original GPU MSE result type.
pub type GpuMse = GpuRgba8ErrorStats;

impl GpuRgba8ErrorStats {
    /// Converts this result into the historical `gpu_mse_rgba8` metric output.
    pub fn into_metric_output(self) -> MetricOutput {
        let mut details = BTreeMap::new();
        details.insert("mse_r".to_string(), self.channel_mse[0]);
        details.insert("mse_g".to_string(), self.channel_mse[1]);
        details.insert("mse_b".to_string(), self.channel_mse[2]);
        details.insert("mse_a".to_string(), self.channel_mse[3]);
        details.insert("mae".to_string(), self.mae);
        details.insert("maxae".to_string(), self.max_abs);
        details.insert("pixels".to_string(), self.pixels as f64);
        MetricOutput {
            name: "gpu_mse_rgba8".to_string(),
            score: self.mse,
            direction: Direction::LowerIsBetter,
            unit: "code_value^2".to_string(),
            details,
        }
    }

    /// Converts the GPU stats into standard normalized error metric outputs.
    pub fn into_normalized_metric_outputs(self) -> Vec<MetricOutput> {
        let mse = self.mse / (255.0 * 255.0);
        let rmse = self.rmse / 255.0;
        let mae = self.mae / 255.0;
        let max_abs = self.max_abs / 255.0;
        let psnr = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (1.0 / mse).log10()
        };
        vec![
            MetricOutput::new("mse", mse, "normalized_code^2", Direction::LowerIsBetter)
                .with_detail("pixels", self.pixels as f64),
            MetricOutput::new("rmse", rmse, "normalized_code", Direction::LowerIsBetter)
                .with_detail("pixels", self.pixels as f64),
            MetricOutput::new("psnr", psnr, "dB", Direction::HigherIsBetter)
                .with_detail("mse", mse)
                .with_detail("pixels", self.pixels as f64),
            MetricOutput::new("mae", mae, "normalized_code", Direction::LowerIsBetter)
                .with_detail("pixels", self.pixels as f64),
            MetricOutput::new(
                "maxae",
                max_abs,
                "normalized_code",
                Direction::LowerIsBetter,
            )
            .with_detail("pixels", self.pixels as f64),
        ]
    }
}

/// Long-lived wgpu context used to run compute kernels.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Params {
    pixel_count: u32,
    groups_x: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PartialStats {
    sum_sq: [f32; 4],
    sum_abs: [f32; 4],
    max_abs: [f32; 4],
}

impl GpuContext {
    /// Creates a GPU context synchronously.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    /// Creates a GPU context asynchronously.
    pub async fn new_async() -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| Error::Gpu(format!("no suitable wgpu adapter: {e}")))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("imq-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| Error::Gpu(format!("failed to request wgpu device: {e}")))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("imq-mse-rgba8"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("imq-mse-rgba8-bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("imq-mse-rgba8-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("imq-mse-rgba8-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            bind_group_layout,
            pipeline,
        })
    }

    /// Computes RGBA8 MSE on the GPU.
    ///
    /// Inputs may be strided; rows are compacted before upload. The core frame
    /// objects still remain borrowed and I/O-free.
    pub fn mse_rgba8(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<GpuMse> {
        self.error_stats_rgba8(reference, distorted)
    }

    /// Computes RGBA8 MSE/RMSE/MAE/max absolute error statistics on the GPU.
    ///
    /// The shader reduces squared error, absolute error, and max absolute error
    /// in one pass; PSNR can then be derived from the returned MSE.
    pub fn error_stats_rgba8(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
    ) -> Result<GpuRgba8ErrorStats> {
        if reference.dimensions() != distorted.dimensions() {
            return Err(Error::incompatible(format!(
                "dimension mismatch: {:?} vs {:?}",
                reference.dimensions(),
                distorted.dimensions()
            )));
        }
        if reference.pixel_format() != PixelFormat::Rgba8
            || distorted.pixel_format() != PixelFormat::Rgba8
        {
            return Err(Error::unsupported(
                "gpu_mse_rgba8 currently requires RGBA8 inputs",
            ));
        }

        let pixels = reference.dimensions().pixels()?;
        if pixels > u32::MAX as usize {
            return Err(Error::unsupported(
                "gpu_mse_rgba8 supports up to u32::MAX pixels per dispatch",
            ));
        }
        let groups = (pixels as u32).div_ceil(WORKGROUP_SIZE).max(1);
        let groups_x = groups.min(65_535);
        let groups_y = groups.div_ceil(groups_x);
        if groups_y > 65_535 {
            return Err(Error::unsupported(
                "gpu_mse_rgba8 dispatch would exceed wgpu workgroup limits",
            ));
        }
        let reference_bytes = rgba8_bytes(reference)?;
        let distorted_bytes = rgba8_bytes(distorted)?;
        let partial_bytes = u64::from(groups) * std::mem::size_of::<PartialStats>() as u64;

        let ref_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imq-mse-ref"),
                contents: reference_bytes.as_ref(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let dist_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imq-mse-dist"),
                contents: distorted_bytes.as_ref(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let partial_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("imq-mse-partial"),
            size: partial_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("imq-mse-readback"),
            size: partial_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params = Params {
            pixel_count: pixels as u32,
            groups_x,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imq-mse-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("imq-mse-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ref_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dist_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: partial_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("imq-mse-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("imq-mse-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        encoder.copy_buffer_to_buffer(&partial_buffer, 0, &readback, 0, partial_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("wgpu poll failed: {e}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("wgpu map callback channel failed: {e}")))?
            .map_err(|e| Error::Gpu(format!("wgpu readback map failed: {e}")))?;

        let mapped = slice.get_mapped_range();
        let partials: &[PartialStats] = bytemuck::cast_slice(&mapped);
        let mut sum_sq = [0.0f64; 4];
        let mut sum_abs = [0.0f64; 4];
        let mut max_abs = [0.0f64; 4];
        for partial in partials {
            for c in 0..4 {
                sum_sq[c] += f64::from(partial.sum_sq[c]);
                sum_abs[c] += f64::from(partial.sum_abs[c]);
                max_abs[c] = max_abs[c].max(f64::from(partial.max_abs[c]));
            }
        }
        drop(mapped);
        readback.unmap();

        let px = pixels as f64;
        let channel_mse = [
            sum_sq[0] / px,
            sum_sq[1] / px,
            sum_sq[2] / px,
            sum_sq[3] / px,
        ];
        let channel_mae = [
            sum_abs[0] / px,
            sum_abs[1] / px,
            sum_abs[2] / px,
            sum_abs[3] / px,
        ];
        let mse = sum_sq.iter().sum::<f64>() / (px * 4.0);
        let mae = sum_abs.iter().sum::<f64>() / (px * 4.0);
        Ok(GpuRgba8ErrorStats {
            mse,
            channel_mse,
            rmse: mse.sqrt(),
            mae,
            channel_mae,
            max_abs: max_abs.into_iter().fold(0.0, f64::max),
            channel_max_abs: max_abs,
            pixels: pixels as u64,
        })
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn rgba8_bytes<'a>(frame: &FrameView<'a, Validated>) -> Result<Cow<'a, [u8]>> {
    let dims = frame.dimensions();
    let (w, h) = dims.as_usize()?;
    let row_bytes = w
        .checked_mul(4)
        .ok_or_else(|| Error::invalid_frame("RGBA row byte size overflow"))?;
    let plane = frame.plane(0)?;
    let total_bytes = row_bytes
        .checked_mul(h)
        .ok_or_else(|| Error::invalid_frame("RGBA buffer byte size overflow"))?;
    if plane.stride == row_bytes {
        return plane
            .data
            .get(..total_bytes)
            .map(Cow::Borrowed)
            .ok_or_else(|| Error::invalid_frame("RGBA plane is shorter than expected"));
    }
    let mut out = Vec::with_capacity(total_bytes);
    for y in 0..h {
        let row = plane.row(y, row_bytes)?;
        out.extend_from_slice(row);
    }
    Ok(Cow::Owned(out))
}
