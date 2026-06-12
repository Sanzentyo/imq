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
const SSIM_WINDOW_SHADER: &str = include_str!("shaders/ssim_window_rgba8.wgsl");

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

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SsimWindowParams {
    width: u32,
    height: u32,
    window_width: u32,
    window_height: u32,
    stride_x: u32,
    stride_y: u32,
    windows_x: u32,
    window_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SsimWindowOut {
    values: [f32; 4],
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

    /// Computes box-windowed luma SSIM for RGBA8 frames on the GPU.
    ///
    /// The shader evaluates one local SSIM window per invocation and the CPU
    /// performs the final mean reduction after readback. This keeps the public
    /// API small while moving the expensive per-window sample scans to wgpu.
    pub fn windowed_ssim_rgba8(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
        window_width: usize,
        window_height: usize,
        stride_x: usize,
        stride_y: usize,
    ) -> Result<MetricOutput> {
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
                "gpu_wssim_rgba8 currently requires RGBA8 inputs",
            ));
        }

        let (width, height) = reference.dimensions().as_usize()?;
        let window_width = window_width.max(1).min(width);
        let window_height = window_height.max(1).min(height);
        let stride_x = stride_x.max(1);
        let stride_y = stride_y.max(1);
        let windows_x = window_axis_count(width, window_width, stride_x);
        let windows_y = window_axis_count(height, window_height, stride_y);
        let window_count = windows_x
            .checked_mul(windows_y)
            .ok_or_else(|| Error::unsupported("gpu_wssim_rgba8 window count overflows usize"))?;
        if window_count > u32::MAX as usize {
            return Err(Error::unsupported(
                "gpu_wssim_rgba8 supports up to u32::MAX local windows",
            ));
        }

        let groups = (window_count as u32).div_ceil(WORKGROUP_SIZE).max(1);
        let output_bytes = window_count
            .checked_mul(std::mem::size_of::<SsimWindowOut>())
            .ok_or_else(|| {
                Error::unsupported("gpu_wssim_rgba8 output byte count overflows usize")
            })? as u64;
        let reference_bytes = rgba8_bytes(reference)?;
        let distorted_bytes = rgba8_bytes(distorted)?;

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("imq-ssim-window-rgba8"),
                source: wgpu::ShaderSource::Wgsl(SSIM_WINDOW_SHADER.into()),
            });
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("imq-ssim-window-rgba8-bgl"),
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
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("imq-ssim-window-rgba8-layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("imq-ssim-window-rgba8-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let ref_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imq-ssim-ref"),
                contents: reference_bytes.as_ref(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let dist_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imq-ssim-dist"),
                contents: distorted_bytes.as_ref(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("imq-ssim-window-output"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("imq-ssim-window-readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params = SsimWindowParams {
            width: width as u32,
            height: height as u32,
            window_width: window_width as u32,
            window_height: window_height as u32,
            stride_x: stride_x as u32,
            stride_y: stride_y as u32,
            windows_x: windows_x as u32,
            window_count: window_count as u32,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imq-ssim-window-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("imq-ssim-window-bind-group"),
            layout: &bind_group_layout,
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
                    resource: output_buffer.as_entire_binding(),
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
                label: Some("imq-ssim-window-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("imq-ssim-window-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, output_bytes);
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
        let windows: &[SsimWindowOut] = bytemuck::cast_slice(&mapped);
        let mut sum_ssim = 0.0;
        let mut sum_cs = 0.0;
        let mut min_ssim = f64::INFINITY;
        let mut max_ssim = f64::NEG_INFINITY;
        for window in windows {
            let ssim = f64::from(window.values[0]);
            let cs = f64::from(window.values[1]);
            sum_ssim += ssim;
            sum_cs += cs;
            min_ssim = min_ssim.min(ssim);
            max_ssim = max_ssim.max(ssim);
        }
        let count = windows.len() as f64;
        drop(mapped);
        readback.unmap();
        Ok(MetricOutput::new(
            "gpu_wssim_rgba8",
            sum_ssim / count,
            "unitless",
            Direction::HigherIsBetter,
        )
        .with_detail("windows", count)
        .with_detail("window_width", window_width as f64)
        .with_detail("window_height", window_height as f64)
        .with_detail("stride_x", stride_x as f64)
        .with_detail("stride_y", stride_y as f64)
        .with_detail("mean_contrast_structure", sum_cs / count)
        .with_detail("min_window_ssim", min_ssim)
        .with_detail("max_window_ssim", max_ssim))
    }
}

fn window_axis_count(size: usize, window: usize, stride: usize) -> usize {
    let max_start = size.saturating_sub(window);
    1 + max_start.div_ceil(stride.max(1))
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
