//! wgpu compute kernels for image-quality primitives.
//!
//! The first kernel is intentionally narrow: RGBA8 MSE. That is the most common
//! interchange format when frames come from `image` or `ffmpeg -pix_fmt rgba`,
//! and it maps cleanly to a single `u32` per pixel on the GPU.

use crate::frame::{FrameView, PixelFormat, Validated};
use crate::metrics::{Direction, MetricOutput};
use crate::{Error, Result};
use bytemuck::{Pod, Zeroable};
use std::collections::BTreeMap;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;
const SHADER: &str = include_str!("shaders/mse_rgba8.wgsl");

/// GPU-computed MSE result for RGBA8 frames.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuMse {
    /// Mean squared error over R, G, B, and A channels in code-value units.
    pub mse: f64,
    /// Per-channel MSE in RGBA order.
    pub channel_mse: [f64; 4],
    /// Pixel count used by the kernel.
    pub pixels: u64,
}

impl GpuMse {
    /// Converts this result into a metric output.
    pub fn into_metric_output(self) -> MetricOutput {
        let mut details = BTreeMap::new();
        details.insert("mse_r".to_string(), self.channel_mse[0]);
        details.insert("mse_g".to_string(), self.channel_mse[1]);
        details.insert("mse_b".to_string(), self.channel_mse[2]);
        details.insert("mse_a".to_string(), self.channel_mse[3]);
        details.insert("pixels".to_string(), self.pixels as f64);
        MetricOutput {
            name: "gpu_mse_rgba8".to_string(),
            score: self.mse,
            direction: Direction::LowerIsBetter,
            unit: "code_value^2".to_string(),
            details,
        }
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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
        let reference_words = rgba8_words(reference)?;
        let distorted_words = rgba8_words(distorted)?;
        let partial_bytes = u64::from(groups) * std::mem::size_of::<[f32; 4]>() as u64;

        let ref_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imq-mse-ref"),
                contents: bytemuck::cast_slice(&reference_words),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let dist_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imq-mse-dist"),
                contents: bytemuck::cast_slice(&distorted_words),
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
            _pad0: 0,
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
            pass.dispatch_workgroups(groups, 1, 1);
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
        let partials: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
        let mut sums = [0.0f64; 4];
        for partial in partials {
            for c in 0..4 {
                sums[c] += f64::from(partial[c]);
            }
        }
        drop(mapped);
        readback.unmap();

        let px = pixels as f64;
        let channel_mse = [sums[0] / px, sums[1] / px, sums[2] / px, sums[3] / px];
        let mse = sums.iter().sum::<f64>() / (px * 4.0);
        Ok(GpuMse {
            mse,
            channel_mse,
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

fn rgba8_words(frame: &FrameView<'_, Validated>) -> Result<Vec<u32>> {
    let dims = frame.dimensions();
    let (w, h) = dims.as_usize()?;
    let row_bytes = w
        .checked_mul(4)
        .ok_or_else(|| Error::invalid_frame("RGBA row byte size overflow"))?;
    let plane = frame.plane(0)?;
    let mut out = Vec::with_capacity(dims.pixels()?);
    for y in 0..h {
        let row = plane.row(y, row_bytes)?;
        for px in row.chunks_exact(4) {
            out.push(u32::from_le_bytes([px[0], px[1], px[2], px[3]]));
        }
    }
    Ok(out)
}
