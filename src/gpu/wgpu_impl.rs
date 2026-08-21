//! wgpu compute kernels for image-quality primitives.
//!
//! The kernels intentionally focus on RGBA8 full-reference workloads. That is
//! the common interchange format for image decoders, ffmpeg rawvideo, WebGL,
//! and WebGPU capture, and maps cleanly to one `u32` per pixel on the GPU.

use crate::frame::{FrameView, PixelFormat, Validated};
use crate::metrics::{Direction, Metric, MetricOutput, WindowedSsim};
use crate::{Error, Result};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;
// A WSSIM window is scanned serially by one shader invocation. Keeping the
// per-invocation and aggregate work bounded avoids GPU watchdog resets for
// adversarial layouts while leaving conventional 8x8/11x11 windows far below
// the limits.
const MAX_GPU_WSSIM_SAMPLES_PER_WINDOW: u64 = 1 << 20;
const MAX_GPU_WSSIM_TOTAL_SAMPLE_VISITS: u64 = 1 << 28;
// Bound the peak transient working set created by the true-batch path. This
// includes upload overlap and GPU output/readback buffers, but not the caller's
// source frames, which already exist before the comparison.
const MAX_GPU_BATCH_WORKING_SET_BYTES: u64 = 512 * 1024 * 1024;
const SHADER: &str = include_str!("shaders/mse_rgba8.wgsl");
const SSIM_WINDOW_SHADER: &str = include_str!("shaders/ssim_window_rgba8.wgsl");

/// Adapter power preference used while creating a [`GpuContext`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum GpuPowerPreference {
    /// Let wgpu choose without a power preference.
    None,
    /// Prefer an integrated or otherwise power-efficient adapter.
    LowPower,
    /// Prefer the highest-performance adapter.
    #[default]
    HighPerformance,
}

impl From<GpuPowerPreference> for wgpu::PowerPreference {
    fn from(value: GpuPowerPreference) -> Self {
        match value {
            GpuPowerPreference::None => Self::None,
            GpuPowerPreference::LowPower => Self::LowPower,
            GpuPowerPreference::HighPerformance => Self::HighPerformance,
        }
    }
}

/// Options controlling wgpu adapter selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuContextOptions {
    /// Adapter power preference.
    pub power_preference: GpuPowerPreference,
    /// Require a fallback adapter, which is normally a software implementation.
    pub force_fallback_adapter: bool,
}

/// Stable, serializable adapter and limit information for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuCapabilities {
    /// Human-readable adapter name.
    pub adapter_name: String,
    /// Graphics API backend, for example `Metal`, `Vulkan`, or `Dx12`.
    pub backend: String,
    /// Physical device category reported by wgpu.
    pub device_type: String,
    /// Driver name.
    pub driver: String,
    /// Backend-specific driver information.
    pub driver_info: String,
    /// Backend-specific vendor identifier.
    pub vendor_id: u32,
    /// Backend-specific device identifier.
    pub device_id: u32,
    /// Whether this context explicitly requested a fallback adapter.
    pub fallback_adapter: bool,
    /// Maximum size of an individual GPU buffer.
    pub max_buffer_size: u64,
    /// Maximum size exposed through one storage binding.
    pub max_storage_buffer_binding_size: u64,
    /// Maximum workgroup count in each dispatch dimension.
    pub max_compute_workgroups_per_dimension: u32,
    /// Maximum tightly packed RGBA8 input pixel count accepted by one buffer.
    pub max_rgba8_input_pixels: u64,
    /// Conservative maximum RGBA8 pixel count accepted by the error kernel.
    pub max_error_stats_pixels: u64,
    /// Conservative maximum local-window count accepted by the SSIM kernel.
    pub max_windowed_ssim_windows: u64,
}

/// Error metric derivations available from one GPU reduction pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum GpuErrorMetric {
    /// Mean squared error.
    Mse,
    /// Root mean squared error.
    Rmse,
    /// Peak signal-to-noise ratio.
    Psnr,
    /// Mean absolute error.
    Mae,
    /// Maximum absolute error.
    MaxAbsoluteError,
}

/// RGBA8 channel domain used when projecting GPU error statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum GpuErrorDomain {
    /// Compare RGB channels and ignore alpha.
    Color,
    /// Compare all RGBA channels.
    #[default]
    All,
}

impl GpuErrorMetric {
    /// Every metric derived by the RGBA8 error kernel.
    pub const ALL: [Self; 5] = [
        Self::Mse,
        Self::Rmse,
        Self::Psnr,
        Self::Mae,
        Self::MaxAbsoluteError,
    ];
}

/// Local-window configuration for the GPU SSIM kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuWindowedSsimOptions {
    /// Window width in pixels.
    pub window_width: usize,
    /// Window height in pixels.
    pub window_height: usize,
    /// Horizontal distance between adjacent windows.
    pub stride_x: usize,
    /// Vertical distance between adjacent windows.
    pub stride_y: usize,
}

impl Default for GpuWindowedSsimOptions {
    fn default() -> Self {
        Self {
            window_width: 8,
            window_height: 8,
            stride_x: 8,
            stride_y: 8,
        }
    }
}

/// GPU-computed RGBA8 error statistics.
#[derive(Debug, Clone, PartialEq)]
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
        self.normalized_metric_outputs_in_domain(&GpuErrorMetric::ALL, GpuErrorDomain::All)
    }

    /// Produces selected normalized metric outputs without rerunning a kernel.
    ///
    /// Scores cover all four RGBA channels. The `samples`, `channels`, and
    /// `pixels` details make that domain explicit for report consumers.
    pub fn normalized_metric_outputs(&self, metrics: &[GpuErrorMetric]) -> Vec<MetricOutput> {
        self.normalized_metric_outputs_in_domain(metrics, GpuErrorDomain::All)
    }

    /// Produces selected normalized metric outputs for RGB color or all RGBA
    /// channels without rerunning a kernel.
    pub fn normalized_metric_outputs_in_domain(
        &self,
        metrics: &[GpuErrorMetric],
        domain: GpuErrorDomain,
    ) -> Vec<MetricOutput> {
        let (mse_code, mae_code, max_abs_code, channels, suffix) = match domain {
            GpuErrorDomain::Color => (
                self.channel_mse[..3].iter().sum::<f64>() / 3.0,
                self.channel_mae[..3].iter().sum::<f64>() / 3.0,
                self.channel_max_abs[..3]
                    .iter()
                    .copied()
                    .fold(0.0, f64::max),
                3.0,
                "color",
            ),
            GpuErrorDomain::All => (self.mse, self.mae, self.max_abs, 4.0, "all"),
        };
        let mse = mse_code / (255.0 * 255.0);
        let rmse = mse.sqrt();
        let mae = mae_code / 255.0;
        let max_abs = max_abs_code / 255.0;
        let psnr = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (1.0 / mse).log10()
        };
        let details = |mut output: MetricOutput| {
            output
                .details
                .insert("pixels".to_string(), self.pixels as f64);
            output
                .details
                .insert("samples".to_string(), self.pixels as f64 * channels);
            output.details.insert("channels".to_string(), channels);
            output
        };
        metrics
            .iter()
            .map(|metric| match metric {
                GpuErrorMetric::Mse => details(MetricOutput::new(
                    format!("mse:{suffix}"),
                    mse,
                    "normalized_code^2",
                    Direction::LowerIsBetter,
                )),
                GpuErrorMetric::Rmse => details(MetricOutput::new(
                    format!("rmse:{suffix}"),
                    rmse,
                    "normalized_code",
                    Direction::LowerIsBetter,
                )),
                GpuErrorMetric::Psnr => details(
                    MetricOutput::new(
                        format!("psnr:{suffix}"),
                        psnr,
                        "dB",
                        Direction::HigherIsBetter,
                    )
                    .with_detail("mse", mse),
                ),
                GpuErrorMetric::Mae => details(MetricOutput::new(
                    format!("mae:{suffix}"),
                    mae,
                    "normalized_code",
                    Direction::LowerIsBetter,
                )),
                GpuErrorMetric::MaxAbsoluteError => details(MetricOutput::new(
                    format!("maxae:{suffix}"),
                    max_abs,
                    "normalized_code",
                    Direction::LowerIsBetter,
                )),
            })
            .collect()
    }
}

/// Long-lived wgpu context used to run compute kernels.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    error_bind_group_layout: wgpu::BindGroupLayout,
    error_pipeline: wgpu::ComputePipeline,
    ssim_bind_group_layout: wgpu::BindGroupLayout,
    ssim_pipeline: wgpu::ComputePipeline,
    options: GpuContextOptions,
    capabilities: GpuCapabilities,
    limits: wgpu::Limits,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Params {
    pixel_count: u32,
    groups_x: u32,
    group_count: u32,
    pair_count: u32,
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
    groups_x: u32,
    group_count: u32,
    _pad2: u32,
    _pad3: u32,
    reference_luma_weights: [f32; 4],
    distorted_luma_weights: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SsimWindowOut {
    values: [f32; 4],
}

impl GpuContext {
    /// Creates a GPU context synchronously.
    pub fn new() -> Result<Self> {
        Self::new_with_options(GpuContextOptions::default())
    }

    /// Creates a GPU context synchronously with explicit adapter options.
    pub fn new_with_options(options: GpuContextOptions) -> Result<Self> {
        pollster::block_on(Self::new_async_with_options(options))
    }

    /// Creates a GPU context asynchronously.
    pub async fn new_async() -> Result<Self> {
        Self::new_async_with_options(GpuContextOptions::default()).await
    }

    /// Creates a GPU context asynchronously with explicit adapter options.
    pub async fn new_async_with_options(options: GpuContextOptions) -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: options.power_preference.into(),
                compatible_surface: None,
                force_fallback_adapter: options.force_fallback_adapter,
            })
            .await
            .map_err(|e| Error::Gpu(format!("no suitable wgpu adapter: {e}")))?;

        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            max_compute_workgroups_per_dimension: adapter_limits
                .max_compute_workgroups_per_dimension,
            ..wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("imq-device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| Error::Gpu(format!("failed to request wgpu device: {e}")))?;

        let limits = device.limits();
        let capabilities = capabilities_from_adapter(&adapter_info, &limits, options);

        let error_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("imq-mse-rgba8"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let error_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let error_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("imq-mse-rgba8-layout"),
                bind_group_layouts: &[Some(&error_bind_group_layout)],
                immediate_size: 0,
            });

        let error_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("imq-mse-rgba8-pipeline"),
            layout: Some(&error_pipeline_layout),
            module: &error_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let ssim_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("imq-ssim-window-rgba8"),
            source: wgpu::ShaderSource::Wgsl(SSIM_WINDOW_SHADER.into()),
        });
        let ssim_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let ssim_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("imq-ssim-window-rgba8-layout"),
            bind_group_layouts: &[Some(&ssim_bind_group_layout)],
            immediate_size: 0,
        });
        let ssim_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("imq-ssim-window-rgba8-pipeline"),
            layout: Some(&ssim_pipeline_layout),
            module: &ssim_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            error_bind_group_layout,
            error_pipeline,
            ssim_bind_group_layout,
            ssim_pipeline,
            options,
            capabilities,
            limits,
        })
    }

    /// Returns the adapter-selection options used to build this context.
    pub fn options(&self) -> GpuContextOptions {
        self.options
    }

    /// Returns stable adapter information and effective device limits.
    pub fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }

    /// Returns the maximum candidate count accepted by
    /// [`GpuContext::error_stats_many_rgba8`] for a given pixel count.
    ///
    /// This accounts for candidate storage, partial-reduction storage, shader
    /// indexing, and the adapter's Z-dispatch limit.
    pub fn max_error_stats_candidates(&self, pixels: u64) -> u32 {
        if pixels == 0 || pixels > u64::from(u32::MAX) {
            return 0;
        }
        let storage_limit = self
            .limits
            .max_buffer_size
            .min(self.limits.max_storage_buffer_binding_size);
        let candidate_bytes = match pixels.checked_mul(4) {
            Some(bytes) if bytes != 0 => bytes,
            _ => return 0,
        };
        let groups = pixels.div_ceil(u64::from(WORKGROUP_SIZE));
        let dispatch_dimension = u64::from(self.limits.max_compute_workgroups_per_dimension);
        if groups > dispatch_dimension.saturating_mul(dispatch_dimension) {
            return 0;
        }
        let partial_bytes = match groups.checked_mul(std::mem::size_of::<PartialStats>() as u64) {
            Some(bytes) if bytes != 0 => bytes,
            _ => return 0,
        };
        let batch_budget = match MAX_GPU_BATCH_WORKING_SET_BYTES.checked_sub(candidate_bytes) {
            Some(bytes) if candidate_bytes <= MAX_GPU_BATCH_WORKING_SET_BYTES / 2 => bytes,
            _ => return 0,
        };
        let upload_bytes_per_candidate = match candidate_bytes.checked_mul(2) {
            Some(bytes) if bytes != 0 => bytes,
            _ => return 0,
        };
        let execution_bytes_per_candidate = match partial_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(candidate_bytes))
        {
            Some(bytes) if bytes != 0 => bytes,
            _ => return 0,
        };
        let candidates = (storage_limit / candidate_bytes)
            .min(storage_limit / partial_bytes)
            .min(u64::from(u32::MAX) / pixels)
            .min(u64::from(u32::MAX) / groups)
            .min(dispatch_dimension)
            .min(batch_budget / upload_bytes_per_candidate)
            .min(batch_budget / execution_bytes_per_candidate);
        candidates.min(u64::from(u32::MAX)) as u32
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
        let input_bytes = (pixels as u64)
            .checked_mul(4)
            .ok_or_else(|| Error::unsupported("gpu_mse_rgba8 input size overflows u64"))?;
        self.ensure_storage_buffer_size("gpu_mse_rgba8 input", input_bytes)?;
        let (groups, groups_x, groups_y) =
            self.dispatch_dimensions("gpu_mse_rgba8", pixels as u64)?;
        let reference_bytes = rgba8_bytes(reference)?;
        let distorted_bytes = rgba8_bytes(distorted)?;
        let partial_bytes = u64::from(groups) * std::mem::size_of::<PartialStats>() as u64;
        self.ensure_storage_buffer_size("gpu_mse_rgba8 partial output", partial_bytes)?;

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
            group_count: groups,
            pair_count: 1,
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
            layout: &self.error_bind_group_layout,
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
            pass.set_pipeline(&self.error_pipeline);
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
        let stats = error_stats_from_partials(partials, pixels as u64);
        drop(mapped);
        readback.unmap();
        Ok(stats)
    }

    /// Compares one reference against multiple equally-sized RGBA8 candidates
    /// in a single queue submission and readback.
    ///
    /// The reference is uploaded once, candidates are packed into one storage
    /// buffer, and the dispatch's Z dimension selects the candidate. This is
    /// the preferred batch path for golden-image and render-variant checks.
    pub fn error_stats_many_rgba8(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &[&FrameView<'_, Validated>],
    ) -> Result<Vec<GpuRgba8ErrorStats>> {
        if distorted.is_empty() {
            return Ok(Vec::new());
        }
        for (index, candidate) in distorted.iter().enumerate() {
            validate_rgba8_pair(reference, candidate).map_err(|error| {
                Error::Gpu(format!("RGBA8 candidate {index} is incompatible: {error}"))
            })?;
        }

        let pixels = reference.dimensions().pixels()?;
        if pixels > u32::MAX as usize {
            return Err(Error::unsupported(
                "gpu_mse_rgba8 supports up to u32::MAX pixels per candidate",
            ));
        }
        let candidate_count = u32::try_from(distorted.len())
            .map_err(|_| Error::unsupported("gpu_mse_rgba8 candidate count exceeds u32::MAX"))?;
        let dispatch_limit = self.limits.max_compute_workgroups_per_dimension;
        if candidate_count > dispatch_limit {
            return Err(Error::unsupported(format!(
                "gpu_mse_rgba8 candidate count {candidate_count} exceeds the adapter Z-dispatch limit of {dispatch_limit}"
            )));
        }
        let combined_pixels = (pixels as u64)
            .checked_mul(u64::from(candidate_count))
            .ok_or_else(|| Error::unsupported("gpu_mse_rgba8 batch pixel count overflows u64"))?;
        if combined_pixels > u64::from(u32::MAX) {
            return Err(Error::unsupported(
                "gpu_mse_rgba8 batch supports up to u32::MAX combined candidate pixels",
            ));
        }
        let input_bytes = (pixels as u64)
            .checked_mul(4)
            .ok_or_else(|| Error::unsupported("gpu_mse_rgba8 input size overflows u64"))?;
        let candidate_bytes = combined_pixels
            .checked_mul(4)
            .ok_or_else(|| Error::unsupported("gpu_mse_rgba8 batch size overflows u64"))?;
        self.ensure_storage_buffer_size("gpu_mse_rgba8 reference", input_bytes)?;
        self.ensure_storage_buffer_size("gpu_mse_rgba8 candidates", candidate_bytes)?;

        let (groups, groups_x, groups_y) =
            self.dispatch_dimensions("gpu_mse_rgba8", pixels as u64)?;
        let partial_count = u64::from(groups)
            .checked_mul(u64::from(candidate_count))
            .ok_or_else(|| Error::unsupported("gpu_mse_rgba8 partial count overflows u64"))?;
        if partial_count > u64::from(u32::MAX) {
            return Err(Error::unsupported(
                "gpu_mse_rgba8 batch supports up to u32::MAX partial reductions",
            ));
        }
        let partial_bytes = partial_count
            .checked_mul(std::mem::size_of::<PartialStats>() as u64)
            .ok_or_else(|| Error::unsupported("gpu_mse_rgba8 partial size overflows u64"))?;
        self.ensure_storage_buffer_size("gpu_mse_rgba8 partial output", partial_bytes)?;
        self.ensure_batch_working_set_size(input_bytes, candidate_bytes, partial_bytes)?;

        let reference_bytes = rgba8_bytes(reference)?;
        let ref_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imq-mse-batch-ref"),
                contents: reference_bytes.as_ref(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        drop(reference_bytes);
        let candidate_capacity = usize::try_from(candidate_bytes).map_err(|_| {
            Error::unsupported("gpu_mse_rgba8 candidate buffer size overflows usize")
        })?;
        let mut packed_candidates = Vec::with_capacity(candidate_capacity);
        for candidate in distorted {
            packed_candidates.extend_from_slice(rgba8_bytes(candidate)?.as_ref());
        }

        let dist_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imq-mse-batch-candidates"),
                contents: &packed_candidates,
                usage: wgpu::BufferUsages::STORAGE,
            });
        drop(packed_candidates);
        let partial_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("imq-mse-batch-partial"),
            size: partial_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("imq-mse-batch-readback"),
            size: partial_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params = Params {
            pixel_count: pixels as u32,
            groups_x,
            group_count: groups,
            pair_count: candidate_count,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imq-mse-batch-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("imq-mse-batch-bind-group"),
            layout: &self.error_bind_group_layout,
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
                label: Some("imq-mse-batch-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("imq-mse-batch-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.error_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, candidate_count);
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
            .map_err(|error| Error::Gpu(format!("wgpu batch poll failed: {error}")))?;
        rx.recv()
            .map_err(|error| Error::Gpu(format!("wgpu batch map callback failed: {error}")))?
            .map_err(|error| Error::Gpu(format!("wgpu batch readback map failed: {error}")))?;

        let mapped = slice.get_mapped_range();
        let partials: &[PartialStats] = bytemuck::cast_slice(&mapped);
        let groups_per_candidate = groups as usize;
        let results = partials
            .chunks_exact(groups_per_candidate)
            .map(|candidate_partials| error_stats_from_partials(candidate_partials, pixels as u64))
            .collect();
        drop(mapped);
        readback.unmap();
        Ok(results)
    }

    /// Computes error statistics for multiple frame pairs with one reusable
    /// context and cached pipelines.
    ///
    /// Results preserve input order. Validation is fail-fast and reports the
    /// zero-based pair index in the error message.
    pub fn error_stats_batch_rgba8(
        &self,
        pairs: &[(&FrameView<'_, Validated>, &FrameView<'_, Validated>)],
    ) -> Result<Vec<GpuRgba8ErrorStats>> {
        pairs
            .iter()
            .enumerate()
            .map(|(index, (reference, distorted))| {
                self.error_stats_rgba8(reference, distorted)
                    .map_err(|error| {
                        Error::Gpu(format!("RGBA8 batch pair {index} failed: {error}"))
                    })
            })
            .collect()
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
        validate_windowed_ssim_options(GpuWindowedSsimOptions {
            window_width,
            window_height,
            stride_x,
            stride_y,
        })?;

        let (width, height) = reference.dimensions().as_usize()?;
        let window_width = window_width.min(width);
        let window_height = window_height.min(height);
        let stride_x_u32 = u32::try_from(stride_x).map_err(|_| {
            Error::unsupported("gpu_wssim_rgba8 horizontal stride exceeds u32::MAX")
        })?;
        let stride_y_u32 = u32::try_from(stride_y)
            .map_err(|_| Error::unsupported("gpu_wssim_rgba8 vertical stride exceeds u32::MAX"))?;
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
        ensure_gpu_wssim_workload(window_width, window_height, window_count)?;

        let input_pixels = (width as u64)
            .checked_mul(height as u64)
            .ok_or_else(|| Error::unsupported("gpu_wssim_rgba8 pixel count overflows u64"))?;
        if input_pixels > u64::from(u32::MAX) {
            return Err(Error::unsupported(
                "gpu_wssim_rgba8 supports up to u32::MAX pixels",
            ));
        }
        let input_bytes = input_pixels
            .checked_mul(4)
            .ok_or_else(|| Error::unsupported("gpu_wssim_rgba8 input size overflows u64"))?;
        self.ensure_storage_buffer_size("gpu_wssim_rgba8 input", input_bytes)?;
        let (_groups, groups_x, groups_y) =
            self.dispatch_dimensions("gpu_wssim_rgba8", window_count as u64)?;
        let output_bytes = window_count
            .checked_mul(std::mem::size_of::<SsimWindowOut>())
            .ok_or_else(|| {
                Error::unsupported("gpu_wssim_rgba8 output byte count overflows usize")
            })? as u64;
        self.ensure_storage_buffer_size("gpu_wssim_rgba8 output", output_bytes)?;
        let reference_bytes = rgba8_bytes(reference)?;
        let distorted_bytes = rgba8_bytes(distorted)?;

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
            stride_x: stride_x_u32,
            stride_y: stride_y_u32,
            windows_x: windows_x as u32,
            window_count: window_count as u32,
            groups_x,
            group_count: _groups,
            _pad2: 0,
            _pad3: 0,
            reference_luma_weights: gpu_luma_weights(reference),
            distorted_luma_weights: gpu_luma_weights(distorted),
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
            layout: &self.ssim_bind_group_layout,
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
            pass.set_pipeline(&self.ssim_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
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
        let mean_ssim = sum_ssim / count;
        drop(mapped);
        readback.unmap();
        Ok(MetricOutput::new(
            "gpu_wssim_rgba8",
            mean_ssim,
            "unitless",
            Direction::HigherIsBetter,
        )
        .with_detail("windows", count)
        .with_detail("samples", input_pixels as f64)
        .with_detail("window_width", window_width as f64)
        .with_detail("window_height", window_height as f64)
        .with_detail("stride_x", stride_x as f64)
        .with_detail("stride_y", stride_y as f64)
        .with_detail("mean_contrast_structure", sum_cs / count)
        .with_detail("unweighted_mean", mean_ssim)
        .with_detail("min_window_ssim", min_ssim)
        .with_detail("max_window_ssim", max_ssim))
    }

    /// Computes windowed SSIM using an options value suitable for reuse in
    /// higher-level comparison pipelines.
    pub fn windowed_ssim_rgba8_with_options(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
        options: GpuWindowedSsimOptions,
    ) -> Result<MetricOutput> {
        self.windowed_ssim_rgba8(
            reference,
            distorted,
            options.window_width,
            options.window_height,
            options.stride_x,
            options.stride_y,
        )
    }

    fn ensure_storage_buffer_size(&self, operation: &str, bytes: u64) -> Result<()> {
        let limit = self
            .limits
            .max_buffer_size
            .min(self.limits.max_storage_buffer_binding_size);
        if bytes > limit {
            return Err(Error::unsupported(format!(
                "{operation} requires {bytes} bytes, exceeding the adapter storage-buffer limit of {limit} bytes"
            )));
        }
        Ok(())
    }

    fn ensure_batch_working_set_size(
        &self,
        reference_bytes: u64,
        candidate_bytes: u64,
        partial_bytes: u64,
    ) -> Result<()> {
        let bytes = batch_working_set_bytes(reference_bytes, candidate_bytes, partial_bytes)
            .ok_or_else(|| Error::unsupported("gpu_mse_rgba8 batch working set overflows u64"))?;
        if bytes > MAX_GPU_BATCH_WORKING_SET_BYTES {
            return Err(Error::unsupported(format!(
                "gpu_mse_rgba8 batch requires an estimated {bytes} transient bytes, exceeding the safety cap of {MAX_GPU_BATCH_WORKING_SET_BYTES} bytes"
            )));
        }
        Ok(())
    }

    fn dispatch_dimensions(&self, operation: &str, invocations: u64) -> Result<(u32, u32, u32)> {
        let groups = invocations.div_ceil(u64::from(WORKGROUP_SIZE)).max(1);
        if groups > u64::from(u32::MAX) {
            return Err(Error::unsupported(format!(
                "{operation} requires too many workgroups"
            )));
        }
        let max_dimension = u64::from(self.limits.max_compute_workgroups_per_dimension);
        if max_dimension == 0 {
            return Err(Error::Gpu(format!(
                "{operation} cannot run because the adapter reports a zero compute dispatch limit"
            )));
        }
        let groups_x = groups.min(max_dimension);
        let groups_y = groups.div_ceil(groups_x);
        if groups_y > max_dimension {
            return Err(Error::unsupported(format!(
                "{operation} requires {groups} workgroups, exceeding the adapter dispatch capacity of {}",
                max_dimension.saturating_mul(max_dimension)
            )));
        }
        Ok((groups as u32, groups_x as u32, groups_y as u32))
    }
}

fn capabilities_from_adapter(
    adapter: &wgpu::AdapterInfo,
    limits: &wgpu::Limits,
    options: GpuContextOptions,
) -> GpuCapabilities {
    let storage_limit = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size);
    let dispatch_groups = u64::from(limits.max_compute_workgroups_per_dimension)
        .saturating_mul(u64::from(limits.max_compute_workgroups_per_dimension));
    let dispatch_invocations = dispatch_groups.saturating_mul(u64::from(WORKGROUP_SIZE));
    let max_error_input_pixels = storage_limit / 4;
    let max_error_partial_pixels = (storage_limit / std::mem::size_of::<PartialStats>() as u64)
        .saturating_mul(u64::from(WORKGROUP_SIZE));
    let max_error_stats_pixels = max_error_input_pixels
        .min(max_error_partial_pixels)
        .min(dispatch_invocations)
        .min(u64::from(u32::MAX));
    let max_windowed_ssim_windows = (storage_limit / std::mem::size_of::<SsimWindowOut>() as u64)
        .min(dispatch_invocations)
        .min(u64::from(u32::MAX));
    GpuCapabilities {
        adapter_name: adapter.name.clone(),
        backend: format!("{:?}", adapter.backend),
        device_type: format!("{:?}", adapter.device_type),
        driver: adapter.driver.clone(),
        driver_info: adapter.driver_info.clone(),
        vendor_id: adapter.vendor,
        device_id: adapter.device,
        fallback_adapter: options.force_fallback_adapter,
        max_buffer_size: limits.max_buffer_size,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        max_rgba8_input_pixels: max_error_input_pixels.min(u64::from(u32::MAX)),
        max_error_stats_pixels,
        max_windowed_ssim_windows,
    }
}

/// Policy controlling transparent CPU fallback in [`GpuComparator`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum GpuFallbackPolicy {
    /// GPU initialization and execution failures are returned to the caller.
    Never,
    /// Fall back only when no GPU context can be initialized.
    InitializationOnly,
    /// Fall back on initialization, dispatch, or readback failure.
    #[default]
    AnyGpuError,
}

/// Options for the reusable high-level GPU comparator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuComparatorOptions {
    /// wgpu adapter-selection options.
    pub context: GpuContextOptions,
    /// CPU fallback policy.
    pub fallback: GpuFallbackPolicy,
}

/// Backend that produced a comparison result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum GpuExecution {
    /// The wgpu compute kernel produced the result.
    Gpu,
    /// The deterministic CPU implementation produced the result.
    CpuFallback,
}

/// Result of a high-level RGBA8 comparison.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuRgba8Comparison {
    /// One-pass error statistics in 8-bit code-value units.
    pub error_stats: GpuRgba8ErrorStats,
    /// Requested normalized metric projections.
    pub metrics: Vec<MetricOutput>,
    /// Backend that produced `error_stats`.
    pub execution: GpuExecution,
    /// GPU failure that caused CPU fallback, when applicable.
    pub fallback_reason: Option<String>,
}

/// Result of a windowed-luma-SSIM comparison with backend provenance.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuWindowedSsimComparison {
    /// Standard `wssim` metric output.
    pub metric: MetricOutput,
    /// Backend that produced the metric.
    pub execution: GpuExecution,
    /// GPU failure that caused CPU fallback, when applicable.
    pub fallback_reason: Option<String>,
}

/// Reusable comparison facade with optional deterministic CPU fallback.
pub struct GpuComparator {
    context: Option<GpuContext>,
    options: GpuComparatorOptions,
    initialization_error: Option<String>,
}

impl GpuComparator {
    /// Creates a comparator using default high-performance GPU selection and
    /// CPU fallback on any GPU failure.
    pub fn new() -> Result<Self> {
        Self::new_with_options(GpuComparatorOptions::default())
    }

    /// Creates a comparator with explicit adapter and fallback options.
    pub fn new_with_options(options: GpuComparatorOptions) -> Result<Self> {
        pollster::block_on(Self::new_async_with_options(options))
    }

    /// Creates a comparator asynchronously with default options.
    pub async fn new_async() -> Result<Self> {
        Self::new_async_with_options(GpuComparatorOptions::default()).await
    }

    /// Creates a comparator asynchronously with explicit adapter and fallback
    /// options.
    pub async fn new_async_with_options(options: GpuComparatorOptions) -> Result<Self> {
        match GpuContext::new_async_with_options(options.context).await {
            Ok(context) => Ok(Self {
                context: Some(context),
                options,
                initialization_error: None,
            }),
            Err(error) if options.fallback != GpuFallbackPolicy::Never => Ok(Self {
                context: None,
                options,
                initialization_error: Some(error.to_string()),
            }),
            Err(error) => Err(error),
        }
    }

    /// Returns the active GPU capabilities, or `None` when initialization fell
    /// back to CPU.
    pub fn capabilities(&self) -> Option<&GpuCapabilities> {
        self.context.as_ref().map(GpuContext::capabilities)
    }

    /// Returns the GPU initialization failure retained by a CPU-only comparator.
    pub fn initialization_error(&self) -> Option<&str> {
        self.initialization_error.as_deref()
    }

    /// Returns whether an initialized wgpu context is available.
    pub fn is_gpu_available(&self) -> bool {
        self.context.is_some()
    }

    /// Compares one RGBA8 frame pair and derives the requested error metrics
    /// from a single reduction pass.
    pub fn compare_rgba8(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
        metrics: &[GpuErrorMetric],
    ) -> Result<GpuRgba8Comparison> {
        self.compare_rgba8_in_domain(reference, distorted, metrics, GpuErrorDomain::All)
    }

    /// Compares one RGBA8 frame pair in either RGB color or all-RGBA domain.
    pub fn compare_rgba8_in_domain(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
        metrics: &[GpuErrorMetric],
        domain: GpuErrorDomain,
    ) -> Result<GpuRgba8Comparison> {
        validate_rgba8_pair(reference, distorted)?;
        let (error_stats, execution, fallback_reason) = match &self.context {
            Some(context) => match context.error_stats_rgba8(reference, distorted) {
                Ok(stats) => (stats, GpuExecution::Gpu, None),
                Err(error) if self.options.fallback == GpuFallbackPolicy::AnyGpuError => (
                    cpu_error_stats_rgba8(reference, distorted)?,
                    GpuExecution::CpuFallback,
                    Some(error.to_string()),
                ),
                Err(error) => return Err(error),
            },
            None => (
                cpu_error_stats_rgba8(reference, distorted)?,
                GpuExecution::CpuFallback,
                self.initialization_error.clone(),
            ),
        };
        let metrics = error_stats.normalized_metric_outputs_in_domain(metrics, domain);
        Ok(GpuRgba8Comparison {
            error_stats,
            metrics,
            execution,
            fallback_reason,
        })
    }

    /// Computes configurable windowed luma SSIM with the same CPU fallback
    /// policy as [`GpuComparator::compare_rgba8`].
    ///
    /// GPU and CPU results both use the standard `wssim` output name. Backend
    /// provenance remains explicit in [`GpuWindowedSsimComparison::execution`].
    pub fn compare_windowed_ssim_rgba8(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &FrameView<'_, Validated>,
        options: GpuWindowedSsimOptions,
    ) -> Result<GpuWindowedSsimComparison> {
        validate_rgba8_pair(reference, distorted)?;
        validate_windowed_ssim_options(options)?;
        let (mut metric, execution, fallback_reason) = match &self.context {
            Some(context) => {
                match context.windowed_ssim_rgba8_with_options(reference, distorted, options) {
                    Ok(metric) => (metric, GpuExecution::Gpu, None),
                    Err(error) if self.options.fallback == GpuFallbackPolicy::AnyGpuError => (
                        cpu_windowed_ssim_rgba8(reference, distorted, options)?,
                        GpuExecution::CpuFallback,
                        Some(error.to_string()),
                    ),
                    Err(error) => return Err(error),
                }
            }
            None => (
                cpu_windowed_ssim_rgba8(reference, distorted, options)?,
                GpuExecution::CpuFallback,
                self.initialization_error.clone(),
            ),
        };
        metric.name = "wssim".to_string();
        Ok(GpuWindowedSsimComparison {
            metric,
            execution,
            fallback_reason,
        })
    }

    /// Compares one reference against many candidates using the GPU's
    /// single-submit batch path when available.
    pub fn compare_many_rgba8(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &[&FrameView<'_, Validated>],
        metrics: &[GpuErrorMetric],
    ) -> Result<Vec<GpuRgba8Comparison>> {
        self.compare_many_rgba8_in_domain(reference, distorted, metrics, GpuErrorDomain::All)
    }

    /// Compares one reference against many candidates in either RGB color or
    /// all-RGBA domain.
    pub fn compare_many_rgba8_in_domain(
        &self,
        reference: &FrameView<'_, Validated>,
        distorted: &[&FrameView<'_, Validated>],
        metrics: &[GpuErrorMetric],
        domain: GpuErrorDomain,
    ) -> Result<Vec<GpuRgba8Comparison>> {
        if distorted.is_empty() {
            return Ok(Vec::new());
        }
        for candidate in distorted {
            validate_rgba8_pair(reference, candidate)?;
        }
        let (stats, execution, fallback_reason) = match &self.context {
            Some(context) => match gpu_error_stats_many_chunked(context, reference, distorted) {
                Ok(stats) => (stats, GpuExecution::Gpu, None),
                Err(error) if self.options.fallback == GpuFallbackPolicy::AnyGpuError => (
                    distorted
                        .iter()
                        .map(|candidate| cpu_error_stats_rgba8(reference, candidate))
                        .collect::<Result<Vec<_>>>()?,
                    GpuExecution::CpuFallback,
                    Some(error.to_string()),
                ),
                Err(error) => return Err(error),
            },
            None => (
                distorted
                    .iter()
                    .map(|candidate| cpu_error_stats_rgba8(reference, candidate))
                    .collect::<Result<Vec<_>>>()?,
                GpuExecution::CpuFallback,
                self.initialization_error.clone(),
            ),
        };
        Ok(stats
            .into_iter()
            .map(|error_stats| GpuRgba8Comparison {
                metrics: error_stats.normalized_metric_outputs_in_domain(metrics, domain),
                error_stats,
                execution,
                fallback_reason: fallback_reason.clone(),
            })
            .collect())
    }

    /// Compares multiple RGBA8 pairs in order while reusing the initialized
    /// device and cached compute pipelines.
    pub fn compare_batch_rgba8(
        &self,
        pairs: &[(&FrameView<'_, Validated>, &FrameView<'_, Validated>)],
        metrics: &[GpuErrorMetric],
    ) -> Result<Vec<GpuRgba8Comparison>> {
        pairs
            .iter()
            .enumerate()
            .map(|(index, (reference, distorted))| {
                self.compare_rgba8(reference, distorted, metrics)
                    .map_err(|error| {
                        Error::Gpu(format!("RGBA8 batch pair {index} failed: {error}"))
                    })
            })
            .collect()
    }
}

fn validate_rgba8_pair(
    reference: &FrameView<'_, Validated>,
    distorted: &FrameView<'_, Validated>,
) -> Result<()> {
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
            "GPU RGBA8 comparison requires RGBA8 inputs",
        ));
    }
    Ok(())
}

fn validate_windowed_ssim_options(options: GpuWindowedSsimOptions) -> Result<()> {
    if options.window_width == 0 || options.window_height == 0 {
        return Err(Error::unsupported(
            "GPU SSIM window dimensions must be non-zero",
        ));
    }
    if options.stride_x == 0 || options.stride_y == 0 {
        return Err(Error::unsupported(
            "GPU SSIM window strides must be non-zero",
        ));
    }
    Ok(())
}

fn ensure_gpu_wssim_workload(
    window_width: usize,
    window_height: usize,
    window_count: usize,
) -> Result<()> {
    let samples_per_window = (window_width as u64)
        .checked_mul(window_height as u64)
        .ok_or_else(|| Error::unsupported("gpu_wssim_rgba8 window area overflows u64"))?;
    if samples_per_window > MAX_GPU_WSSIM_SAMPLES_PER_WINDOW {
        return Err(Error::unsupported(format!(
            "gpu_wssim_rgba8 window contains {samples_per_window} samples, exceeding the per-invocation safety cap of {MAX_GPU_WSSIM_SAMPLES_PER_WINDOW}"
        )));
    }
    let total_sample_visits = samples_per_window
        .checked_mul(window_count as u64)
        .ok_or_else(|| Error::unsupported("gpu_wssim_rgba8 total work overflows u64"))?;
    if total_sample_visits > MAX_GPU_WSSIM_TOTAL_SAMPLE_VISITS {
        return Err(Error::unsupported(format!(
            "gpu_wssim_rgba8 layout requires {total_sample_visits} sample visits, exceeding the GPU work safety cap of {MAX_GPU_WSSIM_TOTAL_SAMPLE_VISITS}"
        )));
    }
    Ok(())
}

fn batch_working_set_bytes(
    reference_bytes: u64,
    candidate_bytes: u64,
    partial_bytes: u64,
) -> Option<u64> {
    let reference_upload = reference_bytes.checked_mul(2)?;
    let candidate_upload = reference_bytes.checked_add(candidate_bytes.checked_mul(2)?)?;
    let execution = reference_bytes
        .checked_add(candidate_bytes)?
        .checked_add(partial_bytes.checked_mul(2)?)?;
    Some(reference_upload.max(candidate_upload).max(execution))
}

fn gpu_error_stats_many_chunked(
    context: &GpuContext,
    reference: &FrameView<'_, Validated>,
    distorted: &[&FrameView<'_, Validated>],
) -> Result<Vec<GpuRgba8ErrorStats>> {
    let pixels = reference.dimensions().pixels()? as u64;
    let batch_size = context.max_error_stats_candidates(pixels) as usize;
    if batch_size == 0 {
        return Err(Error::unsupported(format!(
            "gpu_mse_rgba8 cannot fit a {pixels}-pixel candidate on this adapter"
        )));
    }
    let mut results = Vec::with_capacity(distorted.len());
    for candidates in distorted.chunks(batch_size) {
        results.extend(context.error_stats_many_rgba8(reference, candidates)?);
    }
    Ok(results)
}

fn cpu_error_stats_rgba8(
    reference: &FrameView<'_, Validated>,
    distorted: &FrameView<'_, Validated>,
) -> Result<GpuRgba8ErrorStats> {
    validate_rgba8_pair(reference, distorted)?;
    let reference = rgba8_bytes(reference)?;
    let distorted = rgba8_bytes(distorted)?;
    let mut sum_sq = [0.0f64; 4];
    let mut sum_abs = [0.0f64; 4];
    let mut max_abs = [0.0f64; 4];
    for (reference_pixel, distorted_pixel) in
        reference.chunks_exact(4).zip(distorted.chunks_exact(4))
    {
        for channel in 0..4 {
            let delta = f64::from(reference_pixel[channel].abs_diff(distorted_pixel[channel]));
            sum_sq[channel] += delta * delta;
            sum_abs[channel] += delta;
            max_abs[channel] = max_abs[channel].max(delta);
        }
    }
    let pixels = (reference.len() / 4) as u64;
    let pixel_count = pixels as f64;
    let channel_mse = sum_sq.map(|sum| sum / pixel_count);
    let channel_mae = sum_abs.map(|sum| sum / pixel_count);
    let mse = sum_sq.iter().sum::<f64>() / (pixel_count * 4.0);
    Ok(GpuRgba8ErrorStats {
        mse,
        channel_mse,
        rmse: mse.sqrt(),
        mae: sum_abs.iter().sum::<f64>() / (pixel_count * 4.0),
        channel_mae,
        max_abs: max_abs.into_iter().fold(0.0, f64::max),
        channel_max_abs: max_abs,
        pixels,
    })
}

fn cpu_windowed_ssim_rgba8(
    reference: &FrameView<'_, Validated>,
    distorted: &FrameView<'_, Validated>,
    options: GpuWindowedSsimOptions,
) -> Result<MetricOutput> {
    validate_rgba8_pair(reference, distorted)?;
    validate_windowed_ssim_options(options)?;
    WindowedSsim::with_sliding_window(
        options.window_width,
        options.window_height,
        options.stride_x,
        options.stride_y,
    )?
    .compare(reference, distorted)
}

fn error_stats_from_partials(partials: &[PartialStats], pixels: u64) -> GpuRgba8ErrorStats {
    let mut sum_sq = [0.0f64; 4];
    let mut sum_abs = [0.0f64; 4];
    let mut max_abs = [0.0f64; 4];
    for partial in partials {
        for channel in 0..4 {
            sum_sq[channel] += f64::from(partial.sum_sq[channel]);
            sum_abs[channel] += f64::from(partial.sum_abs[channel]);
            max_abs[channel] = max_abs[channel].max(f64::from(partial.max_abs[channel]));
        }
    }
    let pixel_count = pixels as f64;
    let channel_mse = sum_sq.map(|sum| sum / pixel_count);
    let channel_mae = sum_abs.map(|sum| sum / pixel_count);
    let mse = sum_sq.iter().sum::<f64>() / (pixel_count * 4.0);
    GpuRgba8ErrorStats {
        mse,
        channel_mse,
        rmse: mse.sqrt(),
        mae: sum_abs.iter().sum::<f64>() / (pixel_count * 4.0),
        channel_mae,
        max_abs: max_abs.into_iter().fold(0.0, f64::max),
        channel_max_abs: max_abs,
        pixels,
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

fn gpu_luma_weights(frame: &FrameView<'_, Validated>) -> [f32; 4] {
    let (red, green, blue) = crate::color::luma_weights(frame.format().color_space);
    [red as f32, green as f32, blue as f32, 0.0]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{ColorSpace, Dimensions, FormatSpec, FrameOwned, OwnedPlane};
    use crate::metrics::{MetricSet, MetricSpec, SampleDomain};

    fn known_pair() -> (FrameOwned, FrameOwned) {
        let reference = FrameOwned::packed_tight(
            vec![0, 10, 20, 30, 255, 250, 240, 230],
            2,
            1,
            PixelFormat::Rgba8,
        )
        .unwrap();
        let distorted = FrameOwned::packed_tight(
            vec![1, 8, 23, 26, 250, 255, 230, 240],
            2,
            1,
            PixelFormat::Rgba8,
        )
        .unwrap();
        (reference, distorted)
    }

    #[test]
    fn gpu_workload_safety_limits_reject_adversarial_layouts() {
        let options = GpuWindowedSsimOptions {
            window_width: 0,
            ..GpuWindowedSsimOptions::default()
        };
        assert!(validate_windowed_ssim_options(options).is_err());
        let options = GpuWindowedSsimOptions {
            stride_y: 0,
            ..GpuWindowedSsimOptions::default()
        };
        assert!(validate_windowed_ssim_options(options).is_err());

        assert!(ensure_gpu_wssim_workload(1024, 1024, 1).is_ok());
        let per_invocation = ensure_gpu_wssim_workload(1025, 1024, 1)
            .unwrap_err()
            .to_string();
        assert!(per_invocation.contains("per-invocation safety cap"));
        let excessive_windows =
            usize::try_from(MAX_GPU_WSSIM_TOTAL_SAMPLE_VISITS / 64 + 1).unwrap();
        let aggregate = ensure_gpu_wssim_workload(8, 8, excessive_windows)
            .unwrap_err()
            .to_string();
        assert!(aggregate.contains("GPU work safety cap"));

        assert_eq!(batch_working_set_bytes(16, 32, 48), Some(144));
        assert!(batch_working_set_bytes(u64::MAX, 1, 1).is_none());
        let over_cap =
            batch_working_set_bytes(4 * 1024 * 1024, 255 * 1024 * 1024, 4 * 1024 * 1024).unwrap();
        assert!(over_cap > MAX_GPU_BATCH_WORKING_SET_BYTES);
    }

    #[test]
    fn cpu_fallback_stats_are_exact_and_multi_metric_projection_is_ordered() {
        let (reference, distorted) = known_pair();
        let stats = cpu_error_stats_rgba8(&reference.as_view(), &distorted.as_view()).unwrap();
        assert_eq!(stats.pixels, 2);
        assert_eq!(stats.channel_mse, [13.0, 14.5, 54.5, 58.0]);
        assert_eq!(stats.mse, 35.0);
        assert_eq!(stats.rmse, 35.0f64.sqrt());
        assert_eq!(stats.mae, 5.0);
        assert_eq!(stats.channel_mae, [3.0, 3.5, 6.5, 7.0]);
        assert_eq!(stats.channel_max_abs, [5.0, 5.0, 10.0, 10.0]);
        assert_eq!(stats.max_abs, 10.0);

        let outputs = stats.normalized_metric_outputs(&[
            GpuErrorMetric::Psnr,
            GpuErrorMetric::Mae,
            GpuErrorMetric::Mse,
        ]);
        assert_eq!(
            outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>(),
            ["psnr:all", "mae:all", "mse:all"]
        );
        assert_eq!(outputs[0].details["samples"], 8.0);
        assert_eq!(outputs[0].details["channels"], 4.0);
        let color_outputs =
            stats.normalized_metric_outputs_in_domain(&GpuErrorMetric::ALL, GpuErrorDomain::Color);
        assert_eq!(color_outputs[0].name, "mse:color");
        assert_eq!(color_outputs[0].details["samples"], 6.0);
        assert_eq!(color_outputs[0].details["channels"], 3.0);
        assert!((color_outputs[0].score - (82.0 / 3.0) / (255.0 * 255.0)).abs() < 1e-12);

        let cpu_metrics = MetricSet::from_specs(
            &GpuErrorMetric::ALL
                .iter()
                .map(|metric| {
                    let name = match metric {
                        GpuErrorMetric::Mse => "mse",
                        GpuErrorMetric::Rmse => "rmse",
                        GpuErrorMetric::Psnr => "psnr",
                        GpuErrorMetric::Mae => "mae",
                        GpuErrorMetric::MaxAbsoluteError => "maxae",
                    };
                    MetricSpec::new(name, SampleDomain::All)
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .compare(&reference.as_view(), &distorted.as_view())
        .unwrap();
        let projected = stats.into_normalized_metric_outputs();
        for (cpu, projected) in cpu_metrics.iter().zip(&projected) {
            assert_eq!(cpu.name, projected.name);
            assert!((cpu.score - projected.score).abs() < 1e-12);
        }
        let cpu_color_metrics = MetricSet::from_specs(
            &GpuErrorMetric::ALL
                .iter()
                .map(|metric| {
                    let name = match metric {
                        GpuErrorMetric::Mse => "mse",
                        GpuErrorMetric::Rmse => "rmse",
                        GpuErrorMetric::Psnr => "psnr",
                        GpuErrorMetric::Mae => "mae",
                        GpuErrorMetric::MaxAbsoluteError => "maxae",
                    };
                    MetricSpec::new(name, SampleDomain::Color)
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .compare(&reference.as_view(), &distorted.as_view())
        .unwrap();
        for (cpu, projected) in cpu_color_metrics.iter().zip(&color_outputs) {
            assert_eq!(cpu.name, projected.name);
            assert!((cpu.score - projected.score).abs() < 1e-12);
        }

        let comparator = GpuComparator {
            context: None,
            options: GpuComparatorOptions::default(),
            initialization_error: Some("test adapter unavailable".to_string()),
        };
        let fallback = comparator
            .compare_rgba8(
                &reference.as_view(),
                &distorted.as_view(),
                &[GpuErrorMetric::Psnr],
            )
            .unwrap();
        assert_eq!(fallback.execution, GpuExecution::CpuFallback);
        assert_eq!(fallback.error_stats.mse, 35.0);
        assert_eq!(fallback.metrics[0].name, "psnr:all");
        assert_eq!(
            fallback.fallback_reason.as_deref(),
            Some("test adapter unavailable")
        );
        let fallback_wssim = comparator
            .compare_windowed_ssim_rgba8(
                &reference.as_view(),
                &distorted.as_view(),
                GpuWindowedSsimOptions::default(),
            )
            .unwrap();
        assert_eq!(fallback_wssim.execution, GpuExecution::CpuFallback);
        assert_eq!(fallback_wssim.metric.name, "wssim");
        assert_eq!(
            fallback_wssim.fallback_reason.as_deref(),
            Some("test adapter unavailable")
        );
        let invalid_options = GpuWindowedSsimOptions {
            stride_x: 0,
            ..GpuWindowedSsimOptions::default()
        };
        assert!(
            comparator
                .compare_windowed_ssim_rgba8(
                    &reference.as_view(),
                    &distorted.as_view(),
                    invalid_options,
                )
                .is_err()
        );
    }

    #[test]
    fn gpu_error_stats_match_cpu_for_strided_rgba8_and_batch_order() {
        let Ok(context) = GpuContext::new() else {
            return;
        };
        let width = 19usize;
        let height = 17usize;
        let stride = width * 4 + 12;
        let mut reference_bytes = vec![0u8; stride * height];
        let mut distorted_bytes = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                for channel in 0..4 {
                    let offset = y * stride + x * 4 + channel;
                    let value = ((x * 17 + y * 13 + channel * 47) % 256) as u8;
                    reference_bytes[offset] = value;
                    distorted_bytes[offset] = value.saturating_add(((x + y + channel) % 7) as u8);
                }
            }
        }
        let reference = FrameOwned::packed(
            reference_bytes,
            width as u32,
            height as u32,
            PixelFormat::Rgba8,
            stride,
        )
        .unwrap();
        let distorted = FrameOwned::packed(
            distorted_bytes,
            width as u32,
            height as u32,
            PixelFormat::Rgba8,
            stride,
        )
        .unwrap();
        let cpu = cpu_error_stats_rgba8(&reference.as_view(), &distorted.as_view()).unwrap();
        let gpu = context
            .error_stats_rgba8(&reference.as_view(), &distorted.as_view())
            .unwrap();
        assert_eq!(gpu, cpu);

        let reference_view = reference.as_view();
        let distorted_view = distorted.as_view();
        let many = context
            .error_stats_many_rgba8(
                &reference_view,
                &[&distorted_view, &reference_view, &distorted_view],
            )
            .unwrap();
        assert_eq!(many.len(), 3);
        assert_eq!(many[0], cpu);
        assert_eq!(many[1].mse, 0.0);
        assert_eq!(many[2], cpu);

        let (known_reference, known_distorted) = known_pair();
        let known_reference_view = known_reference.as_view();
        let known_distorted_view = known_distorted.as_view();
        let batch = context
            .error_stats_batch_rgba8(&[
                (&reference_view, &distorted_view),
                (&known_reference_view, &known_distorted_view),
            ])
            .unwrap();
        assert_eq!(batch[0], cpu);
        assert_eq!(batch[1].mse, 35.0);
    }

    #[test]
    fn gpu_capabilities_and_cached_windowed_ssim_pipeline_are_usable() {
        let Ok(context) = GpuContext::new() else {
            return;
        };
        let capabilities = context.capabilities();
        assert!(!capabilities.adapter_name.is_empty());
        assert!(capabilities.max_error_stats_pixels > 0);
        assert!(capabilities.max_windowed_ssim_windows > 0);
        let dimension = u64::from(capabilities.max_compute_workgroups_per_dimension);
        if dimension > 1 && dimension < u64::from(u32::MAX) {
            let (groups, groups_x, groups_y) = context
                .dispatch_dimensions("test", (dimension + 1) * u64::from(WORKGROUP_SIZE))
                .unwrap();
            assert_eq!(u64::from(groups), dimension + 1);
            assert_eq!(u64::from(groups_x), dimension);
            assert_eq!(groups_y, 2);
        }

        let pixels = (0..16 * 16)
            .flat_map(|pixel| {
                let value = ((pixel * 37) % 256) as u8;
                [value, value.wrapping_add(31), value.wrapping_add(79), 255]
            })
            .collect::<Vec<_>>();
        let frame = FrameOwned::packed_tight(pixels.clone(), 16, 16, PixelFormat::Rgba8).unwrap();
        for _ in 0..2 {
            let output = context
                .windowed_ssim_rgba8_with_options(
                    &frame.as_view(),
                    &frame.as_view(),
                    GpuWindowedSsimOptions::default(),
                )
                .unwrap();
            assert!(
                (output.score - 1.0).abs() < 1e-6,
                "unexpected identical-frame SSIM output: {output:?}"
            );
            assert_eq!(output.details["windows"], 4.0);
        }

        let mut distorted_pixels = pixels;
        for (index, value) in distorted_pixels.iter_mut().enumerate() {
            if index % 4 != 3 && index % 11 == 0 {
                *value = value.saturating_add(7);
            }
        }
        let distorted =
            FrameOwned::packed_tight(distorted_pixels, 16, 16, PixelFormat::Rgba8).unwrap();
        let gpu = context
            .windowed_ssim_rgba8_with_options(
                &frame.as_view(),
                &distorted.as_view(),
                GpuWindowedSsimOptions::default(),
            )
            .unwrap();
        let cpu = MetricSet::from_specs(&[MetricSpec::new("wssim", SampleDomain::Luma)])
            .unwrap()
            .compare(&frame.as_view(), &distorted.as_view())
            .unwrap();
        assert!(
            (gpu.score - cpu[0].score).abs() < 1e-4,
            "GPU/CPU WSSIM mismatch: gpu={gpu:?}, cpu={:?}",
            cpu[0]
        );

        let comparator = GpuComparator {
            context: Some(context),
            options: GpuComparatorOptions::default(),
            initialization_error: None,
        };
        let high_level = comparator
            .compare_windowed_ssim_rgba8(
                &frame.as_view(),
                &distorted.as_view(),
                GpuWindowedSsimOptions::default(),
            )
            .unwrap();
        assert_eq!(high_level.execution, GpuExecution::Gpu);
        assert_eq!(high_level.metric.name, "wssim");
        assert!((high_level.metric.score - cpu[0].score).abs() < 1e-4);

        let mut reference_format = FormatSpec::new(PixelFormat::Rgba8);
        reference_format.color_space = ColorSpace::Bt601;
        let mut distorted_format = FormatSpec::new(PixelFormat::Rgba8);
        distorted_format.color_space = ColorSpace::Bt2020;
        let dimensions = Dimensions::new(16, 16).unwrap();
        let reference_with_metadata = FrameOwned::new(
            dimensions,
            reference_format,
            vec![OwnedPlane::new(
                frame.owned_planes()[0].data.clone(),
                16 * 4,
            )],
        )
        .unwrap();
        let distorted_with_metadata = FrameOwned::new(
            dimensions,
            distorted_format,
            vec![OwnedPlane::new(
                distorted.owned_planes()[0].data.clone(),
                16 * 4,
            )],
        )
        .unwrap();
        let gpu_metadata = comparator
            .compare_windowed_ssim_rgba8(
                &reference_with_metadata.as_view(),
                &distorted_with_metadata.as_view(),
                GpuWindowedSsimOptions::default(),
            )
            .unwrap();
        let cpu_metadata = MetricSet::from_specs(&[MetricSpec::new("wssim", SampleDomain::Luma)])
            .unwrap()
            .compare(
                &reference_with_metadata.as_view(),
                &distorted_with_metadata.as_view(),
            )
            .unwrap();
        assert!(
            (gpu_metadata.metric.score - cpu_metadata[0].score).abs() < 1e-4,
            "GPU/CPU metadata-aware WSSIM mismatch: gpu={gpu_metadata:?}, cpu={:?}",
            cpu_metadata[0]
        );
    }
}
