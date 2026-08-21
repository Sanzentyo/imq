struct Params {
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
    reference_luma_weights: vec4<f32>,
    distorted_luma_weights: vec4<f32>,
};

struct WindowOut {
    values: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> reference_pixels: array<u32>;
@group(0) @binding(1) var<storage, read> distorted_pixels: array<u32>;
@group(0) @binding(2) var<storage, read_write> window_out: array<WindowOut>;
@group(0) @binding(3) var<uniform> params: Params;

fn luma_from_rgba8(pixel: u32, weights: vec4<f32>) -> f32 {
    let r = f32(pixel & 0xffu) / 255.0;
    let g = f32((pixel >> 8u) & 0xffu) / 255.0;
    let b = f32((pixel >> 16u) & 0xffu) / 255.0;
    return weights.x * r + weights.y * g + weights.z * b;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let linear_workgroup_id = workgroup_id.x + workgroup_id.y * params.groups_x;
    if (linear_workgroup_id >= params.group_count) {
        return;
    }
    let index = linear_workgroup_id * 256u + local_id.x;
    if (index >= params.window_count) {
        return;
    }

    let wx = index % params.windows_x;
    let wy = index / params.windows_x;
    let max_x = params.width - params.window_width;
    let max_y = params.height - params.window_height;
    let x0 = min(wx * params.stride_x, max_x);
    let y0 = min(wy * params.stride_y, max_y);

    var sum_x = 0.0;
    var sum_y = 0.0;
    var sum_x2 = 0.0;
    var sum_y2 = 0.0;
    var sum_xy = 0.0;
    var sum_abs_diff = 0.0;

    var yy = 0u;
    loop {
        if (yy >= params.window_height) { break; }
        var xx = 0u;
        loop {
            if (xx >= params.window_width) { break; }
            let pixel_index = (y0 + yy) * params.width + (x0 + xx);
            let xr = luma_from_rgba8(
                reference_pixels[pixel_index],
                params.reference_luma_weights,
            );
            let yd = luma_from_rgba8(
                distorted_pixels[pixel_index],
                params.distorted_luma_weights,
            );
            sum_x = sum_x + xr;
            sum_y = sum_y + yd;
            sum_x2 = sum_x2 + xr * xr;
            sum_y2 = sum_y2 + yd * yd;
            sum_xy = sum_xy + xr * yd;
            sum_abs_diff = sum_abs_diff + abs(xr - yd);
            xx = xx + 1u;
        }
        yy = yy + 1u;
    }

    let n = f32(params.window_width * params.window_height);
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let denom = max(n - 1.0, 1.0);
    let var_x = max(sum_x2 - (sum_x * sum_x / n), 0.0) / denom;
    let var_y = max(sum_y2 - (sum_y * sum_y / n), 0.0) / denom;
    let cov_xy = (sum_xy - (sum_x * sum_y / n)) / denom;
    let c1 = 0.0001;
    let c2 = 0.0009;
    let luminance = (2.0 * mean_x * mean_y + c1) / (mean_x * mean_x + mean_y * mean_y + c1);
    let contrast_structure = (2.0 * cov_xy + c2) / (var_x + var_y + c2);
    let ssim = luminance * contrast_structure;
    if (sum_abs_diff == 0.0) {
        window_out[index].values = vec4<f32>(1.0, 1.0, n, 0.0);
    } else {
        window_out[index].values = vec4<f32>(ssim, contrast_structure, n, 0.0);
    }
}
