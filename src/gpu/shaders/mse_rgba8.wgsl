struct Params {
    pixel_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> reference_pixels: array<u32>;
@group(0) @binding(1) var<storage, read> distorted_pixels: array<u32>;
@group(0) @binding(2) var<storage, read_write> partial_sums: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> local_sums: array<vec4<f32>, 256>;

fn rgba8_to_f32(px: u32) -> vec4<f32> {
    let r = f32(px & 0xffu);
    let g = f32((px >> 8u) & 0xffu);
    let b = f32((px >> 16u) & 0xffu);
    let a = f32((px >> 24u) & 0xffu);
    return vec4<f32>(r, g, b, a);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let pixel_index = global_id.x;
    let lane = local_id.x;

    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (pixel_index < params.pixel_count) {
        let a = rgba8_to_f32(reference_pixels[pixel_index]);
        let b = rgba8_to_f32(distorted_pixels[pixel_index]);
        let d = a - b;
        sum = d * d;
    }

    local_sums[lane] = sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (lane < stride) {
            local_sums[lane] = local_sums[lane] + local_sums[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        partial_sums[workgroup_id.x] = local_sums[0];
    }
}
