struct Params {
    pixel_count: u32,
    groups_x: u32,
    _pad1: u32,
    _pad2: u32,
};

struct PartialStats {
    sum_sq: vec4<f32>,
    sum_abs: vec4<f32>,
    max_abs: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> reference_pixels: array<u32>;
@group(0) @binding(1) var<storage, read> distorted_pixels: array<u32>;
@group(0) @binding(2) var<storage, read_write> partial_stats: array<PartialStats>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> local_sum_sq: array<vec4<f32>, 256>;
var<workgroup> local_sum_abs: array<vec4<f32>, 256>;
var<workgroup> local_max_abs: array<vec4<f32>, 256>;

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
    let linear_workgroup_id = workgroup_id.x + workgroup_id.y * params.groups_x;
    let pixel_index = global_id.x + global_id.y * params.groups_x * 256u;
    let lane = local_id.x;

    var sum_sq = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var sum_abs = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var max_abs = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (pixel_index < params.pixel_count) {
        let a = rgba8_to_f32(reference_pixels[pixel_index]);
        let b = rgba8_to_f32(distorted_pixels[pixel_index]);
        let ad = abs(a - b);
        sum_sq = ad * ad;
        sum_abs = ad;
        max_abs = ad;
    }

    local_sum_sq[lane] = sum_sq;
    local_sum_abs[lane] = sum_abs;
    local_max_abs[lane] = max_abs;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (lane < stride) {
            local_sum_sq[lane] = local_sum_sq[lane] + local_sum_sq[lane + stride];
            local_sum_abs[lane] = local_sum_abs[lane] + local_sum_abs[lane + stride];
            local_max_abs[lane] = max(local_max_abs[lane], local_max_abs[lane + stride]);
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        partial_stats[linear_workgroup_id].sum_sq = local_sum_sq[0];
        partial_stats[linear_workgroup_id].sum_abs = local_sum_abs[0];
        partial_stats[linear_workgroup_id].max_abs = local_max_abs[0];
    }
}
