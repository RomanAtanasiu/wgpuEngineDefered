#include tonemappers.wgsl


@group(0) @binding(0) var in_texture: texture_2d<f32>;
struct VertexInput {
    @builtin(instance_index) instance_id : u32,
    @location(0) position: vec3f,
#unique vertex @location(1) uv: vec2f,
#unique vertex @location(2) normal: vec3f,
#unique vertex @location(3) tangent: vec4f,
#unique vertex @location(4) color: vec3f,
#unique vertex @location(5) weights: vec4f,
#unique vertex @location(6) joints: vec4i
};

struct DefferedVertexOut {
    @builtin(position)  position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(in: VertexInput) -> DefferedVertexOut {
    var out: DefferedVertexOut;
    out.position = vec4f(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

@fragment
fn fs_main(in: DefferedVertexOut, @builtin(front_facing) is_front_facing: bool) -> FragmentOutput {
    var out: FragmentOutput;
    let screen_dims = textureDimensions(in_texture);
    let color = textureLoad(in_texture, vec2<i32>(in.uv * vec2<f32>(screen_dims)), 0);
    var color_tonemapped: vec3f = tonemap_khronos_pbr_neutral(color.rgb);

    let gamma = 2.2f;
    out.color = vec4f(pow(color_tonemapped.rgb, vec3f(1.0f / gamma)), color.a);
    return out;
}