#include mesh_includes.wgsl

@group(0) @binding(0) var gbuffer_albedo_metallic: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_normal_roughness: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_depth_buffer: texture_depth_2d;
@group(0) @binding(3) var sampler_2d : sampler;

#dynamic @group(1) @binding(0) var<uniform> camera_data : CameraData;

#include pbr_light.wgsl
#include pbr_functions.wgsl
#include pbr_material.wgsl
#define MAX_LIGHTS

@group(2) @binding(0) var irradiance_texture: texture_cube<f32>;
@group(2) @binding(1) var brdf_lut_texture: texture_2d<f32>;
@group(2) @binding(2) var sampler_clamp: sampler;
@group(2) @binding(3) var<uniform> lights : array<Light, MAX_LIGHTS>;
@group(2) @binding(4) var<uniform> num_lights : u32;
@group(2) @binding(5) var lights_shadow_maps: texture_depth_2d_array;
@group(2) @binding(6) var shadow_sampler: sampler_comparison;

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

    out.color = vec4f(textureSample(gbuffer_albedo_metallic, sampler_2d, in.uv).xyz, 1.0);
    let i = textureSample(gbuffer_normal_roughness, sampler_2d, vec2f(0.0));
    let j = textureLoad(gbuffer_depth_buffer, vec2i(0), 0);

    return out;
}