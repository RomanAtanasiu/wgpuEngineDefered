#include mesh_includes.wgsl
#include tonemappers.wgsl


fn decode(f_in: vec2<f32>) -> vec3<f32> {
    var f = f_in * 2.0 - vec2<f32>(1.0);

    var n = vec3<f32>(
        f.x,
        f.y,
        1.0 - abs(f.x) - abs(f.y)
    );

    let t = clamp(-n.z, 0.0, 1.0);

    let adjust = select(vec2<f32>(t), vec2<f32>(-t), n.xy >= vec2<f32>(0.0));
    //n.xy += adjust;

    n.x += adjust.x;
    n.y += adjust.y;
    return normalize(n);
}

@group(0) @binding(0) var gbuffer_albedo_metallic_roughness: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_normal_velocity: texture_2d<f32>;
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
    let albedo_metallic_roughness = textureSample(gbuffer_albedo_metallic_roughness, sampler_2d, in.uv);
    let normal_velocity = textureSample(gbuffer_normal_velocity, sampler_2d, in.uv);
    let metallic_roughness: vec2f = unpack2x16float(u32(albedo_metallic_roughness.a));
//todo
//camera_data.screen size does not work, since it is webgpu_context->screen_width
//and gbuffer_depth_buffer is webgpu_context->render_width
    let screen_dims = textureDimensions(gbuffer_depth_buffer);

    //let uv = vec2f(1.0/screen_dims.x,1/.0/screen_dims.y/1.0);

    var depth = f32(textureLoad(gbuffer_depth_buffer, vec2<i32>(in.uv * vec2<f32>(screen_dims)), 0));


    var uv_clip = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - 2.0 * in.uv.y);
    var clip_coords = vec4f(
        uv_clip.x, 
        uv_clip.y,
        depth,
        1.0
    );



    var not_norm_world_pos = camera_data.inv_view_projection * clip_coords;
    var world_pos = not_norm_world_pos.xyz / not_norm_world_pos.w;

    var m : PbrMaterial;

    m.pos = world_pos;
    m.albedo = albedo_metallic_roughness.rgb;
    m.metallic = metallic_roughness.r;
    let normal = decode(normal_velocity.rg);
    
    m.normal = normal;
    m.roughness = metallic_roughness.g;


    m.view_dir = normalize(camera_data.eye - m.pos);
    m.n_dot_v = clamp(dot(m.normal, m.view_dir), 0.0, 1.0);
    m.reflected_dir = normalize(reflect(-m.view_dir, m.normal));


    m.roughness = max(m.roughness, 0.04);
    m.diffuse = mix(m.albedo, vec3f(0.0), m.metallic);
    m.f0 = mix(vec3f(0.04), m.albedo, m.metallic);
    m.f0_dielectric = vec3f(0.04);
    m.clearcoat_fresnel = vec3f(0.0);
    m.emissive = vec3f(0.0);
    m.ao = 1.0;
    m.clearcoat_fresnel = vec3f(0.0);
    m.f90 = vec3f(1.0);
    m.ior = 1.5; // default IOR for most materials
    m.specular_weight = 1.0;

    if(camera_data.show_depthbuffer == 0 && camera_data.show_gbuffers == 0){
        var final_color : vec3f = vec3f(0.0);//vec3f(m.albedo);

        final_color = get_indirect_light(&m);
        //var final_color2 = get_direct_light(&m);
        final_color += get_direct_light(&m);
        final_color *= camera_data.exposure;
        final_color = tonemap_khronos_pbr_neutral(final_color);
        out.color = vec4f(final_color, 1.0);
    }
    else if (camera_data.show_gbuffers == 1){
        out.color = vec4f(albedo_metallic_roughness.rgb,1.0);
    }
    else if (camera_data.show_gbuffers == 2){
        out.color = vec4f(normal_velocity);
    } else{
        out.color = vec4f(vec3f(depth),1.0);
    }

    return out;


    out.color = vec4f(textureSample(gbuffer_albedo_metallic_roughness, sampler_2d, in.uv).xyz, 1.0);
    let i = textureSample(gbuffer_normal_velocity, sampler_2d, vec2f(0.0));
    let j = textureLoad(gbuffer_depth_buffer, vec2i(0), 0);

    return out;

}