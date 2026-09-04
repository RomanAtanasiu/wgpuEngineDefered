#include mesh_includes.wgsl

@group(0) @binding(0) var in_texture: texture_2d<f32>;//light buffer+post-processing texture
@group(1) @binding(0) var accumulation_buffer: texture_2d<f32>;
@group(1) @binding(1) var gbuffer_normal_velocity: texture_2d<f32>;
@group(1) @binding(2) var prev_velocity_buffer: texture_2d<f32>;



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
    let pixel = vec2<i32>(in.uv * vec2<f32>(screen_dims));
    let color_current = textureLoad(in_texture, pixel, 0);

//reprojection
    let velocityUV = textureLoad(gbuffer_normal_velocity, pixel, 0).ba;
    let prevUV = in.uv - velocityUV;
    let prevPixel = vec2<i32>(prevUV * vec2<f32>(screen_dims));
    let previousColor = textureLoad(accumulation_buffer, prevPixel, 0);

//velocity rejection
    let prev_velocityUV = textureLoad(prev_velocity_buffer, prevPixel, 0).ba;
    let velocityLength = length(prev_velocityUV - velocityUV);
    let velocityDisocclusion = saturate((velocityLength - 0.001) * 10.0);

//color clamping

    var minColor = vec3f(9999.0);
    var maxColor = vec3f(-9999.0);

    var blur_color = vec3f(0.0);
    for(var x = -1; x <= 1; x += 1)
    {
        for(var y = -1; y <= 1; y += 1)
        {
            //sample neighbor pixels for min/max clamping
            var color = textureLoad(in_texture, prevPixel + vec2<i32>(x, y), 0); // Sample neighbor
            minColor = min(minColor, color.rgb); // Take min and max
            maxColor = max(maxColor, color.rgb); 

            //average neighbor pixels for velocity rejection
            var actual_color = textureLoad(in_texture, pixel + vec2<i32>(x, y), 0); 
            blur_color += actual_color.rgb;  
        }
    }
    blur_color /= 9.0; 

    let previousColorClamped = clamp(previousColor.rgb, minColor, maxColor);

    //let color_previous = textureLoad(accumulation_buffer, prevPixel, 0);

    let accumulation = mix(previousColorClamped, color_current.rgb, 0.1);

    out.color = vec4f(accumulation*(1.0 - velocityDisocclusion) + blur_color * velocityDisocclusion, 1.0);

    //out.color = vec4f(accumulation, 1.0);

    return out;
}