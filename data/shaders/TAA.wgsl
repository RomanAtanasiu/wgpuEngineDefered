#include mesh_includes.wgsl

@group(0) @binding(0) var in_texture: texture_2d<f32>;
@group(1) @binding(0) var accumulation_buffer: texture_2d<f32>;
@group(1) @binding(1) var gbuffer_normal_velocity: texture_2d<f32>;




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

    let velocityUV = textureLoad(gbuffer_normal_velocity, pixel, 0).ba;
    let prevUV = in.uv + velocityUV;
    let prevPixel = vec2<i32>(prevUV * vec2<f32>(screen_dims));
    let previousColor = textureLoad(accumulation_buffer, prevPixel, 0);

    var minColor = vec3f(9999.0);
    var maxColor = vec3f(-9999.0);

    for(var x = -1; x <= 1; x += 1)
    {
        for(var y = -1; y <= 1; y += 1)
        {
            var color = textureLoad(in_texture, prevPixel + vec2<i32>(x, y), 0); // Sample neighbor
            minColor = min(minColor, color.rgb); // Take min and max
            maxColor = max(maxColor, color.rgb);
        }
    }
    let previousColorClamped = clamp(previousColor.rgb, minColor, maxColor);

    //let color_previous = textureLoad(accumulation_buffer, prevPixel, 0);

    out.color = vec4f(mix(previousColorClamped, color_current.rgb, 0.1), 1.0);

    return out;
}