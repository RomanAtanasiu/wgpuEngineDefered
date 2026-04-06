

@group(0) @binding(0) var in_texture: texture_2d<f32>;
//check format
@group(0) @binding(1) var outTexture: texture_storage_2d<rgba8unorm,write>;
@group(1) @binding(0) var<uniform> blur_level: i32;

@compute @workgroup_size(8, 8)
fn computeMipMap(@builtin(global_invocation_id) id: vec3<u32>) {

    let size = textureDimensions(in_texture);

    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    var count: f32 = 0.0;
    var total_color: vec4<f32> = vec4<f32>(0.0);

    for (var i: i32 = -blur_level; i <= blur_level; i = i + 1) {
        for (var j: i32 = -blur_level; j <= blur_level; j = j + 1) {

            let actual_x = clamp(i32(id.x) + i, 0, i32(size.x) - 1);
            let actual_y = clamp(i32(id.y) + j, 0, i32(size.y) - 1);

            total_color += textureLoad(in_texture, vec2<i32>(actual_x, actual_y), 0);
            count += 1.0;
        }
    }

    total_color = total_color / count;

    textureStore(outTexture, vec2<i32>(id.xy), total_color);
}