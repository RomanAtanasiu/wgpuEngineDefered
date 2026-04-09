

@group(0) @binding(0) var in_texture: texture_2d<f32>;
//check format
@group(0) @binding(1) var outTexture: texture_storage_2d<rgba32float,write>;
//@group(1) @binding(0) var<uniform> blur_level: i32;

//https://www.shadertoy.com/view/XdcXzn
fn contrast_matrix(contrast : f32) -> mat4x4<f32>
 {
    let t = (1.0 - contrast) / 2.0;
    return mat4x4<f32>( 
        contrast, 0, 0, 0,
        0, contrast, 0, 0,
        0, 0, contrast, 0,
        t, t, t, 1);

 }


@compute @workgroup_size(8, 8)
fn contrast_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let contrast: i32 = 5;
    let size = textureDimensions(in_texture);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let color = textureLoad(in_texture, vec2<i32>(id.xy)), 0);
    textureStore(outTexture, vec2<i32>(id.xy), color*contrast_matrix(contrast));
}