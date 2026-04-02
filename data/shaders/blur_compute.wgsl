

@group(0) @binding(0) var in_texture: texture_2d<f32>;
//check format
@group(0) @binding(1) var outTexture: texture_storage_2d<rgba8unorm,write>;
@group(1) @binding(0) var blur_level:  

@compute @workgroup_size(8, 8)
fn computeMipMap(@builtin(global_invocation_id) id: vec3<u32>) {
    let offset = vec2<u32>(0u, 1u);
    let color = (
        textureLoad(previousMipLevel, 2u * id.xy + offset.xx, 0) +
        textureLoad(previousMipLevel, 2u * id.xy + offset.xy, 0) +
        textureLoad(previousMipLevel, 2u * id.xy + offset.yx, 0) +
        textureLoad(previousMipLevel, 2u * id.xy + offset.yy, 0)
    ) * 0.25;
    textureStore(nextMipLevel, id.xy, color);
}