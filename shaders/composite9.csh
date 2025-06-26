#version 460 compatibility

#include "/textile/CSRGBA32UI.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32ui) uniform restrict uimage2D uimg_csrgba32ui;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        uvec4 packedData = texelFetch(usam_tempRGBA32UI, texelPos, 0);
        imageStore(uimg_csrgba32ui, clouds_ss_history_texelToTexel(texelPos), packedData);
    }
}