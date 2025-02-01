#version 460 compatibility
#include "../_Util.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32ui) uniform restrict uimage2D uimg_gbufferData;
layout(rgba8) uniform restrict image2D uimg_temp5;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        GBufferData gData;
        gbuffer_unpack(imageLoad(uimg_gbufferData, texelPos), gData);
        gData.albedo = imageLoad(uimg_temp5, texelPos).rgb;
        uvec4 packedData;
        gbuffer_pack(packedData, gData);
        imageStore(uimg_gbufferData, texelPos, packedData);
    }
}