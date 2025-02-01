#version 460 compatibility

#include "_Util.glsl"
#include "atmosphere/Common.glsl"
#include "general/Lighting.glsl"
#include "general/NDPacking.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform readonly image2D uimg_temp1;
layout(r32f) uniform readonly image2D uimg_gbufferViewZ;
layout(rg32ui) uniform writeonly uimage2D uimg_lastNZ;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 screenCoord = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        float viewZ = imageLoad(uimg_gbufferViewZ, texelPos).r;
        vec3 viewCoord = coords_toViewCoord(screenCoord, viewZ, gbufferProjectionInverse);
        vec3 viewNormal = imageLoad(uimg_temp1, texelPos).rgb;

        imageStore(uimg_lastNZ, texelPos, uvec4(ndpacking_pack(viewNormal, viewZ), 0u, 0u));
    }
}