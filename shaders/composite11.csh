#version 460 compatibility

#include "_Util.glsl"
#include "atmosphere/Common.glsl"
#include "general/Lighting.glsl"
#include "general/NDPacking.glsl"
#include "svgf/Update.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform readonly image2D uimg_temp1;
layout(r32f) uniform readonly image2D uimg_gbufferViewZ;
layout(rg32ui) uniform writeonly uimage2D uimg_prevNZ;

layout(rgba16f) uniform readonly image2D uimg_ssvbil;
layout(rgba16f) uniform readonly image2D uimg_temp4;

layout(rgba16f) uniform restrict image2D uimg_temp3;

layout(rgba16f) uniform writeonly image2D uimg_svgfHistoryColor;
layout(rg16f) uniform writeonly image2D uimg_svgfHistoryMoments;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 screenCoord = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        float viewZ = imageLoad(uimg_gbufferViewZ, texelPos).r;
        vec3 viewCoord = coords_toViewCoord(screenCoord, viewZ, gbufferProjectionInverse);
        vec3 viewNormal = imageLoad(uimg_temp1, texelPos).rgb;

        imageStore(uimg_prevNZ, texelPos, uvec4(ndpacking_pack(viewNormal, viewZ), 0u, 0u));

        vec3 currColor = imageLoad(uimg_ssvbil, texelPos).rgb;
        vec2 prevMoments = imageLoad(uimg_temp3, texelPos).xy;
        vec4 prevColorHLen = imageLoad(uimg_temp4, texelPos);

        vec4 newColorHLen;
        vec2 newMoments;
        vec4 filterInput;
        svgf_update(currColor, prevColorHLen, prevMoments, newColorHLen, newMoments, filterInput);

        imageStore(uimg_svgfHistoryColor, texelPos, newColorHLen);
        imageStore(uimg_svgfHistoryMoments, texelPos, vec4(newMoments, 0.0, 0.0));
        imageStore(uimg_temp3, texelPos, filterInput);
    }
}