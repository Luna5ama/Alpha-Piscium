#version 460 compatibility

#include "_Util.glsl"
#include "atmosphere/Common.glsl"
#include "general/Lighting.glsl"
#include "general/NDPacking.glsl"
#include "svgf/Update.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp1;
uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_ssvbil;

uniform sampler2D usam_temp3;
layout(rgba16f) uniform restrict image2D uimg_temp4;

layout(rg32ui) uniform writeonly uimage2D uimg_prevNZ;
layout(rg16f) uniform writeonly image2D uimg_svgfHistoryMoments;
layout(rgba8) uniform writeonly image2D uimg_temp5;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 screenCoord = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        vec3 viewNormal = texelFetch(usam_temp1, texelPos, 0).rgb;

        imageStore(uimg_prevNZ, texelPos, uvec4(ndpacking_pack(viewNormal, viewZ), 0u, 0u));

        vec3 currColor = texelFetch(usam_ssvbil, texelPos, 0).rgb;
        vec4 prevColorHLen = texelFetch(usam_temp3, texelPos, 0);
        vec2 prevMoments = imageLoad(uimg_temp4, texelPos).xy;

        float newHLen;
        vec2 newMoments;
        vec4 filterInput;
        svgf_update(currColor, prevColorHLen, prevMoments, newHLen, newMoments, filterInput);

        imageStore(uimg_svgfHistoryMoments, texelPos, vec4(newMoments, 0.0, 0.0));
        imageStore(uimg_temp4, texelPos, filterInput);

        float hLenEncoded = saturate((newHLen - 1.0) / 255.0);
        imageStore(uimg_temp5, texelPos, vec4(hLenEncoded, 0.0, 0.0, 0.0));
    }
}