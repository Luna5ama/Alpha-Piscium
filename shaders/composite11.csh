#version 460 compatibility

#include "/atmosphere/Common.glsl"
#include "/general/NDPacking.glsl"
#include "/denoiser/Update.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Colors.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp1;
uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_ssvbil;

uniform sampler2D usam_temp3;
layout(rgba16f) uniform restrict image2D uimg_temp4;

layout(rg32ui) uniform writeonly uimage2D uimg_prevNZ;
layout(rg16f) uniform writeonly image2D uimg_giHistoryMoments;
layout(rgba8) uniform writeonly image2D uimg_temp6;
layout(rgba8) uniform writeonly image2D uimg_temp7;

uniform sampler2D usam_projReject;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 screenCoord = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        vec3 viewNormal = texelFetch(usam_temp1, texelPos, 0).rgb;

        imageStore(uimg_prevNZ, texelPos, uvec4(ndpacking_pack(viewNormal, viewZ), 0u, 0u));

        vec3 currColor = texelFetch(usam_ssvbil, texelPos, 0).rgb;
        vec4 prevColorHLen = texelFetch(usam_temp3, texelPos, 0);
        vec2 prevMoments = imageLoad(uimg_temp4, texelPos).xy;

        vec2 projReject = texelFetch(usam_projReject, texelPos, 0).rg;
        projReject = max(projReject, texelFetchOffset(usam_projReject, texelPos, 0, ivec2(-1, 0)).rg);
        projReject = max(projReject, texelFetchOffset(usam_projReject, texelPos, 0, ivec2(1, 0)).rg);
        projReject = max(projReject, texelFetchOffset(usam_projReject, texelPos, 0, ivec2(0, -1)).rg);
        projReject = max(projReject, texelFetchOffset(usam_projReject, texelPos, 0, ivec2(0, 1)).rg);

        float frustumTest = float(projReject.x > 0.0);
        float newPixel = float(projReject.y > 0.0);

        prevColorHLen.a *= saturate(1.0 - frustumTest);
        prevColorHLen.a *= saturate(1.0 - newPixel * 0.2);

        float newHLen;
        vec2 newMoments;
        vec4 filterInput;
        gi_update(currColor, prevColorHLen, prevMoments, newHLen, newMoments, filterInput);

        imageStore(uimg_giHistoryMoments, texelPos, vec4(newMoments, 0.0, 0.0));
        imageStore(uimg_temp4, texelPos, filterInput);

        float hLenEncoded = saturate((newHLen - 1.0) / 255.0);
        imageStore(uimg_temp6, texelPos, vec4(hLenEncoded, 0.0, 0.0, 0.0));
    }
}