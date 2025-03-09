#version 460 compatibility

#include "/atmosphere/Common.glsl"
#include "/general/NDPacking.glsl"
#include "/denoiser/Update.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Colors.glsl"
#include "/util/Rand.glsl"
#include "/util/Dither.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp1;
uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_ssvbil;

uniform sampler2D usam_temp3;
uniform sampler2D usam_temp4;
uniform sampler2D usam_temp7;

#ifdef SETTING_DENOISER
layout(rgba16f) uniform writeonly image2D uimg_temp4;
#else
layout(rgba16f) uniform restrict image2D uimg_ssvbil;
#endif

layout(rg32ui) uniform writeonly uimage2D uimg_prevNZ;
layout(rgba32ui) uniform writeonly uimage2D uimg_svgfHistory;

uniform sampler2D usam_projReject;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 screenCoord = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        vec3 viewNormal = texelFetch(usam_temp1, texelPos, 0).rgb;
        vec3 worldNormal = mat3(gbufferModelViewInverse) * viewNormal;

        imageStore(uimg_prevNZ, texelPos, uvec4(ndpacking_pack(worldNormal, viewZ), 0u, 0u));

        vec3 currColor = texelFetch(usam_ssvbil, texelPos, 0).rgb;
        vec4 prevColorHLen = texelFetch(usam_temp3, texelPos, 0);
        vec2 prevMoments = texelFetch(usam_temp4, texelPos, 0).rg;

        vec2 projReject = texelFetch(usam_projReject, texelPos, 0).rg;
        projReject = max(projReject, texelFetchOffset(usam_projReject, texelPos, 0, ivec2(-1, 0)).rg);
        projReject = max(projReject, texelFetchOffset(usam_projReject, texelPos, 0, ivec2(1, 0)).rg);
        projReject = max(projReject, texelFetchOffset(usam_projReject, texelPos, 0, ivec2(0, -1)).rg);
        projReject = max(projReject, texelFetchOffset(usam_projReject, texelPos, 0, ivec2(0, 1)).rg);

        float frustumTest = float(projReject.x > 0.0);
        prevColorHLen.a *= saturate(1.0 - frustumTest * 0.9);

        float newHLen;
        vec2 newMoments;
        vec4 filterInput;
        gi_update(currColor, prevColorHLen, prevMoments, newHLen, newMoments, filterInput);

        filterInput.rgb = dither_fp16(filterInput.rgb, rand_IGN(texelPos, frameCounter));

        #ifdef SETTING_DENOISER
        imageStore(uimg_temp4, texelPos, filterInput);
        #else
        imageStore(uimg_ssvbil, texelPos, filterInput);
        #endif

        uvec4 packedData;
        svgf_pack(packedData, vec4(filterInput.rgb, newHLen), newMoments);
        imageStore(uimg_svgfHistory, texelPos, packedData);
    }
}