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

layout(rgba16f) uniform restrict image2D uimg_ssvbil;

layout(rg32ui) uniform writeonly uimage2D uimg_prevNZ;
layout(rgba32ui) uniform writeonly uimage2D uimg_svgfHistory;
layout(rgba8) uniform writeonly image2D uimg_temp6;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 screenCoord = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        vec3 viewNormal = texelFetch(usam_temp1, texelPos, 0).rgb;
        vec3 worldNormal = mat3(gbufferModelViewInverse) * viewNormal;

        uvec4 prevNZOutput = uvec4(0u);
        ndpacking_pack(prevNZOutput.xy, worldNormal, viewZ);
        imageStore(uimg_prevNZ, texelPos, prevNZOutput);

        vec3 currColor = texelFetch(usam_ssvbil, texelPos, 0).rgb;
        vec4 prevColorHLen = texelFetch(usam_temp3, texelPos, 0);
        vec2 prevMoments = texelFetch(usam_temp4, texelPos, 0).rg;

        float newHLen;
        vec2 newMoments;
        vec4 filterInput;
        gi_update(currColor, prevColorHLen, prevMoments, newHLen, newMoments, filterInput);
        filterInput.rgb = dither_fp16(filterInput.rgb, rand_IGN(texelPos, frameCounter));
        imageStore(uimg_ssvbil, texelPos, filterInput);

        float hLenEncoded = saturate((newHLen - 1.0) / 255.0);
        imageStore(uimg_temp6, texelPos, vec4(hLenEncoded, 0.0, 0.0, 0.0));

        uvec4 packedData;
        svgf_pack(packedData, vec4(filterInput.rgb, newHLen), newMoments);
        imageStore(uimg_svgfHistory, texelPos, packedData);
    }
}