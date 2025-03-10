#version 460 compatibility

#extension GL_KHR_shader_subgroup_basic : enable

#include "/atmosphere/Common.glsl"
#include "/denoiser/Update.glsl"
#include "/util/NZPacking.glsl"
#include "/util/Colors.glsl"
#include "/util/Morton.glsl"
#include "/util/Rand.glsl"
#include "/util/Dither.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);

uniform sampler2D usam_temp1;
uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_ssvbil;

layout(rgba16f) uniform writeonly image2D uimg_ssvbil;

layout(rg32ui) uniform writeonly uimage2D uimg_packedNZ;
layout(rgba32ui) uniform writeonly uimage2D uimg_svgfHistory;
layout(rgba8) uniform writeonly image2D uimg_temp6;

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos2x2 = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos2x2, global_mainImageSizeI))) {
        ivec2 texelPos1x1 = texelPos2x2 << 1;
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos1x1, 0).r;
        vec3 viewNormal = texelFetch(usam_temp1, texelPos2x2 + ivec2(global_mipmapSizesI[1].x, 0), 0).rgb;

        vec3 worldNormal = mat3(gbufferModelViewInverse) * viewNormal;

        uvec4 prevNZOutput = uvec4(0u);
        nzpacking_pack(prevNZOutput.xy, worldNormal, viewZ);
        imageStore(uimg_packedNZ, texelPos2x2, prevNZOutput);

        vec3 currColor = texelFetch(usam_ssvbil, texelPos2x2, 0).rgb;
        vec4 prevColorHLen = texelFetch(usam_temp1, texelPos2x2 + ivec2(0, global_mipmapSizesI[1].y), 0);
        vec2 prevMoments = texelFetch(usam_temp1, texelPos2x2 + global_mipmapSizesI[1], 0).rg;

        float newHLen;
        vec2 newMoments;
        vec4 filterInput;
        gi_update(currColor, prevColorHLen, prevMoments, newHLen, newMoments, filterInput);
        filterInput.rgb = dither_fp16(filterInput.rgb, rand_IGN(texelPos2x2, frameCounter));
        imageStore(uimg_ssvbil, texelPos2x2, filterInput);

        float hLenEncoded = saturate((newHLen - 1.0) / 255.0);
        imageStore(uimg_temp6, texelPos2x2, vec4(hLenEncoded, 0.0, 0.0, 0.0));

        uvec4 packedData;
        svgf_pack(packedData, vec4(filterInput.rgb, newHLen), newMoments);
        imageStore(uimg_svgfHistory, texelPos2x2, packedData);
    }
}