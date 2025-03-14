#version 460 compatibility

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable

#include "/util/Morton.glsl"
#include "/denoiser/Update.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);

#define SSVBIL_SAMPLE_STEPS SETTING_VBGI_STEPS
#define SSVBIL_SAMPLE_SLICES SETTING_VBGI_SLICES
#include "/post/gtvbgi/GTVBGI.glsl"

layout(rgba16f) uniform writeonly image2D uimg_temp4;
layout(rgba32ui) uniform writeonly uimage2D uimg_svgfHistory;
layout(rgba8) uniform writeonly image2D uimg_temp6;

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos2x2 = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos2x2, global_mainImageSizeI))) {
        vec4 ssvbilData = gtvbgi(texelPos2x2);
        vec4 prevColorHLen = texelFetch(usam_temp1, texelPos2x2 + ivec2(0, global_mipmapSizesI[1].y), 0);
        vec2 prevMoments = texelFetch(usam_temp1, texelPos2x2 + global_mipmapSizesI[1], 0).rg;

        float newHLen;
        vec2 newMoments;
        vec4 filterInput;
        gi_update(ssvbilData.rgb, prevColorHLen, prevMoments, newHLen, newMoments, filterInput);
        filterInput.rgb = dither_fp16(filterInput.rgb, rand_IGN(texelPos2x2, frameCounter));
        imageStore(uimg_temp4, texelPos2x2, filterInput);

        float hLenEncoded = saturate((newHLen - 1.0) / 255.0);
        imageStore(uimg_temp6, texelPos2x2, vec4(hLenEncoded, 0.0, 0.0, 0.0));

        uvec4 packedData;
        svgf_pack(packedData, vec4(filterInput.rgb, newHLen), newMoments);
        imageStore(uimg_svgfHistory, texelPos2x2, packedData);
    }
}