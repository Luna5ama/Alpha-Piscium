#version 460 compatibility

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#define HIZ_SUBGROUP_CHECK a

#include "/techniques/svgf/Update.glsl"
#include "/util/Morton.glsl"
#include "/techniques/HiZ.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);

#define SSVBIL_SAMPLE_STEPS SETTING_VBGI_STEPS
#define SSVBIL_SAMPLE_SLICES SETTING_VBGI_SLICES
#include "/techniques/gtvbgi/GTVBGI.glsl"

layout(rg32ui) uniform writeonly uimage2D uimg_tempRG32UI;
layout(rgba16f) uniform writeonly image2D uimg_debug;

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos2x2 = ivec2(mortonGlobalPosU);
    ivec2 texelPos1x1Base = texelPos2x2 << 1;
    ivec2 texelPos1x1 = texelPos1x1Base + ivec2(morton_8bDecode(uint(frameCounter) & 3u));

    if (all(lessThan(texelPos1x1, global_mainImageSizeI))) {
        vec3 ssvbilData = vec3(0.0);

        if (hiz_groupGroundCheckSubgroup(gl_WorkGroupID.xy, 5)) {
            ssvbilData = gtvbgi(texelPos1x1);
        }

        imageStore(uimg_tempRG32UI, texelPos2x2, uvec4(packHalf4x16(vec4(ssvbilData, 0.0)), 0u, 0u));
        #ifdef SETTING_DEBUG_DEDICATED
        imageStore(uimg_debug, texelPos2x2, vec4(ssvbilData, 1.0));
        #endif
    }
}