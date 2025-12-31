#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/HiZCheck.glsl"
#include "/util/GBufferData.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgba8) uniform writeonly image2D uimg_rgba8;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
#include "/techniques/gi/Reproject.glsl"

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos, 4, texelPos);
        vec3 geomNormal = vec3(0.0);
        vec3 normal = vec3(0.0);

        if (viewZ > -65536.0) {
            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

            gi_reproject(texelPos, viewZ, gData);

            geomNormal = gData.geomNormal;
            normal = gData.normal;
        }

        transient_lowCloudRender_store(texelPos, uvec4(0u));
        transient_lowCloudAccumulated_store(texelPos, uvec4(0u));
    }
}
