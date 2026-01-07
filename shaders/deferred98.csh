#version 460 compatibility
#define COMP 1

#extension GL_KHR_shader_subgroup_basic : enable

#define SST_DEBUG_PASS a
#include "/techniques/SST.glsl"
#include "/util/Coords.glsl"
#include "/util/GBufferData.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba8) uniform restrict writeonly image2D uimg_temp5;

void main() {
    sst_init();

    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        GBufferData gData = gbufferData_init();
        gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
        gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
        vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
        vec3 viewDir = normalize(-viewPos);
        vec3 refDir = reflect(-viewDir, gData.normal);

        SSTResult sstResult = sst_trace(viewPos, refDir, 0.01);
        vec3 hitColor = vec3(0.0);
        if (sstResult.hit){
            hitColor = transient_solidAlbedo_sample(sstResult.hitScreenPos.xy).rgb;
        }
        imageStore(uimg_temp5, texelPos, vec4(hitColor, 1.0));
    }
}

