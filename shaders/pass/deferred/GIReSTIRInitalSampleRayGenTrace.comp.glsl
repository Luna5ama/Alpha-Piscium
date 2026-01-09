/*
    References:
        [WYM23] Wyman, Chris, et al. "A Gentle Introduction to ReSTIR". SIGGRAPH 2023.
            https://intro-to-restir.cwyman.org/
        [ANA23] Anagnostou, Kostas. "A Gentler Introduction to ReSTIR". Interplay of Light. 2023.
            https://interplayoflight.wordpress.com/2023/12/17/a-gentler-introduction-to-restir/
        [ALE22] Alegruz. "Screen-Space-ReSTIR-GI". GitHub. 2022.
            https://github.com/Alegruz/Screen-Space-ReSTIR-GI
            BSD 3-Clause License. Copyright (c) 2022, Alegruz.

        You can find full license texts in /licenses

    Other Credits:
        - Belmu (https://github.com/BelmuTM) - Advice on ReSTIR.
*/
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/HiZCheck.glsl"
#include "/techniques/SST2.glsl"
#include "/techniques/gi/InitialSample.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Morton.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(std430, binding = 4) buffer RayData {
    uvec4 ssbo_rayData[];
};

layout(std430, binding = 5) buffer RayDataIndices {
    uint ssbo_rayDataIndices[];
};

layout(rgba16f) uniform restrict writeonly image2D uimg_temp1;
layout(r32f) uniform restrict writeonly image2D uimg_r32f;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(rgba8) uniform restrict writeonly image2D uimg_rgba8;

void main() {
    sst_init(SETTING_GI_SST_THICKNESS);

    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    uvec2 binId = swizzledWGPos >> 1u;
    uint numBinX = (uval_mainImageSizeI.x + 31) >> 5; // 32x32 bin
    uint binIdx = binId.y * numBinX + binId.x;
    ivec2 binLocalPos = texelPos & 31; // 32x32 bin
    uint binLocalIndex = sst2_encodeBinLocalIndex(binLocalPos);
    uint binWriteBaseIndex = binIdx * 1024;
    uint dataIndex = binWriteBaseIndex + binLocalIndex;

    uint rayIndex = 0xFFFFFFFFu;
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos, 4, texelPos);

        if (viewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
            Material material = material_decode(gData);
            vec4 albedoAndEmissive = vec4(gData.albedo, gData.pbrSpecular.a);
            vec4 geomNormalData = vec4(gData.geomNormal * 0.5 + 0.5, 0.0);
            vec4 viewNormalData = vec4(gData.normal * 0.5 + 0.5, 0.0);
            transient_solidAlbedo_store(texelPos, albedoAndEmissive);
            transient_geomViewNormal_store(texelPos, geomNormalData);
            transient_viewNormal_store(texelPos, viewNormalData);
            history_geomViewNormal_store(texelPos, geomNormalData);
            history_viewNormal_store(texelPos, viewNormalData);
            vec3 rayDirView = restir_initialSample_generateRayDir(texelPos, gData.geomNormal, material.tbn);

            SSTRay sstRay = sstray_setup(texelPos, viewPos, rayDirView);
            sst_trace(sstRay, 24);

            if (sstRay.currT > 0.0) {
                uvec4 packedData = sstray_pack(sstRay);
                ssbo_rayData[dataIndex] = packedData;
                rayIndex = sst2_encodeRayIndexBits(binLocalIndex, sstRay);
            } else {
                float hitDistance = restir_initialSample_handleRayResult(sstRay);
                transient_gi_initialSampleHitDistance_store(texelPos, vec4(hitDistance));
            }
        } else {
            transient_geomViewNormal_store(texelPos, vec4(0.0));
            transient_viewNormal_store(texelPos, vec4(0.0));
            history_geomViewNormal_store(texelPos, vec4(0.0));
            history_viewNormal_store(texelPos, vec4(0.0));
            transient_solidAlbedo_store(texelPos, vec4(0.0));
        }
    }
    ssbo_rayDataIndices[dataIndex] = rayIndex;
}
