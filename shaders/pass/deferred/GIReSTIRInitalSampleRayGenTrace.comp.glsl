#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/HiZCheck.glsl"
#include "/techniques/SST2.glsl"
#include "/techniques/restir/InitialSample.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"
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
layout(rgba32ui) uniform restrict writeonly uimage2D uimg_rgba32ui;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;

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

    uint rayIndex = 0xFFFFFFFFu;
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos, 4, texelPos);

        if (viewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
            transient_geomViewNormal_store(texelPos, vec4(gData.geomNormal * 0.5 + 0.5, 0.0));
            transient_viewNormal_store(texelPos, vec4(gData.normal * 0.5 + 0.5, 0.0));
            Material material = material_decode(gData);

            #ifdef SETTING_DEBUG_GI_TEXT
            if (RANDOM_FRAME < MAX_FRAMES && RANDOM_FRAME >= 0) {
            #endif
                uvec4 randKey = uvec4(texelPos, 1919810u, RANDOM_FRAME);
//                uvec4 randKey = uvec4(texelPos, 1919810u, 1314u);

                vec2 rand2 = hash_uintToFloat(hash_44_q3(randKey).zw);
                // vec2 rand2 = rand_stbnVec2(texelPos, RANDOM_FRAME);

                // vec4 sampleDirTangentAndPdf = rand_sampleInHemisphere(rand2);
                vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
                vec3 sampleDirView = normalize(material.tbn * sampleDirTangentAndPdf.xyz);

                // ivec2 stbnPos = texelPos + ivec2(rand_r2Seq2(RANDOM_FRAME / 64u) * vec2(128, 128));
                // vec3 sampleDirTangent = rand_stbnUnitVec3Cosine(stbnPos, RANDOM_FRAME);
                // vec3 sampleDirView = normalize(material.tbn * sampleDirTangent);

                SSTRay sstRay = sstray_setup(texelPos, viewPos, sampleDirView);
                sst_trace(sstRay, 24);

                if (sstRay.currT > 0.0) {
                    // TODO: cleanup
                    uvec4 packedData = sstray_pack(sstRay);
                    uint binWriteBaseIndex = binIdx * 1024;
                    uint writeIndex = binWriteBaseIndex + binLocalIndex;
                    ssbo_rayData[writeIndex] = packedData;
                    rayIndex = sst2_encodeRayIndexBits(binLocalIndex, sstRay);
                } else {
                    restir_InitialSampleData sampleData = restir_initialSample_handleRayResult(sstRay);
                    transient_restir_initialSample_store(texelPos, restir_initialSampleData_pack(sampleData));
                    #if SETTING_DEBUG_OUTPUT
//                    imageStore(uimg_temp1, texelPos, vec4(sampleData.hitRadiance, 0.0));
                    #endif
                }
            #ifdef SETTING_DEBUG_GI_TEXT
            }
            #endif

        } else {
            transient_geomViewNormal_store(texelPos, vec4(0.0));
            transient_viewNormal_store(texelPos, vec4(0.0));
        }
    }
    uint binIndicesWriteBaseIndex = binIdx * 1024;
    uint indexWriteIndex = binIndicesWriteBaseIndex + binLocalIndex;
    ssbo_rayDataIndices[indexWriteIndex] = rayIndex;
}
