#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"
#include "/util/Morton.glsl"
#include "/techniques/HiZCheck.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
#include "/techniques/SSGI.glsl"

void main() {
    sst_init();

    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID + gl_SubgroupInvocationID * gl_NumSubgroups;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec3 geomNormal = vec3(0.0);
        vec3 normal = vec3(0.0);
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(gl_WorkGroupID.xy, 4, texelPos);
        if (viewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
            geomNormal = gData.geomNormal;
            normal = gData.normal;
            Material material = material_decode(gData);

            if (RANDOM_FRAME < MAX_FRAMES && RANDOM_FRAME >= 0) {
                GIHistoryData historyData = gi_historyData_init();
                gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));

                uvec3 workGroupUniformRandKey = uvec3(gl_WorkGroupID.xy, 114514);
                uvec3 subgroupUniformRandKey = uvec3(gl_WorkGroupID.xy, gl_SubgroupID);
                uvec3 pixelRandKey = uvec3(texelPos, 1919810u);
                #if SETTING_GI_COHERENCE_OPTIMIZATION == 0
                uvec3 finalRandKey = pixelRandKey;
                #elif SETTING_GI_COHERENCE_OPTIMIZATION == 1
                uvec3 finalRandKey = subgroupUniformRandKey;
                #elif SETTING_GI_COHERENCE_OPTIMIZATION == 2
                uvec3 finalRandKey = workGroupUniformRandKey;
                #endif

                vec2 rand2 = hash_uintToFloat(hash_44_q3(uvec4(finalRandKey, RANDOM_FRAME)).zw);
                //                vec2 rand2 = rand_stbnVec2(texelPos, RANDOM_FRAME);
                vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
                //                vec4 sampleDirTangentAndPdf = rand_sampleInHemisphere(rand2);
                vec3 sampleDirView = normalize(material.tbn * sampleDirTangentAndPdf.xyz);

                //                ivec2 stbnPos = texelPos + ivec2(rand_r2Seq2(RANDOM_FRAME / 64u) * vec2(128, 128));
                //                vec3 sampleDirTangent = rand_stbnUnitVec3Cosine(stbnPos, RANDOM_FRAME);
                //                vec3 sampleDirView = normalize(material.tbn * sampleDirTangent);

                vec4 resultStuff = ssgiEvalF2(texelPos, viewPos, sampleDirView);

                InitialSampleData initialSample = initialSampleData_init();
                initialSample.hitRadiance = resultStuff.xyz;
                initialSample.directionAndLength.xyz = sampleDirView;
                initialSample.directionAndLength.w = resultStuff.w;
                transient_restir_initialSample_store(texelPos, initialSampleData_pack(initialSample));
            }
        }
        transient_geomViewNormal_store(texelPos, vec4(geomNormal * 0.5 + 0.5, 0.0));
        transient_viewNormal_store(texelPos, vec4(normal * 0.5 + 0.5, 0.0));
    }
}
