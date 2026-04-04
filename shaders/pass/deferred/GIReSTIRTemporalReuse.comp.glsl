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
#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/InitialSample.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Sampling.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(r32f) uniform restrict writeonly image2D uimg_r32f;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;


shared mat3 shared_prevViewToCurrView;
shared vec3 shared_prevViewToCurrViewTrans;

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    if (threadIdx == 0u) {
        // Precompute prevViewToCurrView matrix for the workgroup
        shared_prevViewToCurrView = mat3(gbufferModelView) * mat3(gbufferPrevModelViewInverse);
        shared_prevViewToCurrViewTrans = mat3(gbufferModelView) * (gbufferPrevModelViewInverse[3].xyz - uval_cameraDelta) + gbufferModelView[3].xyz;
    }
    barrier();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        ReSTIRReservoir temporalReservoir = restir_initReservoir();
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos.xy, 4, texelPos);
        if (viewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

            // GBuffer packing opt: Only unpack what's needed for Reprojection
            // GBufferData gData = gbufferData_init();
            // gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            // gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

            vec3 gGeomNormal;
            vec3 gNormal;
            {
                uvec4 data1 = texelFetch(usam_gbufferData1, texelPos, 0);
                vec3 gGeomTangent;
                nzpacking_unpackNormalOct16(data1.r, gGeomNormal, gGeomTangent);
                gGeomNormal = coords_dir_worldToView(gGeomNormal);
                gNormal = coords_dir_worldToView(nzpacking_unpackNormalOct32(data1.b));
            }

            vec4 prevSample = vec4(0.0);
            vec3 prevHitNormal = vec3(0.0);
            float prevPHat = 0.0;
            float wSum = 0.0;

            uvec4 reprojInfoData = transient_gi_diffuse_reprojInfo_fetch(texelPos);
            ReprojectInfo reprojInfo = reprojectInfo_unpack(reprojInfoData);
            float ageResetRand = rand_stbnVec1(rand_newStbnPos(texelPos, RANDOM_FRAME / 64u + 1u), RANDOM_FRAME);
            if (reprojInfo.historyResetFactor > ageResetRand) {
                reprojInfo.bilateralWeights = pow(reprojInfo.bilateralWeights, vec4(1.0 / 16.0));
                vec2 curr2PrevTexelPos = reprojInfo.curr2PrevScreenPos * uval_mainImageSize;
                curr2PrevTexelPos = clamp(curr2PrevTexelPos, vec2(0.5), uval_mainImageSize - 0.5);
                vec2 gatherTexelPos = floor(curr2PrevTexelPos - 0.5) + 1.0;
                vec2 pixelPosFract = fract(curr2PrevTexelPos - 0.5);
                vec2 bilinearWeights2 = pixelPosFract;
                vec4 bilinearWeights4;
                bilinearWeights4.yz = bilinearWeights2.xx;
                bilinearWeights4.xw = 1.0 - bilinearWeights2.xx;
                bilinearWeights4.xy *= bilinearWeights2.yy;
                bilinearWeights4.zw *= 1.0 - bilinearWeights2.yy;

                ivec2 iGatherTexelPos = ivec2(gatherTexelPos);
                uint baseRandSeed = RANDOM_FRAME / 64u + 2u;
                bool oddFrame = bool(frameCounter & 1);

                // Unrolled 4-tap bilinear temporal gather
                // Layout (gather order matches bilinearWeights xyzw):
                //   x = top-left    iGatherTexelPos + (-1,  0)
                //   y = top-right   iGatherTexelPos + ( 0,  0)
                //   z = bottom-right iGatherTexelPos + ( 0, -1)
                //   w = bottom-left  iGatherTexelPos + (-1, -1)

                // --- Neighbor x: top-left ---
                {
                    float combinedWeight = bilinearWeights4.x * reprojInfo.bilateralWeights.x * reprojInfo.historyResetFactor;
                    if (combinedWeight > 0.0) {
                        ivec2 neighborTexelPos = iGatherTexelPos + ivec2(-1, 0);
                        uvec4 prevTemporalReservoirData = oddFrame
                            ? history_restir_reservoirTemporal2_fetch(neighborTexelPos)
                            : history_restir_reservoirTemporal1_fetch(neighborTexelPos);
                        ReSTIRReservoir neighborReservoir = restir_reservoir_unpack(prevTemporalReservoirData);
                        if (restir_isReservoirValid(neighborReservoir)) {
                            vec3 neighborHitNormalRaw = history_restir_prevHitNormal_fetch(neighborTexelPos).xyz;
                            vec4 neighborSample = history_restir_prevSample_fetch(neighborTexelPos);
                            vec3 neighborHitNormal = normalize(shared_prevViewToCurrView * (neighborHitNormalRaw * 2.0 - 1.0));
                            bool valid = true;
                            if (neighborReservoir.Y.w > 0.0) {
                                vec2 neighborScreenPos = coords_texelToUV(neighborTexelPos, uval_mainImageSizeRcp);
                                float neighborViewZ = history_viewZ_fetch(neighborTexelPos).x;
                                vec3 neighborViewPos = coords_toViewCoord(neighborScreenPos, neighborViewZ, global_prevCamProjInverse);
                                vec3 neighborHitViewPos = neighborViewPos + neighborReservoir.Y.xyz * neighborReservoir.Y.w;
                                vec3 prev2CurrHitViewPos = shared_prevViewToCurrView * neighborHitViewPos + shared_prevViewToCurrViewTrans;
                                vec3 hitDiff = prev2CurrHitViewPos - viewPos;
                                float hitDistance = length(hitDiff);
                                neighborReservoir.Y.xyz = hitDiff / hitDistance;
                                neighborReservoir.Y.w = hitDistance;
                                vec4 prev2CurrHitClipPos = global_camProj * vec4(prev2CurrHitViewPos, 1.0);
                                uint clipFlag = uint(prev2CurrHitClipPos.z > 0.0);
                                clipFlag &= uint(all(lessThan(abs(prev2CurrHitClipPos.xy), prev2CurrHitClipPos.ww)));
                                vec3 prev2CurrHitScreenPos = vec3(prev2CurrHitClipPos.xy / prev2CurrHitClipPos.w * 0.5 + 0.5, prev2CurrHitClipPos.z / prev2CurrHitClipPos.w);
                                clipFlag &= uint(saturate(prev2CurrHitScreenPos) == prev2CurrHitScreenPos);
                                if (!bool(clipFlag)) {
                                    valid = false;
                                } else {
                                    vec3 prev2CurrNeighborViewPos = shared_prevViewToCurrView * neighborViewPos + shared_prevViewToCurrViewTrans;
                                    float RA2 = hitDistance * hitDistance;
                                    vec3 dirA = neighborReservoir.Y.xyz;
                                    vec3 offsetB = prev2CurrHitViewPos - prev2CurrNeighborViewPos;
                                    float RB2 = dot(offsetB, offsetB);
                                    vec3 dirB = offsetB * inversesqrt(max(RB2, 1e-12));
                                    float cosPhiA = -dot(dirA, neighborHitNormal);
                                    float cosPhiB = -dot(dirB, neighborHitNormal);
                                    float jacobian = 1.0;
                                    if (cosPhiA <= 0.0 || dot(gNormal, dirA) <= 0.0) {
                                        jacobian = 0.0;
                                    } else if (cosPhiB > 5e-2) {
                                        jacobian = min((RB2 * cosPhiA) / (RA2 * cosPhiB), 256.0);
                                    }
                                    neighborReservoir.avgWY *= jacobian;
                                }
                            } else {
                                neighborReservoir.Y.xyz = normalize(shared_prevViewToCurrView * neighborReservoir.Y.xyz);
                            }
                            if (valid) {
                                float neighborPHat = length(neighborSample.xyz * neighborSample.w);
                                neighborReservoir.m *= combinedWeight;
                                float wi = max(0.0, neighborReservoir.avgWY) * neighborReservoir.m * neighborPHat;
                                float neighborRand = rand_stbnVec1(rand_newStbnPos(texelPos, baseRandSeed), RANDOM_FRAME);
                                if (restir_updateReservoir(temporalReservoir, wSum, neighborReservoir.Y, wi, neighborReservoir.m, neighborRand)) {
                                    prevSample = neighborSample;
                                    prevHitNormal = neighborHitNormal;
                                    prevPHat = neighborPHat;
                                }
                            }
                        }
                    }
                }

                // --- Neighbor y: top-right ---
                {
                    float combinedWeight = bilinearWeights4.y * reprojInfo.bilateralWeights.y * reprojInfo.historyResetFactor;
                    if (combinedWeight > 0.0) {
                        ivec2 neighborTexelPos = iGatherTexelPos;
                        uvec4 prevTemporalReservoirData = oddFrame
                            ? history_restir_reservoirTemporal2_fetch(neighborTexelPos)
                            : history_restir_reservoirTemporal1_fetch(neighborTexelPos);
                        ReSTIRReservoir neighborReservoir = restir_reservoir_unpack(prevTemporalReservoirData);
                        if (restir_isReservoirValid(neighborReservoir)) {
                            vec3 neighborHitNormalRaw = history_restir_prevHitNormal_fetch(neighborTexelPos).xyz;
                            vec4 neighborSample = history_restir_prevSample_fetch(neighborTexelPos);
                            vec3 neighborHitNormal = normalize(shared_prevViewToCurrView * (neighborHitNormalRaw * 2.0 - 1.0));
                            bool valid = true;
                            if (neighborReservoir.Y.w > 0.0) {
                                vec2 neighborScreenPos = coords_texelToUV(neighborTexelPos, uval_mainImageSizeRcp);
                                float neighborViewZ = history_viewZ_fetch(neighborTexelPos).x;
                                vec3 neighborViewPos = coords_toViewCoord(neighborScreenPos, neighborViewZ, global_prevCamProjInverse);
                                vec3 neighborHitViewPos = neighborViewPos + neighborReservoir.Y.xyz * neighborReservoir.Y.w;
                                vec3 prev2CurrHitViewPos = shared_prevViewToCurrView * neighborHitViewPos + shared_prevViewToCurrViewTrans;
                                vec3 hitDiff = prev2CurrHitViewPos - viewPos;
                                float hitDistance = length(hitDiff);
                                neighborReservoir.Y.xyz = hitDiff / hitDistance;
                                neighborReservoir.Y.w = hitDistance;
                                vec4 prev2CurrHitClipPos = global_camProj * vec4(prev2CurrHitViewPos, 1.0);
                                uint clipFlag = uint(prev2CurrHitClipPos.z > 0.0);
                                clipFlag &= uint(all(lessThan(abs(prev2CurrHitClipPos.xy), prev2CurrHitClipPos.ww)));
                                vec3 prev2CurrHitScreenPos = vec3(prev2CurrHitClipPos.xy / prev2CurrHitClipPos.w * 0.5 + 0.5, prev2CurrHitClipPos.z / prev2CurrHitClipPos.w);
                                clipFlag &= uint(saturate(prev2CurrHitScreenPos) == prev2CurrHitScreenPos);
                                if (!bool(clipFlag)) {
                                    valid = false;
                                } else {
                                    vec3 prev2CurrNeighborViewPos = shared_prevViewToCurrView * neighborViewPos + shared_prevViewToCurrViewTrans;
                                    float RA2 = hitDistance * hitDistance;
                                    vec3 dirA = neighborReservoir.Y.xyz;
                                    vec3 offsetB = prev2CurrHitViewPos - prev2CurrNeighborViewPos;
                                    float RB2 = dot(offsetB, offsetB);
                                    vec3 dirB = offsetB * inversesqrt(max(RB2, 1e-12));
                                    float cosPhiA = -dot(dirA, neighborHitNormal);
                                    float cosPhiB = -dot(dirB, neighborHitNormal);
                                    float jacobian = 1.0;
                                    if (cosPhiA <= 0.0 || dot(gNormal, dirA) <= 0.0) {
                                        jacobian = 0.0;
                                    } else if (cosPhiB > 5e-2) {
                                        jacobian = min((RB2 * cosPhiA) / (RA2 * cosPhiB), 256.0);
                                    }
                                    neighborReservoir.avgWY *= jacobian;
                                }
                            } else {
                                neighborReservoir.Y.xyz = normalize(shared_prevViewToCurrView * neighborReservoir.Y.xyz);
                            }
                            if (valid) {
                                float neighborPHat = length(neighborSample.xyz * neighborSample.w);
                                neighborReservoir.m *= combinedWeight;
                                float wi = max(0.0, neighborReservoir.avgWY) * neighborReservoir.m * neighborPHat;
                                float neighborRand = rand_stbnVec1(rand_newStbnPos(texelPos, baseRandSeed + 1u), RANDOM_FRAME);
                                if (restir_updateReservoir(temporalReservoir, wSum, neighborReservoir.Y, wi, neighborReservoir.m, neighborRand)) {
                                    prevSample = neighborSample;
                                    prevHitNormal = neighborHitNormal;
                                    prevPHat = neighborPHat;
                                }
                            }
                        }
                    }
                }

                // --- Neighbor z: bottom-right ---
                {
                    float combinedWeight = bilinearWeights4.z * reprojInfo.bilateralWeights.z * reprojInfo.historyResetFactor;
                    if (combinedWeight > 0.0) {
                        ivec2 neighborTexelPos = iGatherTexelPos + ivec2(0, -1);
                        uvec4 prevTemporalReservoirData = oddFrame
                            ? history_restir_reservoirTemporal2_fetch(neighborTexelPos)
                            : history_restir_reservoirTemporal1_fetch(neighborTexelPos);
                        ReSTIRReservoir neighborReservoir = restir_reservoir_unpack(prevTemporalReservoirData);
                        if (restir_isReservoirValid(neighborReservoir)) {
                            vec3 neighborHitNormalRaw = history_restir_prevHitNormal_fetch(neighborTexelPos).xyz;
                            vec4 neighborSample = history_restir_prevSample_fetch(neighborTexelPos);
                            vec3 neighborHitNormal = normalize(shared_prevViewToCurrView * (neighborHitNormalRaw * 2.0 - 1.0));
                            bool valid = true;
                            if (neighborReservoir.Y.w > 0.0) {
                                vec2 neighborScreenPos = coords_texelToUV(neighborTexelPos, uval_mainImageSizeRcp);
                                float neighborViewZ = history_viewZ_fetch(neighborTexelPos).x;
                                vec3 neighborViewPos = coords_toViewCoord(neighborScreenPos, neighborViewZ, global_prevCamProjInverse);
                                vec3 neighborHitViewPos = neighborViewPos + neighborReservoir.Y.xyz * neighborReservoir.Y.w;
                                vec3 prev2CurrHitViewPos = shared_prevViewToCurrView * neighborHitViewPos + shared_prevViewToCurrViewTrans;
                                vec3 hitDiff = prev2CurrHitViewPos - viewPos;
                                float hitDistance = length(hitDiff);
                                neighborReservoir.Y.xyz = hitDiff / hitDistance;
                                neighborReservoir.Y.w = hitDistance;
                                vec4 prev2CurrHitClipPos = global_camProj * vec4(prev2CurrHitViewPos, 1.0);
                                uint clipFlag = uint(prev2CurrHitClipPos.z > 0.0);
                                clipFlag &= uint(all(lessThan(abs(prev2CurrHitClipPos.xy), prev2CurrHitClipPos.ww)));
                                vec3 prev2CurrHitScreenPos = vec3(prev2CurrHitClipPos.xy / prev2CurrHitClipPos.w * 0.5 + 0.5, prev2CurrHitClipPos.z / prev2CurrHitClipPos.w);
                                clipFlag &= uint(saturate(prev2CurrHitScreenPos) == prev2CurrHitScreenPos);
                                if (!bool(clipFlag)) {
                                    valid = false;
                                } else {
                                    vec3 prev2CurrNeighborViewPos = shared_prevViewToCurrView * neighborViewPos + shared_prevViewToCurrViewTrans;
                                    float RA2 = hitDistance * hitDistance;
                                    vec3 dirA = neighborReservoir.Y.xyz;
                                    vec3 offsetB = prev2CurrHitViewPos - prev2CurrNeighborViewPos;
                                    float RB2 = dot(offsetB, offsetB);
                                    vec3 dirB = offsetB * inversesqrt(max(RB2, 1e-12));
                                    float cosPhiA = -dot(dirA, neighborHitNormal);
                                    float cosPhiB = -dot(dirB, neighborHitNormal);
                                    float jacobian = 1.0;
                                    if (cosPhiA <= 0.0 || dot(gNormal, dirA) <= 0.0) {
                                        jacobian = 0.0;
                                    } else if (cosPhiB > 5e-2) {
                                        jacobian = min((RB2 * cosPhiA) / (RA2 * cosPhiB), 256.0);
                                    }
                                    neighborReservoir.avgWY *= jacobian;
                                }
                            } else {
                                neighborReservoir.Y.xyz = normalize(shared_prevViewToCurrView * neighborReservoir.Y.xyz);
                            }
                            if (valid) {
                                float neighborPHat = length(neighborSample.xyz * neighborSample.w);
                                neighborReservoir.m *= combinedWeight;
                                float wi = max(0.0, neighborReservoir.avgWY) * neighborReservoir.m * neighborPHat;
                                float neighborRand = rand_stbnVec1(rand_newStbnPos(texelPos, baseRandSeed + 2u), RANDOM_FRAME);
                                if (restir_updateReservoir(temporalReservoir, wSum, neighborReservoir.Y, wi, neighborReservoir.m, neighborRand)) {
                                    prevSample = neighborSample;
                                    prevHitNormal = neighborHitNormal;
                                    prevPHat = neighborPHat;
                                }
                            }
                        }
                    }
                }

                // --- Neighbor w: bottom-left ---
                {
                    float combinedWeight = bilinearWeights4.w * reprojInfo.bilateralWeights.w * reprojInfo.historyResetFactor;
                    if (combinedWeight > 0.0) {
                        ivec2 neighborTexelPos = iGatherTexelPos + ivec2(-1, -1);
                        uvec4 prevTemporalReservoirData = oddFrame
                            ? history_restir_reservoirTemporal2_fetch(neighborTexelPos)
                            : history_restir_reservoirTemporal1_fetch(neighborTexelPos);
                        ReSTIRReservoir neighborReservoir = restir_reservoir_unpack(prevTemporalReservoirData);
                        if (restir_isReservoirValid(neighborReservoir)) {
                            vec3 neighborHitNormalRaw = history_restir_prevHitNormal_fetch(neighborTexelPos).xyz;
                            vec4 neighborSample = history_restir_prevSample_fetch(neighborTexelPos);
                            vec3 neighborHitNormal = normalize(shared_prevViewToCurrView * (neighborHitNormalRaw * 2.0 - 1.0));
                            bool valid = true;
                            if (neighborReservoir.Y.w > 0.0) {
                                vec2 neighborScreenPos = coords_texelToUV(neighborTexelPos, uval_mainImageSizeRcp);
                                float neighborViewZ = history_viewZ_fetch(neighborTexelPos).x;
                                vec3 neighborViewPos = coords_toViewCoord(neighborScreenPos, neighborViewZ, global_prevCamProjInverse);
                                vec3 neighborHitViewPos = neighborViewPos + neighborReservoir.Y.xyz * neighborReservoir.Y.w;
                                vec3 prev2CurrHitViewPos = shared_prevViewToCurrView * neighborHitViewPos + shared_prevViewToCurrViewTrans;
                                vec3 hitDiff = prev2CurrHitViewPos - viewPos;
                                float hitDistance = length(hitDiff);
                                neighborReservoir.Y.xyz = hitDiff / hitDistance;
                                neighborReservoir.Y.w = hitDistance;
                                vec4 prev2CurrHitClipPos = global_camProj * vec4(prev2CurrHitViewPos, 1.0);
                                uint clipFlag = uint(prev2CurrHitClipPos.z > 0.0);
                                clipFlag &= uint(all(lessThan(abs(prev2CurrHitClipPos.xy), prev2CurrHitClipPos.ww)));
                                vec3 prev2CurrHitScreenPos = vec3(prev2CurrHitClipPos.xy / prev2CurrHitClipPos.w * 0.5 + 0.5, prev2CurrHitClipPos.z / prev2CurrHitClipPos.w);
                                clipFlag &= uint(saturate(prev2CurrHitScreenPos) == prev2CurrHitScreenPos);
                                if (!bool(clipFlag)) {
                                    valid = false;
                                } else {
                                    vec3 prev2CurrNeighborViewPos = shared_prevViewToCurrView * neighborViewPos + shared_prevViewToCurrViewTrans;
                                    float RA2 = hitDistance * hitDistance;
                                    vec3 dirA = neighborReservoir.Y.xyz;
                                    vec3 offsetB = prev2CurrHitViewPos - prev2CurrNeighborViewPos;
                                    float RB2 = dot(offsetB, offsetB);
                                    vec3 dirB = offsetB * inversesqrt(max(RB2, 1e-12));
                                    float cosPhiA = -dot(dirA, neighborHitNormal);
                                    float cosPhiB = -dot(dirB, neighborHitNormal);
                                    float jacobian = 1.0;
                                    if (cosPhiA <= 0.0 || dot(gNormal, dirA) <= 0.0) {
                                        jacobian = 0.0;
                                    } else if (cosPhiB > 5e-2) {
                                        jacobian = min((RB2 * cosPhiA) / (RA2 * cosPhiB), 256.0);
                                    }
                                    neighborReservoir.avgWY *= jacobian;
                                }
                            } else {
                                neighborReservoir.Y.xyz = normalize(shared_prevViewToCurrView * neighborReservoir.Y.xyz);
                            }
                            if (valid) {
                                float neighborPHat = length(neighborSample.xyz * neighborSample.w);
                                neighborReservoir.m *= combinedWeight;
                                float wi = max(0.0, neighborReservoir.avgWY) * neighborReservoir.m * neighborPHat;
                                float neighborRand = rand_stbnVec1(rand_newStbnPos(texelPos, baseRandSeed + 3u), RANDOM_FRAME);
                                if (restir_updateReservoir(temporalReservoir, wSum, neighborReservoir.Y, wi, neighborReservoir.m, neighborRand)) {
                                    prevSample = neighborSample;
                                    prevHitNormal = neighborHitNormal;
                                    prevPHat = neighborPHat;
                                }
                            }
                        }
                    }
                }
            }

            // Re-fetch and fully unpack for material decoding (needed for Spatial / Initial) using fresh registers
            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
            Material material = material_decode(gData);

            {
                float hitDistance = transient_gi_initialSampleHitDistance_fetch(texelPos).x;
                restir_InitialSampleData initialSample = restir_initalSample_restoreData(texelPos, viewZ, gData.geomNormal, material, hitDistance);
                vec3 hitRadiance = initialSample.hitRadiance;
                vec3 sampleDirView = initialSample.directionAndLength.xyz;

                vec3 hitViewPos = viewPos + sampleDirView * hitDistance;
                vec3 hitScreenPos = coords_viewToScreen(hitViewPos, global_camProj);
                ivec2 hitTexelPos = ivec2(hitScreenPos.xy * uval_mainImageSize);
                vec4 hitGeomNormalData = transient_geomViewNormal_fetch(hitTexelPos);
                vec3 hitGeomNormal = normalize(hitGeomNormalData.xyz * 2.0 - 1.0);
                float geomNormalDot = dot(hitGeomNormal, gData.geomNormal);

                if (geomNormalDot > 0.99) {
                    transient_gi_initialSampleHitDistance_store(texelPos, vec4(-1.0));
                }

                float brdf = saturate(dot(gData.normal, sampleDirView)) / PI;
                vec3 initalSample = brdf * hitRadiance;

                float samplePdf = brdf;

                float newPHat = length(initalSample);
                float newWi = samplePdf <= 0.0 ? 0.0 : newPHat / samplePdf;

                float reservoirRand1 = rand_stbnVec1(rand_newStbnPos(texelPos, RANDOM_FRAME / 64u + 6u), RANDOM_FRAME);

                float reservoirPHat = prevPHat;
                vec4 finalSample = prevSample;
                vec3 hitNormal = prevHitNormal;
                if (restir_updateReservoir(temporalReservoir, wSum, vec4(sampleDirView, hitDistance), newWi, 1.0, reservoirRand1)) {
                    reservoirPHat = newPHat;
                    finalSample = vec4(hitRadiance, brdf);

                    vec4 hitNormalData = transient_viewNormal_fetch(hitTexelPos);
                    hitNormal = normalize(hitNormalData.xyz * 2.0 - 1.0);
                }
                float avgWSum = wSum / temporalReservoir.m;
                temporalReservoir.avgWY = reservoirPHat <= 0.0 ? 0.0 : (avgWSum / reservoirPHat);
                temporalReservoir.m = clamp(temporalReservoir.m, 0.0, float(SETTING_GI_TEMPORAL_REUSE_LIMIT));
                #if USE_REFERENCE
                vec4 ssgiOut = vec4(initalSample * safeRcp(samplePdf), hitDistance);
                ssgiOut.rgb = clamp(ssgiOut.rgb, 0.0, FP16_MAX);
                transient_ssgiOut_store(texelPos, ssgiOut);
                #elif !defined(SETTING_GI_SPATIAL_REUSE)
                vec4 ssgiOut = vec4(finalSample.rgb * finalSample.a * temporalReservoir.avgWY, hitDistance);
                ssgiOut.rgb = clamp(ssgiOut.rgb, 0.0, FP16_MAX);
                transient_ssgiOut_store(texelPos, ssgiOut);
                #endif

                SpatialSampleData spatialSample = spatialSampleData_init();
                spatialSample.sampleValue = finalSample;
                spatialSample.geomNormal = gData.geomNormal;
                spatialSample.normal = gData.normal;
                spatialSample.hitNormal = hitNormal;
                transient_restir_spatialInput_store(texelPos, spatialSampleData_pack(spatialSample));
            }
        }
        uvec4 packedReservoir = restir_reservoir_pack(temporalReservoir);
        if (bool(frameCounter & 1)) {
            history_restir_reservoirTemporal1_store(texelPos, packedReservoir);
        } else {
            history_restir_reservoirTemporal2_store(texelPos, packedReservoir);
        }
    }
}