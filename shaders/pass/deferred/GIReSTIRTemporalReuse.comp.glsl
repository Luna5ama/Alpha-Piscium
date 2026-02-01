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
#include "/util/Hash.glsl"
#include "/util/Sampling.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;

int selectWeighted(vec4 bilinearWeights, vec4 bilateralWeights, float rand) {
    vec4 combinedWeights = bilinearWeights * bilateralWeights;

    vec4 prefixSum;
    prefixSum.x = combinedWeights.x;
    prefixSum.y = prefixSum.x + combinedWeights.y;
    prefixSum.z = prefixSum.y + combinedWeights.z;
    prefixSum.w = prefixSum.z + combinedWeights.w;

    float total = prefixSum.w;
    float threshold = rand * total;

    vec4 cmp = step(prefixSum, vec4(threshold));
    int selectedIndex = int(sum4(cmp));
    vec4 bilateralWeightMasked = mix(vec4(0.0), bilateralWeights, equal(ivec4(selectedIndex), ivec4(0, 1, 2, 3)));
    float selectedWeight = sum4(bilateralWeightMasked);
    return mix(-1, selectedIndex, selectedWeight > 0.96);
}

shared mat3 shared_prevViewToCurrView;
shared mat4 shared_prevViewToCurrViewPos;

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

        mat4 prevToCurrWorld = mat4(1.0);
        prevToCurrWorld[3].xyz = -uval_cameraDelta;
        shared_prevViewToCurrViewPos = gbufferModelView * prevToCurrWorld * gbufferPrevModelViewInverse;
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
            bool gIsHand;
            {
                uvec4 data1 = texelFetch(usam_gbufferData1, texelPos, 0);
                vec3 gGeomTangent;
                nzpacking_unpackNormalOct16(data1.r, gGeomNormal, gGeomTangent);
                gGeomNormal = coords_dir_worldToView(gGeomNormal);
                gNormal = coords_dir_worldToView(nzpacking_unpackNormalOct32(data1.b));

                uvec4 data2 = texelFetch(usam_gbufferData2, texelPos, 0);
                gIsHand = bool(bitfieldExtract(data2.r, 24, 1));
            }

            uvec3 baseRandKey = uvec3(texelPos, RANDOM_FRAME);

            vec4 prevSample = vec4(0.0);
            vec3 prevHitNormal = vec3(0.0);

            uvec4 reprojInfoData = transient_gi_diffuse_reprojInfo_fetch(texelPos);
            ReprojectInfo reprojInfo = reprojectInfo_unpack(reprojInfoData);
            float ageResetRand = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 987123654u)).x);
            if (reprojInfo.historyResetFactor > ageResetRand) {
                ivec2 prevTexelPos = ivec2(-1);

                vec3 currViewPos = viewPos;
                vec4 curr2PrevViewPos = coord_viewCurrToPrev(vec4(currViewPos, 1.0), gIsHand);
                vec4 curr2PrevClipPos = global_prevCamProj * curr2PrevViewPos;
                uint clipFlag = uint(curr2PrevClipPos.z > 0.0);
                clipFlag &= uint(all(lessThan(abs(curr2PrevClipPos.xy), curr2PrevClipPos.ww)));
                if (bool(clipFlag)) {
                    vec2 curr2PrevNDC = curr2PrevClipPos.xy / curr2PrevClipPos.w;
                    vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;
                    ivec2 curr2PrevTexelPos = ivec2(curr2PrevScreen * uval_mainImageSize);

                    vec4 prevGeomNormalData = history_geomViewNormal_fetch(curr2PrevTexelPos);
                    vec4 prevNormalData = history_viewNormal_fetch(curr2PrevTexelPos);
                    vec4 prevViewZData = history_viewZ_fetch(curr2PrevTexelPos);

                    vec3 prevGeomNormal = normalize(shared_prevViewToCurrView * (prevGeomNormalData.xyz * 2.0 - 1.0));
                    vec3 prevNormal = normalize(shared_prevViewToCurrView * (prevNormalData.xyz * 2.0 - 1.0));

                    float prevViewZ = prevViewZData.x;
                    vec3 prevViewPos = coords_toViewCoord(curr2PrevScreen, prevViewZ, global_prevCamProjInverse);
                    vec3 prev2CurrViewPos = (shared_prevViewToCurrViewPos * vec4(prevViewPos, 1.0)).xyz;

                    float geomNormalDot = dot(gGeomNormal, prevGeomNormal);
                    float normalDot = dot(gNormal, prevNormal);
                    float planeDistance = gi_planeDistance(viewPos, gGeomNormal, prev2CurrViewPos, prevGeomNormal);

                    // geomNormal: 0.998629534755 ~= 5 degrees
                    // normal: 0.992546151641 ~= 7 degrees
                    float zThreshold = max(abs(viewZ), 1.0) * 0.01; // 1% of depth
                    if (geomNormalDot > 0.998629534755 && normalDot > 0.992546151641 && planeDistance < zThreshold) {
                        prevTexelPos = curr2PrevTexelPos;
                    }
                }

                if (prevTexelPos == ivec2(-1)) {
                    vec2 curr2PrevTexelPos = reprojInfo.curr2PrevScreenPos * uval_mainImageSize;
                    vec2 centerPixel = curr2PrevTexelPos - 0.5;
                    vec2 gatherOrigin = floor(centerPixel);
                    vec2 gatherTexelPos = gatherOrigin + 1.0;
                    vec2 pixelPosFract = fract(centerPixel);

                    vec2 bilinearWeights2 = pixelPosFract;
                    vec4 blinearWeights4;
                    blinearWeights4.yz = bilinearWeights2.xx;
                    blinearWeights4.xw = 1.0 - bilinearWeights2.xx;
                    blinearWeights4.xy *= bilinearWeights2.yy;
                    blinearWeights4.zw *= 1.0 - bilinearWeights2.yy;

                    float rand = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 987654u)).x);
                    int selectedIndex = selectWeighted(blinearWeights4, reprojInfo.bilateralWeights, rand);

                    if (selectedIndex >= 0){
                        vec2 selectedTexelPos = gatherTexelPos + sampling_indexToGatherOffset(selectedIndex) * 0.5;
                        prevTexelPos = ivec2(selectedTexelPos);
                    }
                }

                if (prevTexelPos != ivec2(-1)) {
                    uvec4 prevTemporalReservoirData;
                    if (bool(frameCounter & 1)) {
                        prevTemporalReservoirData = history_restir_reservoirTemporal2_fetch(prevTexelPos);
                    } else {
                        prevTemporalReservoirData = history_restir_reservoirTemporal1_fetch(prevTexelPos);
                    }

                    ReSTIRReservoir prevTemporalReservoir = restir_reservoir_unpack(prevTemporalReservoirData);
                    prevTemporalReservoir.m = uint(ceil(float(prevTemporalReservoir.m) * global_historyResetFactor));
                    if (restir_isReservoirValid(prevTemporalReservoir)) {
                        vec3 prevHitNormalData = history_restir_prevHitNormal_fetch(prevTexelPos).xyz;
                        prevSample = history_restir_prevSample_fetch(prevTexelPos);
                        prevHitNormal = normalize(shared_prevViewToCurrView * (prevHitNormalData * 2.0 - 1.0));

                        if (prevTemporalReservoir.Y.w > 0.0) {
                            vec2 prevScreenPos = coords_texelToUV(prevTexelPos, uval_mainImageSizeRcp);
                            float prevViewZ = history_viewZ_fetch(prevTexelPos).x;
                            vec3 prevViewPos = coords_toViewCoord(prevScreenPos, prevViewZ, global_prevCamProjInverse);

                            vec3 prevHitViewPos = prevViewPos + prevTemporalReservoir.Y.xyz * prevTemporalReservoir.Y.w;
                            vec3 prev2CurrHitViewPos = (shared_prevViewToCurrViewPos * vec4(prevHitViewPos, 1.0)).xyz;
                            vec3 hitDiff = prev2CurrHitViewPos - viewPos;
                            float hitDistance = length(hitDiff);

                            prevTemporalReservoir.Y.xyz = hitDiff / hitDistance;
                            prevTemporalReservoir.Y.w = hitDistance;

                            vec4 prev2CurrHitClipPos = global_camProj * vec4(prev2CurrHitViewPos, 1.0);
                            uint clipFlag = uint(prev2CurrHitClipPos.z > 0.0);
                            clipFlag &= uint(all(lessThan(abs(prev2CurrHitClipPos.xy), prev2CurrHitClipPos.ww)));
                            // Reuse clip pos for screen conversion instead of calling coords_viewToScreen
                            vec3 prev2CurrHitScreenPos = vec3(prev2CurrHitClipPos.xy / prev2CurrHitClipPos.w * 0.5 + 0.5, prev2CurrHitClipPos.z / prev2CurrHitClipPos.w);
                            clipFlag &= uint(saturate(prev2CurrHitScreenPos) == prev2CurrHitScreenPos);

                            if (bool(clipFlag)) {
                                // Jacobian correction for reconnection shift
                                {
                                    // Original path: from the temporal neighbor pixel (where the sample came from) to hit point
                                    // prev2CurrNeighborViewPos is the neighbor's position transformed to current frame coordinates
                                    vec3 prev2CurrNeighborViewPos = (shared_prevViewToCurrViewPos * vec4(prevViewPos, 1.0)).xyz;

                                    // Vector from current pixel to hit point (shifted path) - this is the new path
                                    // Reuse hitDiff as offsetA and hitDistance for RA2
                                    float RA2 = hitDistance * hitDistance;
                                    vec3 dirA = prevTemporalReservoir.Y.xyz; // Already normalized hitDiff/hitDistance

                                    vec3 offsetB = prev2CurrHitViewPos - prev2CurrNeighborViewPos;
                                    float RB2 = dot(offsetB, offsetB);
                                    vec3 dirB = offsetB * inversesqrt(max(RB2, 1e-12));

                                    // Check if neighbor is essentially the same pixel (skip Jacobian when stationary)
                                    vec3 pixelDiff = prev2CurrNeighborViewPos - viewPos;
                                    float pixelDist2 = dot(pixelDiff, pixelDiff);

                                    float jacobian = 1.0;
                                    // Only apply Jacobian if the neighbor pixel is sufficiently different from current pixel
                                    // Use relative threshold based on hit distance
                                    float threshold = RA2 * 1e-4; // 1% relative distance threshold
                                    if (pixelDist2 > threshold) {
                                        // Cosine at hit point for original and shifted paths
                                        float cosPhiB = -dot(dirB, prevHitNormal);
                                        float cosPhiA = -dot(dirA, prevHitNormal);

                                        // Compute Jacobian: |J| = (r_B^2 * cos(phi_A)) / (r_A^2 * cos(phi_B))
                                        // Only apply if both cosines are positive (valid geometry)
                                        if (cosPhiA > 0.0 && cosPhiB > 5e-2) {
                                            jacobian = (RB2 * cosPhiA) / (RA2 * cosPhiB);
                                        } else if (cosPhiA <= 0.0) {
                                            // Hit point is backfacing from current pixel - invalid
                                            jacobian = 0.0;
                                        }

                                        // Clamp Jacobian to avoid fireflies
                                        jacobian = min(jacobian, 16.0);
                                    }

                                    // Invalidate if current surface is backfacing to the ray
                                    if (dot(gNormal, dirA) <= 0.0) {
                                        jacobian = 0.0;
                                    }

                                    // Apply Jacobian to reservoir weight
                                    prevTemporalReservoir.avgWY *= jacobian;
                                }


                                // TODO: retrace for temporal resampling
                                //                    float prevHitDistance;
                                //                    prevSample = ssgiEvalF(viewPos, gData, prevSampleDirView, prevHitDistance);
                                //                    prevPHat = length(prevSample);
                            } else {
                                prevTemporalReservoir = restir_initReservoir();
                            }
                        } else {
                            vec3 currSampleDirView = normalize(shared_prevViewToCurrView * prevTemporalReservoir.Y.xyz);
                            prevTemporalReservoir.Y.xyz = currSampleDirView;
                        }
                    }
                    temporalReservoir = prevTemporalReservoir;
                }
            }

            float prevPHat = length(prevSample.xyz * prevSample.w);
            float wSum = max(0.0, temporalReservoir.avgWY) * float(temporalReservoir.m) * prevPHat;

            // Re-fetch and fully unpack for material decoding (needed for Spatial / Initial) using fresh registers
            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
            Material material = material_decode(gData);

            // TODO: jacobian and reprojection check
            #if SETTING_GI_SPATIAL_REUSE_FEEDBACK
            GIHistoryData historyData = gi_historyData_init();
            gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));
            const float FEEDBACK_THRESHOLD = float(SETTING_GI_SPATIAL_REUSE_FEEDBACK) / 255.0;
            if (historyData.realHistoryLength < FEEDBACK_THRESHOLD) {
                ReSTIRReservoir prevSpatialReservoir = restir_reservoir_unpack(history_restir_reservoirSpatial_fetch(texelPos));
                prevSpatialReservoir.m = uint(float(prevSpatialReservoir.m) * global_historyResetFactor);

                vec3 prevSpatialSampleDirView = normalize(shared_prevViewToCurrView * prevSpatialReservoir.Y.xyz);
                prevSpatialReservoir.Y.xyz = prevSpatialSampleDirView;

                float prevSpatialHitDistance = prevSpatialReservoir.Y.w;

                vec3 prevSpatialHitViewPos = viewPos + prevSpatialSampleDirView * prevSpatialHitDistance;
                vec3 prevSpatialHitScreenPos = coords_viewToScreen(prevSpatialHitViewPos, global_camProj);
                ivec2 prevSpatialHitTexelPos = ivec2(prevSpatialHitScreenPos.xy * uval_mainImageSize);
                vec3 prevSpatialHitRadiance = restir_irradiance_sampleIrradiance(texelPos, material, prevSpatialHitTexelPos, -prevSpatialSampleDirView);
                float brdf = saturate(dot(gData.normal, prevSpatialSampleDirView)) / PI;
                float prevSpatialPHat = length(prevSpatialHitRadiance * brdf);

                float prevSpatialWi = max(prevSpatialReservoir.avgWY * prevSpatialPHat * float(prevSpatialReservoir.m), 0.0);
                float reservoirRand2 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 456546u)).w);

                if (restir_isReservoirValid(prevSpatialReservoir)) {
                    if (restir_updateReservoir(
                        temporalReservoir,
                        wSum,
                        prevSpatialReservoir.Y,
                        prevSpatialWi,
                        prevSpatialReservoir.m,
                        reservoirRand2
                    )) {
                        prevPHat = prevSpatialPHat;
                        prevSample = vec4(prevSpatialHitRadiance, brdf);
                        GBufferData prevSpatialHitGData = gbufferData_init();
                        gbufferData1_unpack(texelFetch(usam_gbufferData1, prevSpatialHitTexelPos, 0), prevSpatialHitGData);
                        prevHitNormal = prevSpatialHitGData.normal;
                    }
                }
            }
            #endif

            {
                float hitDistance = transient_gi_initialSampleHitDistance_fetch(texelPos).x;
                restir_InitialSampleData initialSample = restir_initalSample_restoreData(texelPos, viewZ, gData.geomNormal, material, hitDistance);
                vec3 hitRadiance = initialSample.hitRadiance;
                vec3 sampleDirView = initialSample.directionAndLength.xyz;

                float brdf = saturate(dot(gData.normal, sampleDirView)) / PI;
                vec3 initalSample = brdf * hitRadiance;

                float samplePdf = brdf;

                float newPHat = length(initalSample);
                float newWi = samplePdf <= 0.0 ? 0.0 : newPHat / samplePdf;

                float reservoirRand1 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 547679546u)).w);

                float reservoirPHat = prevPHat;
                vec4 finalSample = prevSample;
                vec3 hitNormal = prevHitNormal;
                if (restir_updateReservoir(temporalReservoir, wSum, vec4(sampleDirView, hitDistance), newWi, 1u, reservoirRand1)) {
                    reservoirPHat = newPHat;
                    finalSample = vec4(hitRadiance, brdf);

                    vec3 hitViewPos = viewPos + sampleDirView * hitDistance;
                    vec3 hitScreenPos = coords_viewToScreen(hitViewPos, global_camProj);
                    ivec2 hitTexelPos = ivec2(hitScreenPos.xy * uval_mainImageSize);
                    vec4 hitNormalData = transient_viewNormal_fetch(hitTexelPos);
                    hitNormal = normalize(hitNormalData.xyz * 2.0 - 1.0);
                }
                float avgWSum = wSum / float(temporalReservoir.m);
                temporalReservoir.avgWY = reservoirPHat <= 0.0 ? 0.0 : (avgWSum / reservoirPHat);
                temporalReservoir.m = clamp(temporalReservoir.m, 0u, 16u);
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