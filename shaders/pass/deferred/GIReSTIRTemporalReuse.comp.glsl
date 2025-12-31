
#extension GL_KHR_shader_subgroup_ballot : enable

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
#include "/techniques/SSGI.glsl"

uint selectWeighted(vec4 weights, float rand) {
    vec4 prefixSum;
    prefixSum.x = weights.x;
    prefixSum.y = prefixSum.x + weights.y;
    prefixSum.z = prefixSum.y + weights.z;
    prefixSum.w = prefixSum.z + weights.w;

    float total = prefixSum.w;
    float threshold = rand * total;

    vec4 cmp = step(prefixSum, vec4(threshold));
    return uint(dot(cmp, vec4(1.0)));
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    sst_init();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 ssgiOut = vec4(0.0, 0.0, 0.0, -1.0);
        ReSTIRReservoir temporalReservoir = restir_initReservoir(texelPos);
        if (RANDOM_FRAME < MAX_FRAMES && RANDOM_FRAME >= 0) {
            float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(gl_WorkGroupID.xy, 4, texelPos);
            if (viewZ > -65536.0) {
                vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
                vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

                GBufferData gData = gbufferData_init();
                gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
                gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
                Material material = material_decode(gData);

                uvec3 baseRandKey = uvec3(texelPos, RANDOM_FRAME);

                float wSum = 0.0;
                vec4 prevSample = vec4(0.0);
                vec3 prevHitNormal = vec3(0.0);

                {
                    uvec4 reprojInfoData = transient_gi_diffuse_reprojInfo_load(texelPos);
                    ReprojectInfo reprojInfo = reprojectInfo_unpack(reprojInfoData);

                    if (reprojInfo.historyResetFactor > 0.0) {
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
                        //                    vec4 finalWeights = blinearWeights4;
                        vec4 finalWeights = blinearWeights4 * reprojInfo.bilateralWeights;
                        //                    finalWeights = vec4(0.0, 0.0, 0.0, 1.0);

                        float rand = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 987654u)).x);
                        //                    float rand = rand_stbnVec1(texelPos, RANDOM_FRAME);
                        uint selectedIndex = selectWeighted(finalWeights, rand);

                        vec2 selectedTexelPos = gatherTexelPos + sampling_indexToGatherOffset(selectedIndex) * 0.5;
                        ivec2 prevTexelPos = ivec2(selectedTexelPos);

                        ReSTIRReservoir prevTemporalReservoir = restir_reservoir_unpack(history_restir_reservoirTemporal_load(prevTexelPos));
                        prevTemporalReservoir.m = uint(ceil(float(prevTemporalReservoir.m) * global_historyResetFactor * reprojInfo.historyResetFactor));
                        if (restir_isReservoirValid(prevTemporalReservoir)) {
                            vec3 prevHitNormalData = history_restir_prevHitNormal_fetch(prevTexelPos).xyz;
                            prevSample = history_restir_prevSample_load(prevTexelPos);
                            prevHitNormal = normalize(prevHitNormalData * 2.0 - 1.0);
                            prevHitNormal = coords_dir_viewToWorldPrev(prevHitNormal);
                            prevHitNormal = coords_dir_worldToView(prevHitNormal);

                            if (prevTemporalReservoir.Y.w > 0.0) {
                                vec2 prevScreenPos = coords_texelToUV(prevTexelPos, uval_mainImageSizeRcp);
                                float prevViewZ = history_viewZ_fetch(prevTexelPos).x;
                                vec3 prevViewPos = coords_toViewCoord(prevScreenPos, prevViewZ, global_prevCamProjInverse);

                                vec3 prevHitViewPos = prevViewPos + prevTemporalReservoir.Y.xyz * prevTemporalReservoir.Y.w;
                                vec3 prevHitScenePos = coords_pos_viewToWorld(prevHitViewPos, gbufferPrevModelViewInverse);
                                vec3 prev2CurrHitScenePos = coord_scenePrevToCurr(prevHitScenePos);
                                vec3 prev2CurrHitViewPos = coords_pos_worldToView(prev2CurrHitScenePos, gbufferModelView);
                                vec3 hitDiff = prev2CurrHitViewPos - viewPos;
                                float hitDistance = length(hitDiff);

                                prevTemporalReservoir.Y.xyz = hitDiff / hitDistance;
                                prevTemporalReservoir.Y.w = hitDistance;

                                float brdf = saturate(dot(gData.normal, prevTemporalReservoir.Y.xyz)) / PI;

                                vec4 prev2CurrHitClipPos = global_camProj * vec4(prev2CurrHitViewPos, 1.0);
                                uint clipFlag = uint(prev2CurrHitClipPos.z > 0.0);
                                clipFlag &= uint(all(lessThan(abs(prev2CurrHitClipPos.xy), prev2CurrHitClipPos.ww)));
                                vec3 prev2CurrHitScreenPos = coords_viewToScreen(prev2CurrHitViewPos, global_camProj);
                                clipFlag &= uint(saturate(prev2CurrHitScreenPos) == prev2CurrHitScreenPos);
                                ivec2 prevCurrHitTexelPos = ivec2(prev2CurrHitScreenPos.xy * uval_mainImageSize);

                                if (bool(clipFlag)) {
                                    vec3 prevHitRadiance = sampleIrradiance(texelPos, prevCurrHitTexelPos, -prevTemporalReservoir.Y.xyz);

                                    // Jacobian correction for reconnection shift
                                    {
                                        vec3 prevHitScreenPos = coords_viewToScreen(prevHitViewPos, global_prevCamProj);
                                        ivec2 prevHitTexelPos = ivec2(prevHitScreenPos.xy * uval_mainImageSize);

                                        // Original path: from the temporal neighbor pixel (where the sample came from) to hit point
                                        // prev2CurrNeighborViewPos is the neighbor's position transformed to current frame coordinates
                                        vec3 prevNeighborScenePos = coords_pos_viewToWorld(prevViewPos, gbufferPrevModelViewInverse);
                                        vec3 prev2CurrNeighborScenePos = coord_scenePrevToCurr(prevNeighborScenePos);
                                        vec3 prev2CurrNeighborViewPos = coords_pos_worldToView(prev2CurrNeighborScenePos, gbufferModelView);

                                        // Vector from current pixel to hit point (shifted path) - this is the new path
                                        vec3 offsetA = prev2CurrHitViewPos - viewPos;
                                        float RA2 = dot(offsetA, offsetA);
                                        vec3 dirA = offsetA / max(sqrt(RA2), 1e-6);

                                        vec3 offsetB = prev2CurrHitViewPos - prev2CurrNeighborViewPos;
                                        float RB2 = dot(offsetB, offsetB);
                                        vec3 dirB = offsetB / max(sqrt(RB2), 1e-6);

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
                                            if (cosPhiA > 0.0 && cosPhiB > 0.0 && RA2 > 0.0) {
                                                jacobian = (RB2 * cosPhiA) / (RA2 * cosPhiB);
                                            } else if (cosPhiA <= 0.0) {
                                                // Hit point is backfacing from current pixel - invalid
                                                jacobian = 0.0;
                                            }

                                            // Clamp Jacobian to avoid fireflies
                                            const float maxJacobian = 16.0;
                                            jacobian = min(jacobian, maxJacobian);
                                        }

                                        // Invalidate if current surface is backfacing to the ray
                                        if (dot(gData.normal, dirA) <= 0.0) {
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
                                    prevTemporalReservoir = restir_initReservoir(texelPos);
                                }
                            } else {
                                vec3 prevSampleDirWorld = coords_dir_viewToWorldPrev(prevTemporalReservoir.Y.xyz);
                                vec3 currSampleDirView = coords_dir_worldToView(prevSampleDirWorld);
                                prevTemporalReservoir.Y.xyz = currSampleDirView;
                                float brdfMiss = saturate(dot(gData.normal, currSampleDirView)) / PI;
                                vec3 prevHitRadiance = sampleIrradianceMiss(prevSampleDirWorld);
                            }
                        }
                        temporalReservoir = prevTemporalReservoir;
                    }
                }

                float prevPHat = length(prevSample.xyz * prevSample.w);
                wSum = max(0.0, temporalReservoir.avgWY) * float(temporalReservoir.m) * prevPHat;

                #if SPATIAL_REUSE_FEEDBACK
                GIHistoryData historyData = gi_historyData_init();
                gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));
                const float FEEDBACK_THRESHOLD = float(SPATIAL_REUSE_FEEDBACK) / 255.0;
                if (historyData.realHistoryLength < FEEDBACK_THRESHOLD) {
                    ReSTIRReservoir prevSpatialReservoir = restir_reservoir_unpack(history_restir_reservoirSpatial_load(texelPos));
                    prevSpatialReservoir.m = uint(float(prevSpatialReservoir.m) * global_historyResetFactor);

                    vec3 prevSpatialSampleDirView = prevSpatialReservoir.Y.xyz;
                    prevSpatialSampleDirView = coords_dir_viewToWorldPrev(prevSpatialSampleDirView);
                    prevSpatialSampleDirView = coords_dir_worldToView(prevSpatialSampleDirView);
                    prevSpatialReservoir.Y.xyz = prevSpatialSampleDirView;

                    float prevSpatialHitDistance = prevSpatialReservoir.Y.w;

                    vec3 prevSpatialHitViewPos = viewPos + prevSpatialSampleDirView * prevSpatialHitDistance;
                    vec3 prevSpatialHitScreenPos = coords_viewToScreen(prevSpatialHitViewPos, global_camProj);
                    ivec2 prevSpatialHitTexelPos = ivec2(prevSpatialHitScreenPos.xy * uval_mainImageSize);
                    vec3 prevSpatialHitRadiance = sampleIrradiance(texelPos, prevSpatialHitTexelPos, -prevSpatialSampleDirView);
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
                            prevSpatialReservoir.age,
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

                const float RESET_START = 64.0;
                const float RESET_END = 256.0;
                float resetThreshold = linearStep(RESET_START, RESET_END, float(temporalReservoir.age));
                float resetRand = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 123679546u)).w);
                if (resetRand < resetThreshold) {
                    temporalReservoir = restir_initReservoir(texelPos);
                }

                {
                    //                    vec2 rand2 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 12312745u)).zw);
                    //                    vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
                    //                    float samplePdf = sampleDirTangentAndPdf.w;
                    //                    vec3 sampleDirView = normalize(material.tbn * sampleDirTangentAndPdf.xyz);
                    //                    vec4 ssgiData = uintBitsToFloat(imageLoad(uimg_csrgba32ui, csrgba32ui_temp4_texelToTexel(texelPos)));
                    //                    float hitDistance = ssgiData.w;
                    //                    vec3 initalSample = ssgiData.xyz;


                    //                    InitialSampleData initialSample = initialSampleData_init();
                    //                    {
                    //                        uvec3 baseRandKey = uvec3(texelPos, RANDOM_FRAME);
                    //                        vec2 rand2 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 12312745u)).zw);
                    //                        //                vec2 rand2 = rand_stbnVec2(texelPos, RANDOM_FRAME);
                    //                        //                vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
                    //                        vec4 sampleDirTangentAndPdf = rand_sampleInHemisphere(rand2);
                    //                        vec3 sampleDirView = normalize(material.tbn * sampleDirTangentAndPdf.xyz);
                    //
                    //                        //                ivec2 stbnPos = texelPos + ivec2(rand_r2Seq2(RANDOM_FRAME / 64u) * vec2(128, 128));
                    //                        //                vec3 sampleDirTangent = rand_stbnUnitVec3Cosine(stbnPos, RANDOM_FRAME);
                    //                        //                vec3 sampleDirView = normalize(material.tbn * sampleDirTangent);
                    //
                    //                        vec4 ssgiOut = vec4(0.0);
                    //                        //                sharedData[gl_LocalInvocationIndex] = sampleDirView;
                    //                        vec4 resultStuff = ssgiEvalF2(viewPos, sampleDirView);
                    //
                    //                        initialSample.hitRadiance = resultStuff.xyz;
                    //                        initialSample.directionAndLength.xyz = sampleDirView;
                    //                        initialSample.directionAndLength.w = resultStuff.w;
                    //                    }

                    InitialSampleData initialSample = initialSampleData_unpack(transient_restir_initialSample_load(texelPos));
                    vec3 hitRadiance = initialSample.hitRadiance;
                    vec3 sampleDirView = initialSample.directionAndLength.xyz;
                    float hitDistance = initialSample.directionAndLength.w;


                    float brdf = saturate(dot(gData.normal, sampleDirView)) / PI;
                    vec3 initalSample = brdf * hitRadiance;

                    //                    float samplePdf = saturate(dot(gData.normal, sampleDirView)) / PI;
                    float samplePdf = brdf;
                    //                    float samplePdf = 1.0 / (2.0 * PI);

                    float newPHat = length(initalSample);
                    float newWi = samplePdf <= 0.0 ? 0.0 : newPHat / samplePdf;

                    float reservoirRand1 = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 547679546u)).w);

                    float reservoirPHat = prevPHat;
                    vec4 finalSample = prevSample;
                    vec3 hitNormal = prevHitNormal;
                    if (restir_updateReservoir(temporalReservoir, wSum, vec4(sampleDirView, hitDistance), newWi, 1u, 0u, reservoirRand1)) {
                        reservoirPHat = newPHat;
                        finalSample = vec4(hitRadiance, brdf);

                        vec3 hitViewPos = viewPos + sampleDirView * hitDistance;
                        vec3 hitScreenPos = coords_viewToScreen(hitViewPos, global_camProj);
                        ivec2 hitTexelPos = ivec2(hitScreenPos.xy * uval_mainImageSize);
                        GBufferData hitGData = gbufferData_init();
                        gbufferData1_unpack(texelFetch(usam_gbufferData1, hitTexelPos, 0), hitGData);
                        hitNormal = hitGData.normal;
                    }
                    float avgWSum = wSum / float(temporalReservoir.m);
                    temporalReservoir.avgWY = reservoirPHat <= 0.0 ? 0.0 : (avgWSum / reservoirPHat);
                        temporalReservoir.m = clamp(temporalReservoir.m, 0u, 16u);
                    ssgiOut = vec4(finalSample.xyz * finalSample.w * temporalReservoir.avgWY, temporalReservoir.Y.w);
                    #if USE_REFERENCE
                    ssgiOut = vec4(initalSample / samplePdf, hitDistance);
                    #endif

                    SpatialSampleData spatialSample = spatialSampleData_init();
                    spatialSample.sampleValue = finalSample;
                    spatialSample.geomNormal = gData.geomNormal;
                    spatialSample.normal = gData.normal;
                    spatialSample.hitNormal = hitNormal;
                    transient_restir_spatialInput_store(texelPos, spatialSampleData_pack(spatialSample));

                    temporalReservoir.age++;
                }
            }
        }
        ssgiOut.rgb = clamp(ssgiOut.rgb, 0.0, FP16_MAX);
        transient_ssgiOut_store(texelPos, ssgiOut);
        transient_restir_reservoirReprojected_store(texelPos, restir_reservoir_pack(temporalReservoir));
    }
}