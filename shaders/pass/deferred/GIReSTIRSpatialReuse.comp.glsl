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

#include "/techniques/SST2.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/techniques/gi/Reservoir.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"
#include "/util/Mat2.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(std430, binding = 4) buffer RayData {
    uvec4 ssbo_rayData[];
};

layout(std430, binding = 5) buffer RayDataIndices {
    uint ssbo_rayDataIndices[];
};

layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(rgba8) uniform restrict writeonly image2D uimg_temp5;

#if USE_REFERENCE || !defined(SETTING_GI_SPATIAL_REUSE)
void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        SpatialSampleData sampleData = spatialSampleData_unpack(transient_restir_spatialInput_fetch(texelPos));
        history_restir_prevSample_store(texelPos, sampleData.sampleValue);
        history_restir_prevHitNormal_store(texelPos, vec4(sampleData.hitNormal * 0.5 + 0.5, 0.0));
    }
}
#else
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
        SpatialSampleData sampleData = spatialSampleData_unpack(transient_restir_spatialInput_fetch(texelPos));
        history_restir_prevSample_store(texelPos, sampleData.sampleValue);
        history_restir_prevHitNormal_store(texelPos, vec4(sampleData.hitNormal * 0.5 + 0.5, 0.0));
        #if SETTING_DEBUG_OUTPUT
        if (RANDOM_FRAME < MAX_FRAMES && RANDOM_FRAME >= 0){
            #endif
            float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos, 4, texelPos);
            if (viewZ > -65536.0) {
                vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
                vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

                SpatialSampleData centerSampleData = spatialSampleData_unpack(transient_restir_spatialInput_fetch(texelPos));

                uvec3 baseRandKey = uvec3(texelPos, RANDOM_FRAME);

                uvec4 reprojectedData;
                if (bool(frameCounter & 1)) {
                    reprojectedData = history_restir_reservoirTemporal1_fetch(texelPos);
                } else {
                    reprojectedData = history_restir_reservoirTemporal2_fetch(texelPos);
                }
                ReSTIRReservoir spatialReservoir = restir_reservoir_unpack(reprojectedData);

                GIHistoryData historyData = gi_historyData_init();
                gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));

                #ifdef SETTING_GI_SPATIAL_REUSE_COUNT_DYNAMIC
                const uint reuseCount = uint(mix(float(SETTING_GI_SPATIAL_REUSE_COUNT), 1.0, sqrt(linearStep(0.0, 0.5, historyData.realHistoryLength))));
                #else
                const uint reuseCount = uint(SETTING_GI_SPATIAL_REUSE_COUNT);
                #endif
                const float REUSE_RADIUS = float(SETTING_GI_SPATIAL_REUSE_RADIUS);
                vec2 texelPosF = vec2(texelPos) + vec2(0.5);

                float pHatMe = 0.0;
                vec4 originalSample = vec4(0.0);
                {
                    vec3 sampleDirView = spatialReservoir.Y.xyz;
                    vec3 hitViewPos = viewPos + sampleDirView * spatialReservoir.Y.w;
                    vec3 hitScreenPos = coords_viewToScreen(hitViewPos, global_camProj);
                    ivec2 hitTexelPos = ivec2(hitScreenPos.xy * uval_mainImageSize);

                    vec3 hitRadiance = centerSampleData.sampleValue.xyz;

                    float brdf = centerSampleData.sampleValue.w;
                    vec3 f = brdf * hitRadiance;
                    pHatMe = length(f);
                    originalSample = vec4(f, pHatMe);
                }
                float spatialWSum = max(spatialReservoir.avgWY, 0.0) * pHatMe * float(spatialReservoir.m);


                vec4 selectedSampleF = originalSample;

                vec2 noise2 = rand_stbnVec2(texelPos, RANDOM_FRAME);
                float angle = noise2.x * PI_2;
                vec2 rot = vec2(cos(angle), sin(angle));
                float rSteps = 1.0 / float(reuseCount);

                for (uint i = 0u; i < reuseCount; ++i) {
                    rot *= MAT2_GOLDEN_ANGLE;
                    //                    float radius = sqrt((float(i) + noise2.y) * rSteps) * REUSE_RADIUS;
                    float radius = ((float(i) + noise2.y) * rSteps) * REUSE_RADIUS;
                    vec2 offset = rot * radius;

                    vec2 sampleTexelPosF = texelPosF + offset;
                    if (clamp(sampleTexelPosF, vec2(0.0), uval_mainImageSizeI - 1.0) != sampleTexelPosF) {
                        continue;
                    }
                    ivec2 sampleTexelPos = ivec2(sampleTexelPosF);

                    if (sampleTexelPos == texelPos) {
                        continue;
                    }

                    SpatialSampleData neighborData = spatialSampleData_unpack(transient_restir_spatialInput_fetch(sampleTexelPos));

                    if (dot(centerSampleData.geomNormal, neighborData.geomNormal) < 0.99) {
                        continue;
                    }
                    float neighborViewZ = texelFetch(usam_gbufferViewZ, sampleTexelPos, 0).x;
                    uvec4 neighborReservoirData;
                    if (bool(frameCounter & 1)) {
                        neighborReservoirData = history_restir_reservoirTemporal1_fetch(sampleTexelPos);
                    } else {
                        neighborReservoirData = history_restir_reservoirTemporal2_fetch(sampleTexelPos);
                    }
                    ReSTIRReservoir neighborReservoir = restir_reservoir_unpack(neighborReservoirData);
                    vec2 neighborScreenPos = sampleTexelPosF * uval_mainImageSizeRcp;
                    vec3 neighborViewPos = coords_toViewCoord(neighborScreenPos, neighborViewZ, global_camProjInverse);

                    if (restir_isReservoirValid(neighborReservoir)) {
                        vec3 neighborSampleDirView = neighborReservoir.Y.xyz;
                        float neighborSampleHitDistance = neighborReservoir.Y.w;
                        vec3 neighborHitViewPos = neighborViewPos + neighborSampleDirView * neighborReservoir.Y.w;
                        vec3 hitDiff = neighborHitViewPos - viewPos;
                        float hitDist2 = dot(hitDiff, hitDiff);

                        // Safety check: Avoid singularity if reuse sample is at the exact same position
                        if (hitDist2 < 1e-6) continue;

                        neighborSampleHitDistance = sqrt(hitDist2);
                        neighborSampleDirView = hitDiff / neighborSampleHitDistance;

                        vec3 neighborHitScreenPos = coords_viewToScreen(neighborHitViewPos, global_camProj);
                        ivec2 neighborHitTexelPos = ivec2(neighborHitScreenPos.xy * uval_mainImageSize);
                        //
                        vec3 hitRadiance = neighborData.sampleValue.xyz;
                        float brdf = saturate(dot(centerSampleData.normal, neighborSampleDirView)) / PI;
                        vec3 f = brdf * hitRadiance;
                        vec3 neighborSample = f;
                        float neighborPHat = length(neighborSample);

                        vec3 offsetB = neighborHitViewPos - neighborViewPos;
                        vec3 offsetA = hitDiff;

                        if (dot(centerSampleData.normal, offsetA) <= 0.0) {
                            neighborPHat = 0.0;
                        }

                        float RB2 = dot(offsetB, offsetB);
                        float RA2 = hitDist2;

                        if (RB2 < 1e-6) continue;

                        offsetB = normalize(offsetB);
                        offsetA = normalize(offsetA);
                        float cosA = dot(centerSampleData.normal, offsetA);
                        float cosB = dot(neighborData.normal, offsetB);

                        float cosPhiA = -dot(offsetA, neighborData.hitNormal);
                        float cosPhiB = -dot(offsetB, neighborData.hitNormal);
                        if (cosB <= 0.0 || cosPhiB <= 0.0) {
                            continue;
                        }
                        if (cosA <= 0.0 || cosPhiA <= 0.0 || RA2 <= 0.0 || RB2 <= 0.0) {
                            neighborPHat = 0.0;
                        }

                        float maxJacobian = 100.0;
                        float jacobian = RA2 * cosPhiB <= 0.0 ? 0.0 : (RB2 * cosPhiA) / (RA2 * cosPhiB);
                        if (neighborPHat <= 0.0) {
                            neighborReservoir.m = 0u;
                        }
                        if (jacobian <= 0.0) {
                            neighborReservoir.m = 0u;
                        }
                        jacobian = clamp(jacobian, 0.0, maxJacobian);

                        float neighborWi = max(neighborReservoir.avgWY, 0.0) * neighborPHat * float(neighborReservoir.m) * jacobian;

                        float neighborRand = hash_uintToFloat(hash_44_q3(uvec4(baseRandKey, 2u + i)).x);

                        if (restir_updateReservoir(
                            spatialReservoir,
                            spatialWSum,
                            vec4(neighborSampleDirView, neighborSampleHitDistance),
                            neighborWi,
                            neighborReservoir.m,
                            neighborReservoir.age,
                            neighborRand
                        )) {
                            selectedSampleF = vec4(neighborSample, neighborPHat);
                        }
                    }
                }

                vec4 ssgiOut = vec4(0.0, 0.0, 0.0, -1.0);
                ReSTIRReservoir resultReservoir = spatialReservoir;
                float avgWSum = spatialWSum / float(spatialReservoir.m);
                resultReservoir.avgWY = selectedSampleF.w <= 0.0 ? 0.0 : (avgWSum / selectedSampleF.w);
                ssgiOut = vec4(selectedSampleF.xyz * resultReservoir.avgWY, resultReservoir.Y.w);
                #if SETTING_DEBUG_OUTPUT
                vec4 vvv = vec4(0.0);
                #endif
                if (any(notEqual(selectedSampleF, originalSample))) {
                    #if SETTING_DEBUG_OUTPUT
                    vvv = vec4(0.0, 1.0, 0.0, 0.0);
                    #endif

                    SSTRay sstRay;
                    if (spatialReservoir.Y.w > 0.0) {
                        vec3 expectHitViewPos = viewPos + spatialReservoir.Y.xyz * spatialReservoir.Y.w;
                        vec3 rayOrigin = coords_viewToScreen(viewPos, global_camProj);
                        vec3 rayEnd = coords_viewToScreen(expectHitViewPos, global_camProj);
                        vec4 rayDirLen = normalizeAndLength(rayEnd - rayOrigin);
                        sstRay = sstray_setup(texelPos, rayOrigin, rayDirLen.xyz, rayDirLen.w);
                    } else {
                        sstRay = sstray_setup(texelPos, viewPos, spatialReservoir.Y.xyz);
                    }
                    sst_trace(sstRay, 4);
                    if (sstRay.currT > 0.0) {
                        uvec4 packedData = sstray_pack(sstRay);
                        ssbo_rayData[dataIndex] = packedData;
                        rayIndex = sst2_encodeRayIndexBits(binLocalIndex, sstRay);
                    } else {
                        bool discardSptialReuse = true;
                        if (sstRay.currT < -0.99) {
                            discardSptialReuse = false;
                        }

                        if (discardSptialReuse) {
                            resultReservoir = restir_initReservoir();
                            ssgiOut = vec4(0.0);
                            #if SETTING_DEBUG_OUTPUT
                            imageStore(uimg_temp5, texelPos, vec4(0.0, 0.0, 1.0, 0.0));
                            #endif
                        }
                    }
                }
                #if SETTING_DEBUG_OUTPUT
                imageStore(uimg_temp5, texelPos, vvv);
                #endif

                const uint SPATIAL_REUSE_MAX_M = 1u;
                resultReservoir.m = clamp(resultReservoir.m, 0u, SPATIAL_REUSE_MAX_M);
                history_restir_reservoirSpatial_store(texelPos, restir_reservoir_pack(resultReservoir));

                ssgiOut.rgb = clamp(ssgiOut.rgb, 0.0, FP16_MAX);
                transient_ssgiOut_store(texelPos, ssgiOut);
            }
            #if SETTING_DEBUG_OUTPUT
        }
        #endif
    }
    ssbo_rayDataIndices[dataIndex] = rayIndex;
}
#endif