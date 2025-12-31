
#extension GL_KHR_shader_subgroup_ballot : enable

#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"
#include "/util/Mat2.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
#include "/techniques/SSGI.glsl"

#if USE_REFERENCE || !SPATIAL_REUSE
void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        SpatialSampleData sampleData = spatialSampleData_unpack(transient_restir_spatialInput_load(texelPos));
        history_restir_prevSample_store(texelPos, sampleData.sampleValue);
        history_restir_prevHitNormal_store(texelPos, vec4(sampleData.hitNormal * 0.5 + 0.5, 0.0));

        if (RANDOM_FRAME < MAX_FRAMES && RANDOM_FRAME >= 0) {
            uvec4 reprojectedData = transient_restir_reservoirReprojected_load(texelPos);
            history_restir_reservoirTemporal_store(texelPos, reprojectedData);
        }
    }
}
#else
void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    sst_init();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        SpatialSampleData sampleData = spatialSampleData_unpack(transient_restir_spatialInput_load(texelPos));
        history_restir_prevSample_store(texelPos, sampleData.sampleValue);
        history_restir_prevHitNormal_store(texelPos, vec4(sampleData.hitNormal * 0.5 + 0.5, 0.0));
        if (RANDOM_FRAME < MAX_FRAMES && RANDOM_FRAME >= 0){
            float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos, 4, texelPos);
            if (viewZ > -65536.0) {
                vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
                vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

                //                GBufferData gData = gbufferData_init();
                //                gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
                //                gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

                SpatialSampleData centerSampleData = spatialSampleData_unpack(transient_restir_spatialInput_load(texelPos));

                uvec3 baseRandKey = uvec3(texelPos, RANDOM_FRAME);

                uvec4 reprojectedData = transient_restir_reservoirReprojected_load(texelPos);
                ReSTIRReservoir spatialReservoir = restir_reservoir_unpack(reprojectedData);
                history_restir_reservoirTemporal_store(texelPos, reprojectedData);

                GIHistoryData historyData = gi_historyData_init();
                gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));

                #if SETTING_GI_COHERENCE_OPTIMIZATION
                const uint reuseCount = uint(SPATIAL_REUSE_SAMPLES);
                #else
                const uint reuseCount = uint(mix(float(SPATIAL_REUSE_SAMPLES), 1.0, sqrt(linearStep(0.0, 0.5, historyData.realHistoryLength))));
                #endif
                const float REUSE_RADIUS = float(SPATIAL_REUSE_RADIUS);
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

                    //                    GBufferData sampleGData = gbufferData_init();
                    //                    gbufferData1_unpack(texelFetch(usam_gbufferData1, sampleTexelPos, 0), sampleGData);

                    SpatialSampleData neighborData = spatialSampleData_unpack(transient_restir_spatialInput_load(sampleTexelPos));

                    if (dot(centerSampleData.geomNormal, neighborData.geomNormal) < 0.99) {
                        continue;
                    }
                    float neighborViewZ = texelFetch(usam_gbufferViewZ, sampleTexelPos, 0).x;
                    ReSTIRReservoir neighborReservoir = restir_reservoir_unpack(transient_restir_reservoirReprojected_load(sampleTexelPos));
                    vec2 neighborScreenPos = sampleTexelPosF * uval_mainImageSizeRcp;
                    vec3 neighborViewPos = coords_toViewCoord(neighborScreenPos, neighborViewZ, global_camProjInverse);

                    if (restir_isReservoirValid(neighborReservoir)) {
                        //                        ivec2 neighborHitTexelPos = ivec2(neighborReservoir.Y);
                        //                        float neighborHitViewZ = texelFetch(usam_gbufferViewZ, neighborHitTexelPos, 0).x;
                        //                        vec2 neighborHitScreenPos = coords_texelToUV(neighborHitTexelPos, uval_mainImageSizeRcp);
                        //                        vec3 neighborHitViewPos = coords_toViewCoord(neighborHitScreenPos, neighborHitViewZ, global_camProjInverse);

                        vec3 neighborSampleDirView = neighborReservoir.Y.xyz;
                        float neighborSampleHitDistance = neighborReservoir.Y.w;
                        vec3 neighborHitViewPos = neighborViewPos + neighborSampleDirView * neighborReservoir.Y.w;
                        vec3 hitDiff = neighborHitViewPos - viewPos;
                        neighborSampleHitDistance = length(hitDiff);
                        neighborSampleDirView = hitDiff / neighborSampleHitDistance;

                        float neighborSamplePdf = saturate(dot(centerSampleData.normal, neighborSampleDirView)) / PI;
                        //                        float neighborSamplePdf = 1.0 / (2.0 * PI);

                        //                        float newHitDistance;
                        //                        vec3 neighborSample = ssgiEvalF(viewPos, gData, neighborSampleDirView, newHitDistance);
                        //                        float neighborPHat = length(neighborSample);
                        //                        if (neighborPHat <= 0.0){
                        //                            continue;
                        //                        }
                        //                        vec3 newHitViewPos = viewPos + neighborSampleDirView * newHitDistance;
                        //                        neighborHitViewPos = newHitViewPos;
                        //                           neighborSampleHitDistance = newHitDistance;
                        //
                        vec3 neighborHitScreenPos = coords_viewToScreen(neighborHitViewPos, global_camProj);
                        ivec2 neighborHitTexelPos = ivec2(neighborHitScreenPos.xy * uval_mainImageSize);
                        //
                        vec3 hitRadiance = neighborData.sampleValue.xyz;
                        float brdf = saturate(dot(centerSampleData.normal, neighborSampleDirView)) / PI;
                        vec3 f = brdf * hitRadiance;
                        vec3 neighborSample = f;
                        float neighborPHat = length(neighborSample);


                        //                        if (distance(newHitViewPos, neighborHitViewPos) > 0.01) {
                        //                            neighborPHat = 0.0;
                        //                        }

                        //                            vec3 hitRadiance = texture(usam_temp2, neighborHitScreenPos.xy).rgb;
                        //                            float brdf = saturate(dot(centerSampleData.normal, neighborSampleDirView)) / PI;
                        //                            vec3 f = brdf * hitRadiance;
                        //                            float neighborPHat = length(f);


                        //                    // Calculate target function.
                        //                    float3 offsetB = neighborReservoir.position - neighborReservoir.creationPoint;
                        //                    float3 offsetA = neighborReservoir.position - worldPosition;
                        //                    float pNewTN = evalTargetFunction(neighborReservoir.radiance, worldNormal, worldPosition, neighborReservoir.position, evalContext);
                        //                    // Discard back-face.
                        //                    if (dot(worldNormal, offsetA) <= 0.f)
                        //                    {
                        //                        pNewTN = 0.f;
                        //                    }
                        //
                        //                    float RB2 = dot(offsetB, offsetB);
                        //                    float RA2 = dot(offsetA, offsetA);
                        //                    offsetB = normalize(offsetB);
                        //                    offsetA = normalize(offsetA);
                        //                    float cosA = dot(worldNormal, offsetA);
                        //                    float cosB = dot(neighborReservoir.creationNormal, offsetB);
                        //                    float cosPhiA = -dot(offsetA, neighborReservoir.normal);
                        //                    float cosPhiB = -dot(offsetB, neighborReservoir.normal);
                        //                    if (cosB <= 0.f || cosPhiB <= 0.f)
                        //                    {
                        //                        continue;
                        //                    }
                        //                    if (cosA <= 0.f || cosPhiA <= 0.f || RA2 <= 0.f || RB2 <= 0.f)
                        //                    {
                        //                        pNewTN = 0.f;
                        //                    }
                        //
                        //                    bool isVisible = evalSegmentVisibility(computeRayOrigin(worldPosition, worldNormal), neighborReservoir.position);
                        //                    if (!isVisible)
                        //                    {
                        //                        pNewTN = 0.f;
                        //                    }
                        //                        // Calculate Jacobian determinant and weight.
                        //                        const float maxJacobian = enableJacobianClamping ? jacobianClampThreshold : largeFloat;
                        //                        float jacobian = RA2 * cosPhiB <= 0.f ? 0.f : clamp(RB2 * cosPhiA / (RA2 * cosPhiB), 0.f, maxJacobian);
                        //                        float wiTN = clamp(neighborReservoir.avgWeight * pNewTN * neighborReservoir.M * jacobian, 0.f, largeFloat);
                        //
                        //                        // Conditionally update spatial reservoir.
                        //                        bool isUpdated = updateReservoir(wiTN, neighborReservoir, sg, wSumS, spatialReservoir);
                        //                        if (isUpdated) reuseID = nReuse;


                        vec3 offsetB = neighborHitViewPos - neighborViewPos;
                        vec3 offsetA = neighborHitViewPos - viewPos;

                        if (dot(centerSampleData.normal, offsetA) <= 0.0) {
                            neighborPHat = 0.0;
                        }

                        float RB2 = dot(offsetB, offsetB);
                        float RA2 = dot(offsetA, offsetA);
                        offsetB = normalize(offsetB);
                        offsetA = normalize(offsetA);
                        float cosA = dot(centerSampleData.normal, offsetA);
                        float cosB = dot(neighborData.normal, offsetB);

                        //                        GBufferData hitGData = gbufferData_init();
                        //                        gbufferData1_unpack(texelFetch(usam_gbufferData1, neighborHitTexelPos, 0), hitGData);

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
                        //                        float neighborWi = max(neighborReservoir.avgWY * neighborPHat * float(neighborReservoir.m) * jacobian, 0.0);

                        //                        float neighborWi = max(neighborReservoir.avgWY * neighborPHat * float(neighborReservoir.m) , 0.0);



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
                #if SPATIAL_REUSE_VISIBILITY_TRACE
                float avgWSum = spatialWSum / float(spatialReservoir.m);
                resultReservoir.avgWY = selectedSampleF.w <= 0.0 ? 0.0 : (avgWSum / selectedSampleF.w);
                ssgiOut = vec4(selectedSampleF.xyz * resultReservoir.avgWY, resultReservoir.Y.w);
                if (any(notEqual(selectedSampleF, originalSample))) {
                    SSTResult sstResult = sst_trace(viewPos, spatialReservoir.Y.xyz, SETTING_GI_SST_THICKNESS);

                    if (sstResult.hit) {
                        vec2 actualHitTexelPosF = floor(sstResult.hitScreenPos.xy * uval_mainImageSize);
                        vec2 actualHitTexelCenter = actualHitTexelPosF + 0.5;
                        vec2 acutualRoundedHitScreenPos = actualHitTexelCenter * uval_mainImageSizeRcp;
                        float actualHitViewZ = coords_reversedZToViewZ(sstResult.hitScreenPos.z, near);
                        vec3 actualHitViewPos = coords_toViewCoord(acutualRoundedHitScreenPos, actualHitViewZ, global_camProjInverse);

                        vec3 expectHitViewPos = viewPos + spatialReservoir.Y.xyz * spatialReservoir.Y.w;

                        vec3 hitDiff = actualHitViewPos - expectHitViewPos;

                        if (dot(hitDiff, hitDiff) > 0.5) {
                            ssgiOut = vec4(0.0, 0.0, 0.0, -1.0);
                            resultReservoir = restir_initReservoir(texelPos);
                        }
                    } else {
                        ssgiOut = vec4(0.0, 0.0, 0.0, -1.0);
                        resultReservoir = restir_initReservoir(texelPos);
                    }
                }
                #else
                float avgWSum = spatialWSum / float(spatialReservoir.m);
                spatialReservoir.avgWY = selectedSampleF.w <= 0.0 ? 0.0 : (avgWSum / selectedSampleF.w);
                ssgiOut = vec4(selectedSampleF.xyz * spatialReservoir.avgWY, spatialReservoir.Y.w);
                resultReservoir = spatialReservoir;
                #endif
                const uint SPATIAL_REUSE_MAX_M = 1u;
                resultReservoir.m = clamp(resultReservoir.m, 0u, SPATIAL_REUSE_MAX_M);

                history_restir_reservoirSpatial_store(texelPos, restir_reservoir_pack(resultReservoir));

                ssgiOut.rgb = clamp(ssgiOut.rgb, 0.0, FP16_MAX);
                transient_ssgiOut_store(texelPos, ssgiOut);
            }
        }
    }
}
#endif