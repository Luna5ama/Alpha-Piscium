#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 16, local_size_y = 16) in;

#include "/util/Material.glsl"
#include "/util/ThreadGroupTiling.glsl"
#include "/techniques/SST2.glsl"
#include "/techniques/gi/Common.glsl"
#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/ReservoirSplat.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/techniques/gi/PairwiseMISMetadata.glsl"
#include "/techniques/voxel/VoxelTrace.glsl"

const vec2 workGroupsRender = vec2(1.0, 1.0);

//layout(std430, binding = 5) buffer RayData {
//    uvec4 ssbo_rayData[];
//};
//
//layout(std430, binding = 6) buffer RayIndexData {
//    uint ssbo_rayDataIndices[];
//};

layout(rgba16f) uniform image2D uimg_rgba16f;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(r32f) uniform image2D uimg_r32f;
layout(r32ui) uniform restrict writeonly uimage2D uimg_r32ui;
layout(rgba8) uniform restrict writeonly image2D uimg_temp5;

shared uint shared_rayCount[16];

ReSTIRReservoir readTemporalReservoir(ivec2 texelPos) {
    uvec4 reprojectedData;
    if (bool(frameCounter & 1)) {
        reprojectedData = history_restir_reservoirTemporal1_fetch(texelPos);
    } else {
        reprojectedData = history_restir_reservoirTemporal2_fetch(texelPos);
    }
    return restir_reservoir_unpack(reprojectedData);
}

void main() {
    voxel_initShared();
    sst_init(SETTING_GI_SST_THICKNESS);
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    uvec2 binId = swizzledWGPos >> 1u;
    uint numBinX = (uval_mainImageSizeI.x + 31) >> 5;
    uint binIdx = binId.y * numBinX + binId.x;
    ivec2 binLocalPos = texelPos & 31;
    uint binLocalIndex = sst2_encodeBinLocalIndex(binLocalPos);
    uint binWriteBaseIndex = binIdx * 1024;
    uint dataIndex = binWriteBaseIndex + binLocalIndex;
    uint rayIndex = 0xFFFFFFFFu;

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        uint packedPrimary = restir_splatFetchCurrentPrimary(texelPos);

        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos, 4, texelPos);

        if (viewZ > -65536.0) {
            SpatialSampleData centerSampleData = spatialSampleData_unpack(transient_restir_spatialInput_fetch(texelPos));
            centerSampleData.normal = normalize(transient_viewNormal_fetch(texelPos).xyz * 2.0 - 1.0);
            history_restir_prevSample_store(texelPos, centerSampleData.sampleValue);
            history_restir_prevHitNormal_store(
                texelPos,
                uvec4(nzpacking_packNormalOct32(centerSampleData.hitNormal))
            );
            vec4 packedResampleMaterial = transient_restir_resampleMaterial_fetch(texelPos);
            history_restir_prevResampleMaterial_store(texelPos, packedResampleMaterial);

            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
            vec3 primaryViewPos = packedPrimary != 0u
                ? restir_splatUnpackPrimary(texelPos, packedPrimary, global_camProjInverse)
                : viewPos;
            vec3 V = normalize(-primaryViewPos);
            ResampleMaterial centerMaterial = resampleMaterial_unpack(packedResampleMaterial);

            ReSTIRReservoir spatialReservoir = readTemporalReservoir(texelPos);
            if (
                !restir_isReservoirValid(spatialReservoir)
                || !restir_isFinite(centerSampleData.sampleValue)
                || centerSampleData.sampleValue.w <= 0.0
            ) {
                transient_ssgiDiffOut_store(texelPos, vec4(0.0));
                transient_ssgiSpecOut_store(texelPos, vec4(0.0));
                return;
            }
            PairwiseMISMetadata metadata = pairwiseMISMetadata_init();
            metadata.accumM = spatialReservoir.m;
            float spatialTechniqueCount = 1.0;
            #if defined(SETTING_GI_SPATIAL_REUSE) && SETTING_GI_SPATIAL_REUSE_COUNT > 0
            uvec4 packedMetadata = transient_restir_pairwiseMISMetadata_fetch(texelPos);
            if (packedPrimary != 0u && packedMetadata.z != 0u) {
                metadata = pairwiseMISMetadata_unpack(packedMetadata);
                spatialTechniqueCount = float(SETTING_GI_SPATIAL_REUSE_COUNT + 1);
            }
            #endif

            ivec2 winTexel = texelPos + metadata.selectedTexelDelta;
            float mc = metadata.mc;
            float spatialWSum = metadata.spatialWSum;
            bool selectedNeighbor = winTexel != texelPos;

            vec4 originalSample = spatialReservoir.Y;
            float temporalM = spatialReservoir.m;
            spatialReservoir.m = metadata.accumM;

            vec4 selectedSampleF = centerSampleData.sampleValue;
            if (selectedNeighbor) {
                SpatialSampleData winSample = spatialSampleData_unpack(transient_restir_spatialInput_fetch(winTexel));
                winSample.normal = normalize(transient_viewNormal_fetch(winTexel).xyz * 2.0 - 1.0);
                float winViewZ = texelFetch(usam_gbufferSolidViewZ, winTexel, 0).x;
                vec2 winScreenPos = coords_texelToUV(winTexel, uval_mainImageSizeRcp) - uval_taaJitterUV;
                vec3 winViewPos = coords_toViewCoord(winScreenPos, winViewZ, global_camProjInverse);
                uint packedWinPrimary = restir_splatFetchCurrentPrimary(winTexel);
                vec3 winPrimaryViewPos = packedWinPrimary != 0u
                    ? restir_splatUnpackPrimary(winTexel, packedWinPrimary, global_camProjInverse)
                    : winViewPos;

                ReSTIRReservoir winRes = readTemporalReservoir(winTexel);

                ShiftMapping winToCenter = evaluateShiftMapping(winRes, centerMaterial, centerSampleData, winSample, primaryViewPos, winPrimaryViewPos);
                if (shiftMapping_isReusable(winToCenter)) {
                    spatialReservoir.Y = winToCenter.Y;
                    selectedSampleF = vec4(winSample.sampleValue.xyz, winToCenter.unmappedTargetPHat);
                } else {
                    metadata = pairwiseMISMetadata_init();
                    metadata.accumM = temporalM;
                    mc = 1.0;
                    spatialWSum = 0.0;
                    spatialTechniqueCount = 1.0;
                    selectedNeighbor = false;
                    spatialReservoir.Y = originalSample;
                    spatialReservoir.m = temporalM;
                }
            }

            float rcAvgWY = max(spatialReservoir.avgWY, 0.0);
            float canonicalWi = centerSampleData.sampleValue.w * rcAvgWY * mc;
            float canonicalRand = restir_updateRand(texelPos, 3336u);

            bool chooseCanon = restir_updateReservoir(
                spatialReservoir,
                spatialWSum,
                originalSample,
                canonicalWi,
                0.0,
                canonicalRand
            );

            if (chooseCanon || !selectedNeighbor) {
                selectedSampleF = centerSampleData.sampleValue;
            }

            vec4 ssgiDiffOut = vec4(0.0, 0.0, 0.0, -1.0);
            vec4 ssgiSpecOut = vec4(0.0, 0.0, 0.0, -1.0);
            vec4 resultY = spatialReservoir.Y;

            float avgWY = spatialWSum
                / (selectedSampleF.w * spatialTechniqueCount);
            if (!restir_isFinite(avgWY) || avgWY <= 0.0) {
                transient_ssgiDiffOut_store(texelPos, vec4(0.0));
                transient_ssgiSpecOut_store(texelPos, vec4(0.0));
                return;
            }

            vec3 winL_out = resultY.xyz;
            float winHitDist = resultY.w;

            vec3 resolvedNormal = resampleMaterial_resolveNormal(
                centerSampleData.geomNormal,
                centerSampleData.normal,
                V
            );
            float rawNDotL = dot(resolvedNormal, winL_out);
            if (
                rawNDotL <= 0.0
                || dot(centerSampleData.geomNormal, winL_out) <= 0.0
                || dot(centerSampleData.geomNormal, V) <= 0.0
            ) {
                transient_ssgiDiffOut_store(texelPos, vec4(0.0));
                transient_ssgiSpecOut_store(texelPos, vec4(0.0));
                return;
            }
            ResampleBRDF outBRDF = resampleMaterial_evalBRDF(
                centerMaterial,
                resolvedNormal,
                winL_out,
                V
            );

            ssgiDiffOut = vec4((selectedSampleF.xyz * outBRDF.diffuse) * avgWY, winHitDist);
            ssgiSpecOut = vec4((selectedSampleF.xyz * outBRDF.specular) * avgWY, winHitDist);
            float denoiseNDotV = saturate(dot(resolvedNormal, normalize(-viewPos)));
            vec3 specDenoiseFactor = resampleMaterial_specularDenoiseFactor(centerMaterial, denoiseNDotV);
            ssgiSpecOut.rgb *= rcp(specDenoiseFactor);
            ssgiDiffOut.rgb = restir_isFinite(ssgiDiffOut.rgb)
                ? clamp(ssgiDiffOut.rgb, 0.0, FP16_MAX)
                : vec3(0.0);
            ssgiSpecOut.rgb = restir_isFinite(ssgiSpecOut.rgb)
                ? clamp(ssgiSpecOut.rgb, 0.0, FP16_MAX)
                : vec3(0.0);

            #if SETTING_DEBUG_OUTPUT
            vec4 vvv = vec4(0.0);
            #endif
            if (!chooseCanon && selectedNeighbor) {
                #if SETTING_DEBUG_OUTPUT
                vvv = vec4(0.0, 1.0, 0.0, 0.0);
                #endif

                bool discardSptialReuse = false;
                if (winHitDist > 0.0) {
                    float normalOffset = min(0.05, winHitDist * 0.25);
                    vec3 rayOriginView = primaryViewPos + centerSampleData.geomNormal * normalOffset;
                    vec3 expectedHitView = primaryViewPos + winL_out * winHitDist;
                    vec3 rayOffsetView = expectedHitView - rayOriginView;
                    float expectedHitDistance = length(rayOffsetView);
                    vec3 worldPos = coords_pos_viewToWorld(rayOriginView, gbufferModelViewInverse) + vec3(cameraPositionInt) + cameraPositionFract;
                    vec3 worldDir = coords_dir_viewToWorld(rayOffsetView * rcp(expectedHitDistance));
                    VoxelRay voxelRay = voxelray_setup(worldPos, worldDir, 0u);
                    VoxelHit hit = voxel_traceRay(voxelRay, 128);
                    vec3 expectedHitPos = worldPos + worldDir * expectedHitDistance;
                    discardSptialReuse = !hit.hit || distanceSq(hit.hitPos, expectedHitPos) > 0.05;
                }

                if (discardSptialReuse) {
                    spatialReservoir = restir_initReservoir();
                    ssgiDiffOut = vec4(0.0);
                    ssgiSpecOut = vec4(0.0);
                    #if SETTING_DEBUG_OUTPUT
                    vvv = vec4(1.0, 0.0, 0.0, 0.0);
                    #endif
                }
            }
            #if SETTING_DEBUG_OUTPUT
            imageStore(uimg_temp5, texelPos, vvv);
            #endif
            transient_ssgiDiffOut_store(texelPos, ssgiDiffOut);
            transient_ssgiSpecOut_store(texelPos, ssgiSpecOut);
        }
    }
    //    ssbo_rayDataIndices[dataIndex] = rayIndex;
    //    uvec4 subgroupRayCountBalllot = subgroupBallot(rayIndex < 0xFFFFFFFFu);
    //    if (subgroupElect()) {
    //        shared_rayCount[gl_SubgroupID] = subgroupBallotBitCount(subgroupRayCountBalllot);
    //    }
    //    barrier();
    //    if (gl_SubgroupID == 0u) {
    //        uint partialRayCount = gl_SubgroupInvocationID < gl_NumSubgroups ? shared_rayCount[gl_SubgroupInvocationID] : 0u;
    //        uint totalRayCount = subgroupAdd(partialRayCount);
    //        if (subgroupElect()) {
    //            transient_spatialReuseRayCount_store(ivec2(swizzledWGPos), vec4(float(totalRayCount)));
    //        }
    //    }
}
