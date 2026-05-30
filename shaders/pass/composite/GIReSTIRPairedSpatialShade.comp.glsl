#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "/util/Material.glsl"
#include "/util/ThreadGroupTiling.glsl"
#include "/techniques/SST2.glsl"
#include "/techniques/gi/Common.glsl"
#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/techniques/gi/PairwiseMIS.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(std430, binding = 5) buffer RayData {
    uvec4 ssbo_rayData[];
};

layout(std430, binding = 6) buffer RayIndexData {
    uint ssbo_rayDataIndices[];
};

layout(rgba16f) uniform image2D uimg_rgba16f;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(r32f) uniform image2D uimg_r32f;
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
        SpatialSampleData centerSampleData = spatialSampleData_unpack(transient_restir_spatialInput_fetch(texelPos));
        history_restir_prevSample_store(texelPos, centerSampleData.sampleValue);
        history_restir_prevHitNormal_store(texelPos, vec4(centerSampleData.hitNormal * 0.5 + 0.5, 0.0));
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos, 4, texelPos);

        if (viewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
            vec3 V = normalize(-viewPos);
            ResampleMaterial centerMaterial = resampleMaterial_unpack(transient_restir_resampleMaterial_fetch(texelPos));

            PairwiseMISMetadata metadata = pairwiseMISMetadata_unpack(transient_restir_pairwiseMISMetadata_fetch(texelPos));

            ivec2 winTexel = metadata.selectedTexel;
            uint numValidNeighbors = metadata.numValidNeighbors;
            float mc = metadata.mc;
            float spatialWSum = metadata.spatialWSum;

            ReSTIRReservoir spatialReservoir = readTemporalReservoir(texelPos);
            vec4 originalSample = spatialReservoir.Y;
            spatialReservoir.m = metadata.accumM;

            vec4 selectedSampleF = centerSampleData.sampleValue;
            if (winTexel != texelPos) {
                SpatialSampleData winSample = spatialSampleData_unpack(transient_restir_spatialInput_fetch(winTexel));
                float winViewZ = texelFetch(usam_gbufferSolidViewZ, winTexel, 0).x;
                vec2 winScreenPos = coords_texelToUV(winTexel, uval_mainImageSizeRcp);
                vec3 winViewPos = coords_toViewCoord(winScreenPos, winViewZ, global_camProjInverse);

                ReSTIRReservoir winRes = readTemporalReservoir(winTexel);

                ShiftMapping winToCenter = evaluateShiftMapping(winRes, centerMaterial, centerSampleData, winSample, viewPos, winViewPos);
                spatialReservoir.Y = winToCenter.Y;
                selectedSampleF = vec4(winSample.sampleValue.xyz, abs(winToCenter.targetPHat));
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

            if (chooseCanon || winTexel == texelPos) {
                selectedSampleF = centerSampleData.sampleValue;
            }

            vec4 ssgiDiffOut = vec4(0.0, 0.0, 0.0, -1.0);
            vec4 ssgiSpecOut = vec4(0.0, 0.0, 0.0, -1.0);
            ReSTIRReservoir resultReservoir = spatialReservoir;

            float avgWY = spatialWSum * safeRcp(selectedSampleF.w) * safeRcp(float(numValidNeighbors + 1u));
            // resultReservoir.avgWY = avgWY;

            vec3 winL_out = resultReservoir.Y.xyz;
            float winHitDist = resultReservoir.Y.w;
            vec3 H_out = normalize(winL_out + V);

            float outNDotL = saturate(dot(centerSampleData.normal, winL_out));
            float outNDotH = saturate(dot(centerSampleData.normal, H_out));
            float outLDotH = saturate(dot(winL_out, H_out));

            float NDotV = saturate(dot(centerSampleData.normal, V));
            ResampleBRDF outBRDF = resampleMaterial_evalBRDF(centerMaterial, outNDotL, NDotV, outNDotH, outLDotH);
            float diffRatio = outBRDF.diffuse * safeRcp(outBRDF.full);

            vec3 totalOutput = selectedSampleF.xyz * outBRDF.full * avgWY;
            ssgiDiffOut = vec4(totalOutput * diffRatio, winHitDist);
            ssgiSpecOut = vec4(totalOutput * (1.0 - diffRatio), winHitDist);
            vec3 specAlbedo = resampleMaterial_specularAlbedo(centerMaterial, NDotV);
            ssgiSpecOut.rgb *= safeRcp(specAlbedo);

            #if SETTING_DEBUG_OUTPUT
            vec4 vvv = vec4(0.0);
            #endif
            if (!chooseCanon && winTexel != texelPos) {
                #if SETTING_DEBUG_OUTPUT
                vvv = vec4(0.0, 1.0, 0.0, 0.0);
                #endif

                SSTRay sstRay;
                if (resultReservoir.Y.w > 0.0) {
                    vec3 expectHitViewPos = viewPos + resultReservoir.Y.xyz * resultReservoir.Y.w;
                    vec3 rayOrigin = coords_viewToScreen(viewPos, global_camProj);
                    vec3 rayEnd = coords_viewToScreen(expectHitViewPos, global_camProj);
                    vec4 rayDirLen = normalizeAndLength(rayEnd - rayOrigin);
                    sstRay = sstray_setup(texelPos, rayOrigin, rayDirLen.xyz, rayDirLen.w);
                } else {
                    sstRay = sstray_setup(texelPos, viewPos, resultReservoir.Y.xyz);
                }
                sst_trace(sstRay, 4);
                if (sstRay.currT > 0.0) {
                    uvec4 packedData = sstray_pack(sstRay);
                    ssbo_rayData[dataIndex] = packedData;
                    rayIndex = sst2_encodeRayIndexBits(binLocalIndex, sstRay);
                } else {
                    bool discardSptialReuse = true;
                    if (sstRay.currT < -0.99) discardSptialReuse = false;

                    if (discardSptialReuse) {
                        resultReservoir = restir_initReservoir();
                        ssgiDiffOut = vec4(0.0);
                        ssgiSpecOut = vec4(0.0);
                        #if SETTING_DEBUG_OUTPUT
                        vvv = vec4(1.0, 0.0, 0.0, 0.0);
                        #endif
                    }
                }
            }
            #if SETTING_DEBUG_OUTPUT
            imageStore(uimg_temp5, texelPos, vvv);
            #endif

            ssgiDiffOut.rgb = clamp(ssgiDiffOut.rgb, 0.0, FP16_MAX);
            ssgiSpecOut.rgb = clamp(ssgiSpecOut.rgb, 0.0, FP16_MAX);
            transient_ssgiDiffOut_store(texelPos, ssgiDiffOut);
            transient_ssgiSpecOut_store(texelPos, ssgiSpecOut);
        }
    }
    ssbo_rayDataIndices[dataIndex] = rayIndex;
    uvec4 subgroupRayCountBalllot = subgroupBallot(rayIndex < 0xFFFFFFFFu);
    if (subgroupElect()) {
        shared_rayCount[gl_SubgroupID] = subgroupBallotBitCount(subgroupRayCountBalllot);
    }
    barrier();
    if (gl_SubgroupID == 0u) {
        uint partialRayCount = gl_SubgroupInvocationID < gl_NumSubgroups ? shared_rayCount[gl_SubgroupInvocationID] : 0u;
        uint totalRayCount = subgroupAdd(partialRayCount);
        if (subgroupElect()) {
            transient_spatialReuseRayCount_store(ivec2(swizzledWGPos), vec4(float(totalRayCount)));
        }
    }
}
