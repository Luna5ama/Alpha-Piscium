#extension GL_KHR_shader_subgroup_ballot : enable

#define GLOBAL_DATA_MODIFIER restrict buffer

layout(local_size_x = 16, local_size_y = 16) in;

#include "/util/Material.glsl"
#include "/util/ThreadGroupTiling.glsl"
#include "/techniques/gi/Common.glsl"
#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/ReservoirSplat.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/techniques/gi/PairwiseMISMetadata.glsl"
#include "/techniques/gi/PathGuiding.glsl"
#include "/util/BitPacking.glsl"

const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform image2D uimg_rgba16f;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(r32f) uniform image2D uimg_r32f;
layout(r32ui) uniform restrict writeonly uimage2D uimg_r32ui;
layout(rgba32ui) uniform restrict writeonly uimage2D uimg_rgba32ui;
layout(rgba8) uniform restrict writeonly image2D uimg_temp5;
#ifdef PATH_GUIDING_ENABLED
layout(rg32ui) uniform restrict writeonly uimage2D uimg_rg32ui;
#endif

ReSTIRReservoir readTemporalReservoir(ivec2 texelPos) {
    uvec4 reprojectedData = transient_restir_reservoirTemporal_fetch(texelPos);
    return restir_reservoir_unpack(reprojectedData);
}

// Returns the world-direction octant of a queued visibility ray, or NO_TRACE.
const uint NO_TRACE = 8u;

// trainRecord: path guide training record of the final sample (left zero when there is none).
uint spatialShade(ivec2 texelPos, uvec2 swizzledWGPos, inout uvec2 trainRecord) {
    uvec4 packedTemporalReservoir = transient_restir_reservoirTemporal_fetch(texelPos);
    history_restir_reservoirTemporal_store(texelPos, packedTemporalReservoir);
    uint packedPrimary = restir_splatFetchCurrentPrimary(texelPos);
    history_restir_primary_store(texelPos, uvec4(packedPrimary));

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

        ReSTIRReservoir spatialReservoir = restir_reservoir_unpack(packedTemporalReservoir);

        vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
        vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
        vec3 primaryViewPos = packedPrimary != 0u
            ? restir_splatUnpackPrimary(texelPos, packedPrimary, global_camProjInverse)
            : viewPos;
        vec3 V = normalize(-primaryViewPos);
        ResampleMaterial centerMaterial = resampleMaterial_unpack(packedResampleMaterial);

        PairwiseMISMetadata metadata = pairwiseMISMetadata_init();
        metadata.accumM = spatialReservoir.m;
        float spatialTechniqueCount = 1.0;
        // Specular output is mix(ratioSpec, ReSTIR specular, restirSpecWeight); the ratio part has no visibility term.
        vec3 ratioSpec = vec3(0.0);
        float restirSpecWeight = 1.0;
        #if defined(SETTING_GI_SPATIAL_REUSE) && SETTING_GI_SPATIAL_REUSE_COUNT > 0
        uvec4 packedMetadata = transient_restir_pairwiseMISMetadata_fetch(texelPos);
        if (packedMetadata.z != 0u) {
            metadata = pairwiseMISMetadata_unpack(packedMetadata);
            spatialTechniqueCount = float(SETTING_GI_SPATIAL_REUSE_COUNT + 1);
            // GIReSTIRPairedSpatialReuse left the BRDF-ratio resolve here: xyz = weighted mean, w = weight sum.
            vec4 specularRatio = transient_ssgiSpecOut_fetch(texelPos);
            if (specularRatio.w > 0.0) {
                vec3 resolvedNormal = resampleMaterial_resolveNormal(centerSampleData.geomNormal, centerSampleData.normal, V);
                float denoiseNDotV = saturate(dot(resolvedNormal, normalize(-viewPos)));
                ratioSpec = min(specularRatio.xyz * rcp(resampleMaterial_specularDenoiseFactor(centerMaterial, denoiseNDotV)), FP16_MAX);
                restirSpecWeight = centerMaterial.roughness;
            }
        }
        #endif
        vec4 ratioOnlySpecOut = vec4(ratioSpec * (1.0 - restirSpecWeight), 0.0);

        if (!restir_isFinite(centerSampleData.sampleValue)) {
            transient_ssgiDiffOut_store(texelPos, vec4(0.0));
            transient_ssgiSpecOut_store(texelPos, ratioOnlySpecOut);
            return NO_TRACE;
        }

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
            canonicalRand
        );

        if (chooseCanon || !selectedNeighbor) {
            selectedSampleF = centerSampleData.sampleValue;
        }

        vec4 resultY = spatialReservoir.Y;

        float avgWY = spatialWSum
            / (selectedSampleF.w * spatialTechniqueCount);
        vec4 ssgiDiffOut;
        vec4 ssgiSpecOut;
        float diffuseShare = -1.0;
        if (restir_isFinite(avgWY) && avgWY > 0.0) {
            diffuseShare = restir_shadeSample(
                selectedSampleF.xyz,
                resultY,
                avgWY,
                centerSampleData.geomNormal,
                centerSampleData.normal,
                V,
                normalize(-viewPos),
                centerMaterial,
                texelPos,
                ssgiDiffOut,
                ssgiSpecOut
            );
        }
        if (diffuseShare < 0.0) {
            transient_ssgiDiffOut_store(texelPos, vec4(0.0));
            transient_ssgiSpecOut_store(texelPos, ratioOnlySpecOut);
            return NO_TRACE;
        }
        ssgiSpecOut.rgb = mix(ratioSpec, ssgiSpecOut.rgb, restirSpecWeight);

        #ifdef PATH_GUIDING_ENABLED
        // The diffuse guide learns the final sample with probability f_d / (f_d + f_s).
        uint recordFlags = restir_updateRand(texelPos, 0x68e31da4u) < diffuseShare ? PG_RECORD_VALID : PG_RECORD_SPECULAR;
        recordFlags |= !chooseCanon && selectedNeighbor ? PG_RECORD_NEIGHBOR : 0u;
        recordFlags |= resultY.w <= 0.0 ? PG_RECORD_SKY : 0u;
        trainRecord = uvec2(nzpacking_packNormalOct32(coords_dir_viewToWorld(resultY.xyz)), recordFlags);
        #endif

        #if SETTING_DEBUG_OUTPUT
        imageStore(uimg_temp5, texelPos, !chooseCanon && selectedNeighbor ? vec4(0.0, 1.0, 0.0, 0.0) : vec4(0.0));
        #endif
        transient_ssgiDiffOut_store(texelPos, ssgiDiffOut);
        transient_ssgiSpecOut_store(texelPos, ssgiSpecOut);

        #if defined(SETTING_GI_SPATIAL_REUSE) && SETTING_GI_SPATIAL_REUSE_COUNT > 0
        // Neighbor selections are provisional until GIReSTIRSpatialReuseTrace confirms visibility.
        // The trace pass keeps only the same ratio part for occluded selections.
        if (!chooseCanon && selectedNeighbor && resultY.w > 0.0) {
            transient_restir_pairwiseMISMetadata_store(texelPos, uvec4(
                nzpacking_packNormalOct32(resultY.xyz),
                floatBitsToUint(resultY.w),
                packHalf4x16(vec4(ratioSpec, restirSpecWeight))
            ));
            uvec3 dirSign = uvec3(lessThan(mat3(gbufferModelViewInverse) * resultY.xyz, vec3(0.0)));
            return dirSign.x | (dirSign.y << 1u) | (dirSign.z << 2u);
        }
        #endif
    }
    return NO_TRACE;
}

shared uint shared_traceCounts[8 * 16];
shared uint shared_traceBase;

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    ivec2 texelPos = ivec2(workGroupOrigin + mortonPos);

    uint traceBin = NO_TRACE;
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        uvec2 trainRecord = uvec2(0u);
        traceBin = spatialShade(texelPos, swizzledWGPos, trainRecord);
        #ifdef PATH_GUIDING_ENABLED
        transient_pathGuide_trainRecord_store(texelPos, uvec4(trainRecord, 0u, 0u));
        #endif
    }

    #if defined(SETTING_GI_SPATIAL_REUSE) && SETTING_GI_SPATIAL_REUSE_COUNT > 0
    // Append the tile's visibility rays as one contiguous run, grouped by direction octant then Morton order.
    uint binRank = 0u;
    for (uint bin = 0u; bin < 8u; bin++) {
        uvec4 binBallot = subgroupBallot(traceBin == bin);
        if (subgroupElect()) {
            shared_traceCounts[bin * 16u + gl_SubgroupID] = subgroupBallotBitCount(binBallot);
        }
        if (traceBin == bin) {
            binRank = subgroupBallotExclusiveBitCount(binBallot);
        }
    }
    barrier();
    if (threadIdx == 0u) {
        uint total = 0u;
        for (uint bin = 0u; bin < 8u; bin++) {
            for (uint i = 0u; i < gl_NumSubgroups; i++) {
                uint count = shared_traceCounts[bin * 16u + i];
                shared_traceCounts[bin * 16u + i] = total;
                total += count;
            }
        }
        shared_traceBase = total > 0u ? atomicAdd(global_restirVisibilityRayCount, total) : 0u;
    }
    barrier();
    if (traceBin != NO_TRACE) {
        uint queueIndex = shared_traceBase + shared_traceCounts[traceBin * 16u + gl_SubgroupID] + binRank;
        indirectComputeData[queueIndex] = packUInt2x16(uvec2(texelPos));
    }
    #endif
}
