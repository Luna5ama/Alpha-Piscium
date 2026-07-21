#ifndef INCLUDE_techniques_gi_RadianceCacheUpdate_glsl
#define INCLUDE_techniques_gi_RadianceCacheUpdate_glsl a

#define RC_DATA_MODIFIER restrict buffer
#define GLOBAL_DATA_MODIFIER restrict buffer

#include "/techniques/gi/RadianceCache.glsl"
#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/gi/HitDirectLighting.glsl"
#include "/techniques/gi/ResampleMaterial.glsl"
#include "/techniques/voxel/VoxelTrace.glsl"
#include "/techniques/voxel/SurfaceData.glsl"
#include "/util/MaterialIDConst.glsl"
#include "/util/Rand.glsl"

layout(std430, binding = 2) RC_DATA_MODIFIER RadianceCacheUpdateEntryIndexData {
    uint rc_updateEntryIndices[];
};

// x: world key hash, or RC_INVALID
// y bits 0-5: screen touched faces
// y bits 6-11: next-frame hit feedback faces
layout(std430, binding = 5) RC_DATA_MODIFIER RadianceCacheFeedbackData {
    uvec2 rc_feedback[];
};

struct RCCandidate {
    vec3 radiance;
    vec3 dir;
    vec3 hitPos;
    vec3 hitNormal;
    float targetWeight;
    uint flags;
    bool valid;
};

const float RC_CV_ALPHA = 1.0;
const float RC_CV_M_CAP = 128.0;
const float RC_SPATIAL_M_CAP = 8.0;

struct RCCVAccumulator {
    vec3 estimateSum;
    float weightSum;
    bool invalid;
};

RCCVAccumulator rc_cvAccumulatorInit() {
    RCCVAccumulator accumulator;
    accumulator.estimateSum = vec3(0.0);
    accumulator.weightSum = 0.0;
    accumulator.invalid = false;
    return accumulator;
}

vec3 rc_cvInitialEstimate(RCCandidate candidate) {
    return candidate.valid ? candidate.radiance : vec3(0.0);
}

void rc_cvAccumulatorAdd(inout RCCVAccumulator accumulator, vec3 estimate, float weight) {
    if (weight <= 0.0) {
        return;
    }
    if (isnan(weight) || any(isnan(estimate))) {
        accumulator.invalid = true;
        return;
    }

    accumulator.estimateSum += estimate * weight;
    accumulator.weightSum += weight;
}

bool rc_cvAccumulatorValid(RCCVAccumulator accumulator) {
    return !accumulator.invalid
        && accumulator.weightSum > 0.0
        && !isnan(accumulator.weightSum)
        && !any(isnan(accumulator.estimateSum));
}

vec3 rc_cvAccumulatorResolve(RCCVAccumulator accumulator) {
    return accumulator.estimateSum * safeRcp(accumulator.weightSum);
}

uint rc_feedbackRecordIndex(uint side, uint entryIndex) {
    return rc_bufferEntryIndex(side, entryIndex);
}

uint rc_feedbackScreenBits(uint faceMask) {
    return (faceMask & RC_FEEDBACK_FACE_MASK) << RC_FEEDBACK_SCREEN_SHIFT;
}

uint rc_feedbackHitBits(uint faceMask) {
    return (faceMask & RC_FEEDBACK_FACE_MASK) << RC_FEEDBACK_HIT_SHIFT;
}

void rc_feedbackClearRecord(uint side, uint entryIndex) {
    uint recordIndex = rc_feedbackRecordIndex(side, entryIndex);
    rc_feedback[recordIndex] = uvec2(RC_INVALID, 0u);
}

void rc_markScreenTouchedFace(uint level, ivec3 worldCellCoord, uint faceId) {
    if (!rc_worldCellInCurrentClip(level, worldCellCoord)) {
        return;
    }

    uint entryIndex = rc_entryIndex(level, worldCellCoord);
    uint recordIndex = rc_feedbackRecordIndex(rc_currentSide(), entryIndex);
    uint worldKeyHash = rc_worldKeyHash(level, worldCellCoord);
    uint oldKey = atomicCompSwap(rc_feedback[recordIndex].x, RC_INVALID, worldKeyHash);
    if (oldKey == RC_INVALID || oldKey == worldKeyHash) {
        atomicOr(rc_feedback[recordIndex].y, rc_feedbackScreenBits(rc_faceBit(faceId)));
    }
}

void rc_markHitFeedbackFace(uint level, ivec3 worldCellCoord, uint faceId) {
    if (!rc_worldCellInCurrentClip(level, worldCellCoord)) {
        return;
    }

    uint entryIndex = rc_entryIndex(level, worldCellCoord);
    uint recordIndex = rc_feedbackRecordIndex(rc_currentSide(), entryIndex);
    uint worldKeyHash = rc_worldKeyHash(level, worldCellCoord);
    uint oldKey = atomicCompSwap(rc_feedback[recordIndex].x, RC_INVALID, worldKeyHash);
    if (oldKey == RC_INVALID || oldKey == worldKeyHash) {
        atomicOr(rc_feedback[recordIndex].y, rc_feedbackHitBits(rc_faceBit(faceId)));
    }
}

bool rc_reservoirUpdateWeighted(
    inout RCReservoir reservoir,
    inout float wSum,
    RCCandidate candidate,
    float updateWeight,
    float mInc,
    float randValue
) {
    if (
        !candidate.valid
        || candidate.targetWeight <= 0.0
        || updateWeight <= 0.0
        || mInc <= 0.0
    ) {
        return false;
    }

    wSum += updateWeight;
    reservoir.m += mInc;

    float p = updateWeight * safeRcp(wSum);
    if (randValue < p) {
        reservoir.radiance = candidate.radiance;
        reservoir.sampleDir = candidate.dir;
        reservoir.hitPos = candidate.hitPos;
        return true;
    }

    return false;
}

vec3 rc_hemisphereDirection(vec3 normal, vec3 localDir) {
    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, normal));
    vec3 B = cross(normal, T);
    return normalize(T * localDir.x + B * localDir.y + normal * localDir.z);
}

void rc_touchHitFeedback(VoxelHit hit) {
    if (!hit.hit || hit.materialID == 0u || hit.materialID == MATERIAL_ID_WATER) {
        return;
    }

    uint faceId = rc_faceIdFromNormal(hit.normal);
    vec3 faceNormal = rc_faceNormal(faceId);
    vec3 surfacePos = hit.hitPos - faceNormal * 0.02;
    for (uint level = 0u; level < RC_CLIP_LEVELS; level++) {
        ivec3 worldCellCoord = rc_worldCellCoord(surfacePos, level);
        rc_markHitFeedbackFace(level, worldCellCoord, faceId);
    }
}

bool rc_loadPreviousHitReservoir(
    VoxelHit hit,
    out RCReservoir reservoir,
    out vec3 faceNormal
) {
    uint faceId = rc_faceIdFromNormal(hit.normal);
    faceNormal = rc_faceNormal(faceId);
    vec3 surfacePos = hit.hitPos - faceNormal * 0.02;
    uint level = rc_selectLevel(surfacePos);
    ivec3 worldCellCoord = rc_worldCellCoord(surfacePos, level);
    return rc_loadFaceReservoir(rc_previousSide(), level, worldCellCoord, faceId, reservoir);
}

vec3 rc_sampleMissRadiance(vec3 rayDir) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, rayDir);
    return atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
}

vec3 rc_sampleHitRadiance(VoxelHit hit, vec3 outgoingDir, out bool valid) {
    valid = false;
    if (!hit.hit) {
        vec3 missRadiance = rc_sampleMissRadiance(normalize(-outgoingDir));
        valid = rc_luminance(missRadiance) > 0.0 && !any(isnan(missRadiance));
        return valid ? missRadiance : vec3(0.0);
    }

    voxel_SurfaceData surface = voxel_sampleVoxelSurface(hit, 0.0);
    if (!surface.valid) {
        return vec3(0.0);
    }
    vec3 radiance = surface.material.emissive
        + gi_hitDirectLighting(surface.material, hit.hitPos, outgoingDir, hit.normal, hit.normal);
    valid = rc_luminance(radiance) > 0.0 && !any(isnan(radiance));
    surface.material.roughness = max(surface.material.roughness, RC_MAX_ROUGHNESS * 0.5);

    RCReservoir prevReservoir;
    vec3 faceNormal;
    if (!rc_loadPreviousHitReservoir(hit, prevReservoir, faceNormal)) {
        return radiance;
    }

    vec3 incomingDir = normalize(prevReservoir.sampleDir);
    vec3 viewDir = normalize(outgoingDir);
    float NDotL = dot(faceNormal, incomingDir);
    float NDotV = dot(faceNormal, viewDir);
    if (NDotL <= 0.0 || NDotV <= 0.0) {
        return radiance;
    }

    vec3 incomingRadiance = rc_reservoirEstimateRadiance(prevReservoir);
    if (rc_luminance(incomingRadiance) <= 0.0 || any(isnan(incomingRadiance)) || any(isnan(incomingDir))) {
        return radiance;
    }

    vec3 H = incomingDir + viewDir;
    float invHLen = inversesqrt(max(dot(H, H), 1e-6));
    float NDotH = saturate(dot(faceNormal, H * invHLen));
    float LDotH = saturate(dot(incomingDir, H * invHLen));
    ResampleMaterial resampleMaterial = resampleMaterial_fromMaterial(surface.material);
    ResampleBRDF brdf = resampleMaterial_evalBRDF(resampleMaterial, NDotL, NDotV, NDotH, LDotH);
    if (brdf.full <= 0.0) {
        return radiance;
    }

    vec3 totalBRDF = surface.material.albedo * brdf.diffuse + vec3(brdf.specular);
    vec3 bounceRadiance = incomingRadiance * totalBRDF;
    if (rc_luminance(bounceRadiance) <= 0.0 || any(isnan(bounceRadiance))) {
        return radiance;
    }

    radiance += bounceRadiance;
    valid = true;
    return radiance;
}

bool rc_revalidateHistoryReservoir(
    ivec3 worldCellCoord,
    uint level,
    uint faceId,
    inout RCReservoir reservoir
) {
    if (!rc_reservoirValid(reservoir)) {
        return false;
    }

    vec3 sampleDir = normalize(reservoir.sampleDir);
    if (any(isnan(sampleDir))) {
        return false;
    }

    vec3 rayOrigin = rc_faceRayOrigin(worldCellCoord, level, faceId);
    VoxelRay voxelRay = voxelray_setup(rayOrigin, sampleDir, 0u);
    VoxelHit hit = voxel_traceRay(voxelRay, 128);

    uint flags = rc_reservoirMetaFlags(reservoir.meta);
    bool expectSurfaceHit = (flags & RC_RES_FLAG_SURFACE_HIT) != 0u;
    bool expectSkyMiss = (flags & RC_RES_FLAG_SKY_MISS) != 0u;

    if (expectSurfaceHit) {
        if (!hit.hit) {
            return false;
        }

        float hitThreshold = pow2(max(ldexp(0.25, int(level)), 0.1));
        if (distanceSq(hit.hitPos, reservoir.hitPos) > hitThreshold) {
            return false;
        }
    } else if (expectSkyMiss) {
        if (hit.hit) {
            return false;
        }
    } else {
        return false;
    }

    bool radianceValid = false;
    vec3 radiance = rc_sampleHitRadiance(hit, -sampleDir, radianceValid);
    float newTargetWeight = rc_luminance(radiance);
    if (!radianceValid
        || newTargetWeight <= 0.0
        || any(isnan(radiance))
        || isnan(newTargetWeight)
    ) {
        return false;
    }

    float oldTargetWeight = rc_luminance(reservoir.radiance);
    float num = pow2(min(newTargetWeight, oldTargetWeight));
    float denom = pow2(max(newTargetWeight, oldTargetWeight));
    float ratio = saturate(num * safeRcp(denom));
    reservoir.m *= ratio;

    reservoir.radiance = radiance;
    if (hit.hit) {
        reservoir.hitPos = hit.hitPos;
        flags = RC_RES_FLAG_SURFACE_HIT;
    } else {
        flags = RC_RES_FLAG_SKY_MISS;
    }
    reservoir.meta = rc_packReservoirMeta(rc_reservoirMetaAge(reservoir.meta), true, flags);
    return true;
}

bool rc_loadRandomSpatialNeighbor(
    uint entryIndex,
    ivec3 worldCellCoord,
    uint level,
    uint faceId,
    out ivec3 neighborCell,
    out vec3 neighborOrigin,
    out RCReservoir neighborReservoir
) {
    neighborCell = worldCellCoord;
    neighborOrigin = vec3(0.0);
    neighborReservoir = rc_reservoirInit();

    uint neighborIndex = hash_41_q5(uvec4(entryIndex, faceId, frameCounter, 0xC2B2AE35u)) & 7u;
    ivec2 neighborOffset = rc_neighborOffset8(neighborIndex);
    neighborCell = worldCellCoord + rc_neighborPlaneOffset(faceId, neighborOffset.x, neighborOffset.y);

    if (!rc_loadFaceReservoir(rc_previousSide(), level, neighborCell, faceId, neighborReservoir)) {
        return false;
    }
    if (!rc_reservoirIsSurfaceHit(neighborReservoir)) {
        return false;
    }

    neighborOrigin = rc_faceRayOrigin(neighborCell, level, faceId);
    return true;
}

float rc_pairwiseSpatialMIS_MAware(
    vec3 targetOrigin,
    vec3 targetNormal,
    vec3 sourceOrigin,
    vec3 sourceNormal,
    vec3 hitPos,
    vec3 hitNormal,
    float targetM,
    float targetShiftedWeight,
    float sourceM,
    float sourceTargetWeight,
    out float shiftWeight
) {
    float pTargetArea = rc_areaPdfCosineConnection(targetOrigin, targetNormal, hitPos, hitNormal);
    float pSourceArea = rc_areaPdfCosineConnection(sourceOrigin, sourceNormal, hitPos, hitNormal);
    shiftWeight = 0.0;

    if (pTargetArea <= 0.0 || pSourceArea <= 0.0) {
        return 0.0;
    }

    shiftWeight = pTargetArea * safeRcp(pSourceArea);
    float sourceMass = sourceM * sourceTargetWeight;
    float targetMass = targetM * targetShiftedWeight * shiftWeight;
    float denom = sourceMass + targetMass;
    if (denom <= 0.0) {
        return 0.0;
    }

    return sourceMass * safeRcp(denom);
}

RCCandidate rc_generateCandidate(uint entryIndex, ivec3 worldCellCoord, uint level, uint faceId, bool allowHitFeedback) {
    RCCandidate candidate;
    candidate.radiance = vec3(0.0);
    candidate.dir = rc_faceNormal(faceId);
    candidate.hitPos = rc_faceCenter(worldCellCoord, level, faceId);
    candidate.hitNormal = vec3(0.0);
    candidate.targetWeight = 0.0;
    candidate.flags = 0u;
    candidate.valid = false;

    vec3 faceNormal = rc_faceNormal(faceId);
    uvec4 randHash = hash_44_q3(uvec4(entryIndex, faceId, frameCounter, 0x9E3779B9u));
    vec2 randValue = hash_uintToFloat(randHash.xy);
    vec4 localSample = rand_sampleInCosineWeightedHemisphere(randValue);
    vec3 worldDir = rc_hemisphereDirection(faceNormal, localSample.xyz);
    float cosTheta = max(dot(faceNormal, worldDir), 0.0);
    if (cosTheta <= 0.0 || localSample.w <= 0.0) {
        return candidate;
    }

    vec3 rayOrigin = rc_faceRayOrigin(worldCellCoord, level, faceId);
    VoxelRay voxelRay = voxelray_setup(rayOrigin, worldDir, 0u);
    VoxelHit hit = voxel_traceRay(voxelRay, 128);
    if (allowHitFeedback && hit.hit) {
        rc_touchHitFeedback(hit);
    }

    bool radianceValid = false;
    vec3 radiance = rc_sampleHitRadiance(hit, -worldDir, radianceValid);
    float targetWeight = rc_luminance(radiance);
    bool candidateValid = radianceValid
        && targetWeight > 0.0
        && !any(isnan(radiance))
        && !isnan(targetWeight);
    if (!candidateValid) {
        return candidate;
    }

    candidate.radiance = radiance;
    candidate.dir = worldDir;
    if (hit.hit) {
        candidate.hitPos = hit.hitPos;
        candidate.hitNormal = hit.normal;
        candidate.flags = RC_RES_FLAG_SURFACE_HIT;
    } else {
        candidate.flags = RC_RES_FLAG_SKY_MISS;
    }
    candidate.targetWeight = targetWeight;
    candidate.valid = true;
    return candidate;
}

bool rc_generateSpatialCandidate(
    ivec3 worldCellCoord,
    uint level,
    uint faceId,
    vec3 sourceOrigin,
    RCReservoir sourceReservoir,
    float targetM,
    float sourceM,
    out RCCandidate candidate,
    out float misWeight,
    out float shiftWeight
) {
    candidate.radiance = vec3(0.0);
    candidate.dir = rc_faceNormal(faceId);
    candidate.hitPos = rc_faceCenter(worldCellCoord, level, faceId);
    candidate.hitNormal = vec3(0.0);
    candidate.targetWeight = 0.0;
    candidate.flags = 0u;
    candidate.valid = false;
    misWeight = 0.0;
    shiftWeight = 0.0;

    #ifndef SETTING_RC_SPATIAL_ENABLE
        return false;
    #else
        vec3 targetNormal = rc_faceNormal(faceId);
        vec3 targetOrigin = rc_faceRayOrigin(worldCellCoord, level, faceId);

        vec3 hitPos = sourceReservoir.hitPos;
        if (any(isnan(hitPos)) || dot(hitPos, hitPos) <= 1e-6) {
            return false;
        }

        vec3 sourceDir = normalize(sourceReservoir.sampleDir);
        vec3 sourceToHit = hitPos - sourceOrigin;
        float sourceHitDist = dot(sourceToHit, sourceDir);
        if (sourceHitDist <= 0.05) {
            return false;
        }

        vec3 expectedHitDir = normalize(sourceToHit);
        if (dot(expectedHitDir, sourceDir) < 0.95) {
            return false;
        }

        vec3 toHit = hitPos - targetOrigin;
        float hitDistanceSq = dot(toHit, toHit);
        if (hitDistanceSq <= 1e-6) {
            return false;
        }

        vec3 shiftedDir = toHit * inversesqrt(hitDistanceSq);
        float targetCos = dot(targetNormal, shiftedDir);
        if (targetCos <= 0.05) {
            return false;
        }

        VoxelRay ray = voxelray_setup(targetOrigin, shiftedDir, 0u);
        VoxelHit hit = voxel_traceRay(ray, 128);
        if (!hit.hit) {
            return false;
        }

        float hitThreshold = pow2(max(ldexp(0.25, int(level)), 0.1));
        if (distanceSq(hit.hitPos, hitPos) > hitThreshold) {
            return false;
        }
        if (dot(hit.normal, -shiftedDir) <= 0.0) {
            return false;
        }

        bool radianceValid = false;
        vec3 radiance = rc_sampleHitRadiance(hit, -shiftedDir, radianceValid);
        float targetWeight = rc_luminance(radiance);
        float sourceTargetWeight = rc_luminance(sourceReservoir.radiance);
        if (
            !radianceValid
            || targetWeight <= 0.0
            || any(isnan(radiance))
            || isnan(targetWeight)
        ) {
            return false;
        }

        misWeight = rc_pairwiseSpatialMIS_MAware(
            targetOrigin,
            targetNormal,
            sourceOrigin,
            targetNormal,
            hitPos,
            hit.normal,
            targetM,
            targetWeight,
            sourceM,
            sourceTargetWeight,
            shiftWeight
        );
        if (misWeight <= 0.0 || shiftWeight <= 0.0) {
            return false;
        }

        candidate.radiance = radiance;
        candidate.dir = shiftedDir;
        candidate.hitPos = hit.hitPos;
        candidate.hitNormal = hit.normal;
        candidate.targetWeight = targetWeight;
        candidate.flags = RC_RES_FLAG_SURFACE_HIT;
        candidate.valid = true;
        return true;
    #endif
}

RCReservoir rc_reservoirInitFromCandidate(RCCandidate candidate) {
    RCReservoir reservoir;
    if (candidate.valid && candidate.targetWeight > 0.0) {
        reservoir.radiance = candidate.radiance;
        reservoir.avgWY = 1.0;
        reservoir.sampleDir = candidate.dir;
        reservoir.m = 1.0;
        reservoir.hitPos = candidate.hitPos;
        reservoir.meta = rc_packReservoirMeta(0u, true, candidate.flags);
        reservoir.estimate = candidate.radiance;
    } else {
        reservoir = rc_reservoirInit();
    }
    return reservoir;
}

void rc_updateFace(uint entryIndex, uvec4 entry, ivec3 worldCellCoord, uint level, uint faceId) {
    uint reservoirIndex = rc_faceReservoirIndex(entry.x, entry.y, faceId);
    if (reservoirIndex >= uint(SETTING_RC_POOL_SIZE)) {
        return;
    }

    uint worldKeyHash = rc_worldKeyHash(level, worldCellCoord);
    uint feedbackRecordIndex = rc_feedbackRecordIndex(rc_currentSide(), entryIndex);
    uvec2 feedbackRecord = rc_feedback[feedbackRecordIndex];
    uint screenTouchedFaceMask = 0u;
    if (feedbackRecord.x == worldKeyHash && feedbackRecord.x == entry.z) {
        screenTouchedFaceMask = (feedbackRecord.y >> RC_FEEDBACK_SCREEN_SHIFT) & RC_FEEDBACK_FACE_MASK;
    }
    bool allowHitFeedback = rc_hasFace(screenTouchedFaceMask, faceId);

    RCCandidate candidate = rc_generateCandidate(entryIndex, worldCellCoord, level, faceId, allowHitFeedback);
    RCReservoir reservoir = rc_reservoirInit();
    RCCVAccumulator cvAccumulator = rc_cvAccumulatorInit();
    float qInit = candidate.valid ? 1.0 : 0.0;
    rc_cvAccumulatorAdd(cvAccumulator, rc_cvInitialEstimate(candidate), qInit);

    uint prevBufferIndex = rc_bufferEntryIndex(rc_previousSide(), entryIndex);
    uvec4 prevEntry = rc_indirection[prevBufferIndex];
    bool historyValid = prevEntry.x != RC_INVALID
        && prevEntry.z == entry.z
        && rc_entryMetaValid(prevEntry.w)
        && rc_entryMetaLevel(prevEntry.w) == level
        && rc_hasFace(prevEntry.y, faceId);

    uint historyAge = 0u;
    if (historyValid) {
        uint prevReservoirIndex = rc_faceReservoirIndex(prevEntry.x, prevEntry.y, faceId);
        if (prevReservoirIndex < uint(SETTING_RC_POOL_SIZE)) {
            reservoir = rc_reservoirLoad(rc_previousSide(), prevReservoirIndex);
            historyValid = rc_reservoirValid(reservoir);
            if (historyValid) {
                historyAge = rc_reservoirMetaAge(reservoir.meta);
                historyValid = reservoir.avgWY > 0.0
                    && reservoir.m > 0.0
                    && rc_luminance(reservoir.radiance) > 0.0
                    && !isnan(reservoir.avgWY)
                    && !isnan(reservoir.m)
                    && !any(isnan(reservoir.radiance))
                    && !any(isinf(reservoir.radiance));
            }
        } else {
            historyValid = false;
        }
    }
    float wSum = 0.0;
    RCReservoir historyBeforeRevalidate = reservoir;
    if (historyValid) {
        historyBeforeRevalidate = reservoir;
        uint validateId = worldKeyHash + faceId;
        if ((validateId & 7u) == (uint(frameCounter) & 7u)) {
            historyValid = rc_revalidateHistoryReservoir(
                worldCellCoord,
                level,
                faceId,
                reservoir
            );
            if (!historyValid) {
                reservoir = rc_reservoirInit();
            }
        }
    }
    if (historyValid) {
        wSum = historyBeforeRevalidate.avgWY
            * rc_luminance(historyBeforeRevalidate.radiance) * reservoir.m;
        float historyM = reservoir.m;
        float qHistory = min(historyM, RC_CV_M_CAP);
        // Stored history cannot represent the reverse previous-frame shift. This is
        // the one-sided ownership-1 estimator under bijective, full-support reprojection.
        vec3 fromHistory = RC_CV_ALPHA * historyBeforeRevalidate.estimate
            + reservoir.avgWY * (reservoir.radiance - RC_CV_ALPHA * historyBeforeRevalidate.radiance);
        rc_cvAccumulatorAdd(cvAccumulator, fromHistory, qHistory);
    }

    uint selectedFlags = historyValid ? rc_reservoirMetaFlags(reservoir.meta) : 0u;
    uint selectedAge = historyValid ? min(historyAge + 1u, 255u) : 0u;
    bool selectedCandidate = false;
    bool selectedSpatial = false;
    bool spatialNeighborValid = false;

    if (historyValid) {
        float randValue = hash_uintToFloat(hash_41_q5(uvec4(entryIndex, faceId, frameCounter, 0x85EBCA6Bu)));
        selectedCandidate = rc_reservoirUpdateWeighted(
            reservoir,
            wSum,
            candidate,
            candidate.targetWeight,
            1.0,
            randValue
        );
    } else {
        reservoir = rc_reservoirInitFromCandidate(candidate);
        if (rc_reservoirValid(reservoir)) {
            wSum = candidate.targetWeight;
            selectedFlags = candidate.flags;
        }
    }

    if (selectedCandidate) {
        selectedAge = 0u;
        selectedFlags = candidate.flags;
    }

    float preSpatialTargetWeight = rc_luminance(reservoir.radiance);
    reservoir.avgWY = reservoir.m > 0.0 && wSum > 0.0 && preSpatialTargetWeight > 0.0
        ? wSum * safeRcp(reservoir.m) * safeRcp(preSpatialTargetWeight)
        : 0.0;

    #ifdef SETTING_RC_SPATIAL_ENABLE
        ivec3 neighborCell = worldCellCoord;
        vec3 neighborOrigin = vec3(0.0);
        RCReservoir neighborReservoir = rc_reservoirInit();
        spatialNeighborValid = rc_loadRandomSpatialNeighbor(
            entryIndex,
            worldCellCoord,
            level,
            faceId,
            neighborCell,
            neighborOrigin,
            neighborReservoir
        );
        RCCandidate spatialCandidate;
        float storedSourceM = neighborReservoir.m;
        float sourceM = min(storedSourceM, RC_SPATIAL_M_CAP);
        float targetM = max(reservoir.m, 0.0);
        if (spatialNeighborValid && sourceM > 0.0 && SETTING_RC_SPATIAL_STRENGTH > 0.0) {
            float sourceMIS;
            float sourceShiftWeight;
            bool sourceShiftValid = rc_generateSpatialCandidate(
                worldCellCoord,
                level,
                faceId,
                neighborOrigin,
                neighborReservoir,
                targetM,
                sourceM,
                spatialCandidate,
                sourceMIS,
                sourceShiftWeight
            );
            RCCandidate targetShiftCandidate = spatialCandidate;
            float targetMIS = 0.0;
            float targetShiftWeight = 0.0;
            bool targetShiftValid = false;
            if (targetM > 0.0 && (selectedFlags & RC_RES_FLAG_SURFACE_HIT) != 0u) {
                targetShiftValid = rc_generateSpatialCandidate(
                    neighborCell,
                    level,
                    faceId,
                    rc_faceRayOrigin(worldCellCoord, level, faceId),
                    reservoir,
                    sourceM,
                    targetM,
                    targetShiftCandidate,
                    targetMIS,
                    targetShiftWeight
                );
            }
            float spatialConfidence = min(max(targetM, 1.0) * safeRcp(sourceM), 1.0);
            float spatialStrength = SETTING_RC_SPATIAL_STRENGTH * spatialConfidence;
            float qSpatial = min(sourceM, RC_CV_M_CAP) * spatialStrength;
            float sourceMISWeight = sourceShiftValid ? sourceMIS : 1.0;
            float targetMISWeight = targetShiftValid ? targetMIS : 1.0;
            vec3 shiftedSource = sourceShiftValid ? sourceShiftWeight * spatialCandidate.radiance : vec3(0.0);
            vec3 shiftedTarget = targetShiftValid ? targetShiftWeight * targetShiftCandidate.radiance : vec3(0.0);
            vec3 targetTerm = targetMISWeight * reservoir.avgWY * (reservoir.radiance - RC_CV_ALPHA * shiftedTarget);
            vec3 sourceTerm = sourceMISWeight * neighborReservoir.avgWY * (shiftedSource - RC_CV_ALPHA * neighborReservoir.radiance);
            vec3 spatialDifference = targetTerm + sourceTerm;
            vec3 fromSpatial = RC_CV_ALPHA * neighborReservoir.estimate + spatialDifference;
            rc_cvAccumulatorAdd(cvAccumulator, fromSpatial, qSpatial);

            if (sourceShiftValid) {
                float randSpatial = hash_uintToFloat(hash_41_q5(uvec4(entryIndex, faceId, frameCounter, 0x27D4EB2Du)));
                float spatialUpdateWeight = spatialCandidate.targetWeight
                    * sourceShiftWeight
                    * neighborReservoir.avgWY
                    * sourceM
                    * sourceMISWeight
                    * spatialStrength;
                float spatialEffectiveMInc = sourceM * spatialStrength;
                selectedSpatial = rc_reservoirUpdateWeighted(
                    reservoir,
                    wSum,
                    spatialCandidate,
                    spatialUpdateWeight,
                    spatialEffectiveMInc,
                    randSpatial
                );
            }
        } else {
            spatialNeighborValid = false;
        }
    #endif

    #ifdef SETTING_RC_SPATIAL_ENABLE
    if (selectedSpatial) {
        selectedAge = 0u;
        selectedFlags = spatialCandidate.flags;
    }
    #endif

    float unclampedM = reservoir.m;
    float clampedM = clamp(unclampedM, 0.0, float(SETTING_RC_M_CAP));
    if (unclampedM > clampedM && unclampedM > 0.0) {
        wSum *= clampedM * safeRcp(unclampedM);
    }
    reservoir.m = clampedM;

    float selectedTargetWeight = rc_luminance(reservoir.radiance);
    bool reservoirValid = reservoir.m > 0.0
        && wSum > 0.0
        && selectedTargetWeight > 0.0
        && !isnan(selectedTargetWeight)
        && !any(isnan(reservoir.radiance))
        && !isnan(wSum);
    reservoir.avgWY = reservoirValid ? wSum * safeRcp(reservoir.m) * safeRcp(selectedTargetWeight) : 0.0;
    reservoir.meta = rc_packReservoirMeta(selectedAge, reservoirValid, selectedFlags);

    if (reservoirValid && rc_cvAccumulatorValid(cvAccumulator)) {
        reservoir.estimate = rc_cvAccumulatorResolve(cvAccumulator);
        if (spatialNeighborValid) {
            reservoir.meta |= 1u;
        }
    } else {
        reservoir = rc_reservoirInit();
    }

    rc_reservoirStore(rc_currentSide(), reservoirIndex, reservoir);
}

#endif
