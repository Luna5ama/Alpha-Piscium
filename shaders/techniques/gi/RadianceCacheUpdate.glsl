#ifndef INCLUDE_techniques_gi_RadianceCacheUpdate_glsl
#define INCLUDE_techniques_gi_RadianceCacheUpdate_glsl a

#define RC_DATA_MODIFIER restrict buffer
#define GLOBAL_DATA_MODIFIER restrict buffer

#include "/techniques/gi/RadianceCache.glsl"
#include "/techniques/gi/PathGuiding.glsl"
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
const float RC_CV_M_CAP = 64.0;
const float RC_SPATIAL_M_CAP = 32.0;

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

vec3 rc_cvInitialEstimate(RCCandidate candidate, float proposalWeight) {
    return candidate.valid ? candidate.radiance * proposalWeight : vec3(0.0);
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
    uint feedbackBits = rc_feedbackScreenBits(rc_faceBit(faceId));
    uvec2 feedback = rc_feedback[recordIndex];
    if (feedback.x == worldKeyHash && (feedback.y & feedbackBits) != 0u) {
        return;
    }
    uint oldKey = atomicCompSwap(rc_feedback[recordIndex].x, RC_INVALID, worldKeyHash);
    if (oldKey == RC_INVALID || oldKey == worldKeyHash) {
        atomicOr(rc_feedback[recordIndex].y, feedbackBits);
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
    if (!candidate.valid || mInc <= 0.0) {
        return false;
    }

    reservoir.m += mInc;
    if (candidate.targetWeight <= 0.0 || updateWeight <= 0.0) {
        return false;
    }

    wSum += updateWeight;
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

bool rc_isFiniteRadiance(vec3 radiance) {
    return all(lessThanEqual(abs(radiance), vec3(FLT_MAX)));
}

// valid: the radiance is finite; it may be zero.
vec3 rc_sampleHitRadiance(VoxelHit hit, vec3 outgoingDir, out bool valid) {
    valid = false;
    // Callers reject exhausted traces, so a miss here left the voxel grid and is treated as sky.
    if (!hit.hit) {
        vec3 missRadiance = rc_sampleMissRadiance(-outgoingDir);
        valid = rc_isFiniteRadiance(missRadiance);
        return valid ? missRadiance : vec3(0.0);
    }

    voxel_SurfaceData surface = voxel_sampleVoxelSurface(hit, 0.0);
    if (!surface.valid) {
        return vec3(0.0);
    }
    vec3 radiance = surface.material.emissive
        + gi_hitDirectLighting(surface.material, hit.hitPos, outgoingDir, hit.normal, hit.normal);
    valid = rc_isFiniteRadiance(radiance);
    if (!valid) {
        return vec3(0.0);
    }
    surface.material.roughness = max(surface.material.roughness, RC_MAX_ROUGHNESS * 0.5);

    RCReservoir prevReservoir;
    vec3 faceNormal;
    if (!rc_loadPreviousHitReservoir(hit, prevReservoir, faceNormal)) {
        return radiance;
    }

    float NDotV = dot(faceNormal, outgoingDir);
    if (NDotV <= 0.0) {
        return radiance;
    }

    // Same uniform-incidence shading of the face estimate as rc_lookupSampleFace.
    vec2 uniformAlbedo = resampleMaterial_uniformIncidenceAlbedo(resampleMaterial_fromMaterial(surface.material), NDotV);
    vec3 bounceRadiance = (surface.material.albedo * uniformAlbedo.x + uniformAlbedo.y) * rc_reservoirEstimateRadiance(prevReservoir);
    if (!rc_isFiniteRadiance(bounceRadiance)) {
        return radiance;
    }
    return radiance + bounceRadiance;
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
    if (voxel_traceExhausted(hit, voxelRay)) {
        return false;
    }

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

RCCandidate rc_generateCandidate(uint entryIndex, ivec3 worldCellCoord, uint level, uint faceId, bool allowHitFeedback, out float proposalWeight) {
    proposalWeight = 1.0;
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
    #ifdef PATH_GUIDING_ENABLED
        PathGuide guide;
        #if SETTING_DEBUG_PATH_GUIDE >= 1 && SETTING_DEBUG_PATH_GUIDE <= 4
            guide = pathGuide_debugLobe();
        #else
            uvec4 stats0;
            uvec4 stats1;
            pathGuide_previousStats(entryIndex, rc_worldKeyHash(level, worldCellCoord), level, faceId, stats0, stats1);
            guide = pathGuide_fit(stats0, stats1);
        #endif
        uvec2 guideBins = pathGuide_bins(guide, 0.0);
        if (guideBins.x + guideBins.y > 0u) {
            vec3 up = abs(faceNormal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
            vec3 T = normalize(cross(up, faceNormal));
            vec3 B = cross(faceNormal, T);
            mat3 worldToTangent = transpose(mat3(T, B, faceNormal));
            vec3 axis0 = worldToTangent * coords_dir_viewToWorld(guide.lobe0.axisView);
            vec3 axis1 = worldToTangent * coords_dir_viewToWorld(guide.lobe1.axisView);
            uint choiceBin = randHash.z & 255u;
            if (choiceBin < guideBins.x + guideBins.y) {
                bool firstLobe = choiceBin < guideBins.x;
                localSample.xyz = pathGuide_sampleFolded(
                    normalize(firstLobe ? axis0 : axis1),
                    firstLobe ? guide.lobe0.kappa : guide.lobe1.kappa,
                    randValue
                );
            }
            float cosinePdf = localSample.z * RCP_PI;
            vec2 fractions = vec2(guideBins) * (1.0 / 256.0);
            localSample.w = (1.0 - fractions.x - fractions.y) * cosinePdf;
            if (guideBins.x > 0u) {
                localSample.w += fractions.x * pathGuide_foldedPdf(normalize(axis0), guide.lobe0.kappa, localSample.xyz);
            }
            if (guideBins.y > 0u) {
                localSample.w += fractions.y * pathGuide_foldedPdf(normalize(axis1), guide.lobe1.kappa, localSample.xyz);
            }
            // The reservoir and spatial shifts remain in the cosine reference measure.
            proposalWeight = cosinePdf / localSample.w;
        }
    #endif
    vec3 worldDir = rc_hemisphereDirection(faceNormal, localSample.xyz);
    float cosTheta = max(dot(faceNormal, worldDir), 0.0);
    if (cosTheta <= 0.0 || localSample.w <= 0.0) {
        return candidate;
    }

    vec3 rayOrigin = rc_faceRayOrigin(worldCellCoord, level, faceId);
    VoxelRay voxelRay = voxelray_setup(rayOrigin, worldDir, 0u);
    VoxelHit hit = voxel_traceRay(voxelRay, 128);
    if (voxel_traceExhausted(hit, voxelRay)) {
        return candidate;
    }
    if (allowHitFeedback && hit.hit) {
        rc_touchHitFeedback(hit);
    }

    bool radianceValid = false;
    vec3 radiance = rc_sampleHitRadiance(hit, -worldDir, radianceValid);
    if (!radianceValid) {
        return candidate;
    }

    // A zero-radiance sample is legal: it is counted in M and the CV weight but never selected.
    candidate.radiance = radiance;
    candidate.dir = worldDir;
    if (hit.hit) {
        candidate.hitPos = hit.hitPos;
        candidate.hitNormal = hit.normal;
        candidate.flags = RC_RES_FLAG_SURFACE_HIT;
    } else {
        candidate.flags = RC_RES_FLAG_SKY_MISS;
    }
    candidate.targetWeight = rc_luminance(radiance);
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

    float proposalWeight;
    RCCandidate candidate = rc_generateCandidate(entryIndex, worldCellCoord, level, faceId, allowHitFeedback, proposalWeight);
    RCReservoir reservoir = rc_reservoirInit();
    RCCVAccumulator cvAccumulator = rc_cvAccumulatorInit();
    float qInit = candidate.valid ? 1.0 : 0.0;
    rc_cvAccumulatorAdd(cvAccumulator, rc_cvInitialEstimate(candidate, proposalWeight), qInit);

    uint prevBufferIndex = rc_bufferEntryIndex(rc_previousSide(), entryIndex);
    uvec4 prevEntry = rc_indirection[prevBufferIndex];
    bool historyValid = prevEntry.x != RC_INVALID
        && prevEntry.z == entry.z
        && rc_entryMetaValid(prevEntry.w)
        && rc_entryMetaLevel(prevEntry.w) == level
        && rc_hasFace(prevEntry.y, faceId);

    // historyValid: the previous face estimate and M carry over.
    // historySampleValid: it also holds a selected sample for RIS and revalidation.
    uint historyAge = 0u;
    bool historySampleValid = false;
    if (historyValid) {
        uint prevReservoirIndex = rc_faceReservoirIndex(prevEntry.x, prevEntry.y, faceId);
        if (prevReservoirIndex < uint(SETTING_RC_POOL_SIZE)) {
            reservoir = rc_reservoirLoad(rc_previousSide(), prevReservoirIndex);
            historyValid = rc_reservoirValid(reservoir)
                && !isnan(reservoir.m)
                && rc_isFiniteRadiance(reservoir.estimate);
            historySampleValid = historyValid
                && rc_reservoirHasSample(reservoir)
                && rc_luminance(reservoir.radiance) > 0.0
                && !isnan(reservoir.avgWY)
                && rc_isFiniteRadiance(reservoir.radiance);
            historyAge = rc_reservoirMetaAge(reservoir.meta);
        } else {
            historyValid = false;
        }
    }
    if (!historyValid) {
        reservoir = rc_reservoirInit();
    }
    float wSum = 0.0;
    RCReservoir historyBeforeRevalidate = reservoir;
    if (historySampleValid) {
        uint validateId = worldKeyHash + faceId;
        if ((validateId & 7u) == (uint(frameCounter) & 7u)) {
            historyValid = rc_revalidateHistoryReservoir(
                worldCellCoord,
                level,
                faceId,
                reservoir
            );
            historySampleValid = historyValid;
            if (!historyValid) {
                reservoir = rc_reservoirInit();
            }
        }
    }
    if (historySampleValid) {
        wSum = historyBeforeRevalidate.avgWY
            * rc_luminance(historyBeforeRevalidate.radiance) * reservoir.m;
    }
    if (historyValid) {
        float historyM = reservoir.m;
        float qHistory = min(historyM, RC_CV_M_CAP);
        // Stored history cannot represent the reverse previous-frame shift. This is
        // the one-sided ownership-1 estimator under bijective, full-support reprojection.
        vec3 fromHistory = RC_CV_ALPHA * historyBeforeRevalidate.estimate
            + reservoir.avgWY * (reservoir.radiance - RC_CV_ALPHA * historyBeforeRevalidate.radiance);
        rc_cvAccumulatorAdd(cvAccumulator, fromHistory, qHistory);
    }

    uint selectedFlags = historySampleValid ? rc_reservoirMetaFlags(reservoir.meta) : 0u;
    uint selectedAge = historyValid ? min(historyAge + 1u, 255u) : 0u;
    bool selectedSpatial = false;
    bool spatialNeighborValid = false;

    bool selectedCandidate = false;
    if (historyValid) {
        float randValue = hash_uintToFloat(hash_41_q5(uvec4(entryIndex, faceId, frameCounter, 0x85EBCA6Bu)));
        selectedCandidate = rc_reservoirUpdateWeighted(
            reservoir,
            wSum,
            candidate,
            candidate.targetWeight * proposalWeight,
            1.0,
            randValue
        );
    } else if (candidate.valid) {
        // Same result as rc_reservoirUpdateWeighted on an empty reservoir. Spelling it out keeps this kernel
        // below a register cliff (the shared call measured +43% on night-gi-1-720p).
        reservoir.m = 1.0;
        if (candidate.targetWeight > 0.0) {
            reservoir.radiance = candidate.radiance;
            reservoir.sampleDir = candidate.dir;
            reservoir.hitPos = candidate.hitPos;
            wSum = candidate.targetWeight * proposalWeight;
            selectedCandidate = true;
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
    bool sampleValid = reservoir.m > 0.0
        && wSum > 0.0
        && selectedTargetWeight > 0.0
        && !isnan(selectedTargetWeight)
        && !any(isnan(reservoir.radiance))
        && !isnan(wSum);
    bool estimateValid = reservoir.m > 0.0 && rc_cvAccumulatorValid(cvAccumulator);

    if (estimateValid) {
        reservoir.estimate = rc_cvAccumulatorResolve(cvAccumulator);
        if (sampleValid) {
            reservoir.avgWY = wSum * safeRcp(reservoir.m) * safeRcp(selectedTargetWeight);
        } else {
            // Only zero-radiance samples were counted: keep the estimate and M without a selected sample.
            reservoir.radiance = vec3(0.0);
            reservoir.avgWY = 0.0;
            reservoir.sampleDir = rc_faceNormal(faceId);
            reservoir.hitPos = vec3(0.0);
            selectedFlags = 0u;
        }
        reservoir.meta = rc_packReservoirMeta(selectedAge, true, selectedFlags);
        if (spatialNeighborValid) {
            reservoir.meta |= 1u;
        }
    } else {
        reservoir = rc_reservoirInit();
    }

    rc_reservoirStore(rc_currentSide(), reservoirIndex, reservoir);
}

#endif
