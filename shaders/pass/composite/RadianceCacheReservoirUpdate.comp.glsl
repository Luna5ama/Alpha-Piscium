#define RC_DATA_MODIFIER restrict buffer

layout(local_size_x = 128) in;

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/gi/RadianceCache.glsl"
#include "/techniques/gi/ResampleMaterial.glsl"
#include "/techniques/voxel/VoxelTrace.glsl"
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"
#include "/techniques/voxel/SurfaceData.glsl"
#include "/util/Colors2.glsl"
#include "/util/Fresnel.glsl"
#include "/util/HardcodedPBR.glsl"
#include "/util/MaterialIDConst.glsl"
#include "/util/Rand.glsl"


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

    voxel_SurfaceData surface = voxel_sampleVoxelSurface(hit, 8.0);
    if (!surface.valid) {
        return vec3(0.0);
    }
    surface.material.roughness = max(surface.material.roughness, RC_MAX_ROUGHNESS * 0.5);

    vec3 radiance = surface.material.emissive;
    valid = rc_luminance(radiance) > 0.0 && !any(isnan(radiance));

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
    if (rc_reservoirIsParentBootstrap(reservoir)) {
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
    if (rc_reservoirIsParentBootstrap(neighborReservoir)) {
        return false;
    }
    if (!rc_reservoirIsSurfaceHit(neighborReservoir)) {
        return false;
    }

    neighborOrigin = rc_faceRayOrigin(neighborCell, level, faceId);
    return true;
}

bool rc_generateParentBootstrapCandidate(
    uint level,
    ivec3 worldCellCoord,
    uint faceId,
    out RCCandidate candidate
) {
    candidate.radiance = vec3(0.0);
    candidate.dir = rc_faceNormal(faceId);
    candidate.hitPos = rc_faceCenter(worldCellCoord, level, faceId);
    candidate.hitNormal = rc_faceNormal(faceId);
    candidate.targetWeight = 0.0;
    candidate.flags = 0u;
    candidate.valid = false;

    uint parentLevel;
    ivec3 parentCell;
    RCReservoir parentReservoir;
    if (!rc_loadParentFaceReservoir(level, worldCellCoord, faceId, parentLevel, parentCell, parentReservoir)) {
        return false;
    }

    vec3 parentRadiance = rc_reservoirEstimateRadiance(parentReservoir);
    float parentTargetWeight = rc_luminance(parentRadiance);
    if (
        parentTargetWeight <= 0.0
        || isnan(parentTargetWeight)
        || any(isnan(parentRadiance))
    ) {
        return false;
    }

    candidate.radiance = parentRadiance;
    candidate.dir = rc_faceNormal(faceId);
    candidate.hitPos = rc_faceCenter(worldCellCoord, level, faceId);
    candidate.hitNormal = rc_faceNormal(faceId);
    candidate.targetWeight = parentTargetWeight;
    candidate.flags = RC_RES_FLAG_PARENT_BOOTSTRAP;
    candidate.valid = true;
    return true;
}

float rc_pairwiseSpatialMIS_MAware(
    vec3 targetOrigin,
    vec3 targetNormal,
    vec3 neighborOrigin,
    vec3 neighborNormal,
    vec3 hitPos,
    vec3 hitNormal,
    float targetM,
    float sourceM
) {
    float pTarget = rc_areaPdfCosineConnection(targetOrigin, targetNormal, hitPos, hitNormal);
    float pNeighbor = rc_areaPdfCosineConnection(neighborOrigin, neighborNormal, hitPos, hitNormal);

    if (pTarget <= 0.0 || pNeighbor <= 0.0) {
        return 0.0;
    }

    float targetMass = max(targetM, 1.0);
    float sourceMass = max(sourceM, 1.0);
    float denom = targetMass * pTarget + sourceMass * pNeighbor;
    if (denom <= 1e-6) {
        return 0.0;
    }

    return targetMass * pTarget * safeRcp(denom);
}

float rc_spatialEffectiveSourceM(RCReservoir neighborReservoir) {
    float m = neighborReservoir.m;
    if (isnan(m) || m <= 0.0) {
        return 0.0;
    }

    float maxSpatialM = min(float(SETTING_RC_M_CAP), 8.0);
    return clamp(m, 1.0, maxSpatialM);
}

float rc_spatialSourceCorrection(RCReservoir neighborReservoir) {
    float wy = neighborReservoir.avgWY;
    if (isnan(wy) || wy <= 0.0) {
        return 0.0;
    }

    return clamp(wy, 0.0, 2.0);
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
    ivec3 neighborCell,
    vec3 neighborOrigin,
    RCReservoir neighborReservoir,
    float targetM,
    float sourceM,
    out RCCandidate candidate,
    out float spatialReuseWeight,
    out float spatialMInc
) {
    candidate.radiance = vec3(0.0);
    candidate.dir = rc_faceNormal(faceId);
    candidate.hitPos = rc_faceCenter(worldCellCoord, level, faceId);
    candidate.hitNormal = vec3(0.0);
    candidate.targetWeight = 0.0;
    candidate.flags = 0u;
    candidate.valid = false;
    spatialReuseWeight = 0.0;
    spatialMInc = 0.0;

    #ifndef SETTING_RC_SPATIAL_ENABLE
        return false;
    #else
        vec3 targetNormal = rc_faceNormal(faceId);
        vec3 targetOrigin = rc_faceRayOrigin(worldCellCoord, level, faceId);

        vec3 hitPos = neighborReservoir.hitPos;
        if (any(isnan(hitPos)) || dot(hitPos, hitPos) <= 1e-6) {
            return false;
        }

        vec3 neighborDir = normalize(neighborReservoir.sampleDir);
        vec3 neighborToHit = hitPos - neighborOrigin;
        float neighborHitDist = dot(neighborToHit, neighborDir);
        if (neighborHitDist <= 0.05) {
            return false;
        }

        vec3 expectedHitDir = normalize(neighborToHit);
        if (dot(expectedHitDir, neighborDir) < 0.95) {
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
        if (
            !radianceValid
            || targetWeight <= 0.0
            || any(isnan(radiance))
            || isnan(targetWeight)
        ) {
            return false;
        }

        float misWeight = rc_pairwiseSpatialMIS_MAware(
            targetOrigin,
            targetNormal,
            neighborOrigin,
            targetNormal,
            hit.hitPos,
            hit.normal,
            targetM,
            sourceM
        );
        if (misWeight <= 0.0) {
            return false;
        }

        candidate.radiance = radiance;
        candidate.dir = shiftedDir;
        candidate.hitPos = hit.hitPos;
        candidate.hitNormal = hit.normal;
        candidate.targetWeight = targetWeight;
        candidate.flags = RC_RES_FLAG_SURFACE_HIT;
        candidate.valid = true;
        spatialReuseWeight = misWeight;
        spatialMInc = sourceM * misWeight;
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

    RCCandidate candidate = rc_generateCandidate(entryIndex, worldCellCoord, level, faceId, allowHitFeedback);
    RCReservoir reservoir = rc_reservoirInit();

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
//                reservoir.m *= global_historyResetFactor;
                historyAge = rc_reservoirMetaAge(reservoir.meta);
                historyValid = reservoir.avgWY > 0.0
                    && reservoir.m > 0.0
                && all(greaterThan(reservoir.radiance, vec3(0.0)))
                    && !isnan(reservoir.avgWY)
                    && !isnan(reservoir.m)
                    && !any(isnan(reservoir.radiance));
            }
        } else {
            historyValid = false;
        }
    }
    float wSum = 0.0;
    if (historyValid) {
        wSum = reservoir.avgWY * rc_luminance(reservoir.radiance);
        uint validateId = gl_WorkGroupID.x + (gl_WorkGroupID.x >> 3);
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

    uint selectedFlags = historyValid ? rc_reservoirMetaFlags(reservoir.meta) : 0u;
    uint selectedAge = historyValid ? min(historyAge + 1u, 255u) : 0u;
    bool selectedCandidate = false;
    bool selectedParentBootstrap = false;
    bool selectedSpatial = false;
    bool hadTemporalHistory = historyValid;
    float temporalMBeforeCurrent = historyValid ? reservoir.m : 0.0;
    bool spatialNeighborValid = false;

    if (historyValid) {
        float randValue = hash_uintToFloat(hash_41_q5(uvec4(entryIndex, faceId, frameCounter, 0x85EBCA6Bu)));
        wSum *= reservoir.m;
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

    bool parentBootstrapEligible =
        !hadTemporalHistory
        || temporalMBeforeCurrent < 2.0
        || reservoir.m < 2.0;
    if (parentBootstrapEligible) {
        RCCandidate parentCandidate;
        if (rc_generateParentBootstrapCandidate(level, worldCellCoord, faceId, parentCandidate)) {
            float parentStrength = 0.25;
            float parentUpdateWeight = parentCandidate.targetWeight * parentStrength;
            float parentMInc = parentStrength;
            float randParent = hash_uintToFloat(hash_41_q5(uvec4(entryIndex, faceId, frameCounter, 0xA24BAED4u)));
            selectedParentBootstrap = rc_reservoirUpdateWeighted(
                reservoir,
                wSum,
                parentCandidate,
                parentUpdateWeight,
                parentMInc,
                randParent
            );
        }
    }

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
        float spatialReuseWeight;
        float spatialMInc;
        float sourceM = rc_spatialEffectiveSourceM(neighborReservoir);
        float targetM = clamp(max(reservoir.m, 1.0), 1.0, float(SETTING_RC_M_CAP));
        if (spatialNeighborValid && sourceM > 0.0 && SETTING_RC_SPATIAL_STRENGTH > 0.0 && rc_generateSpatialCandidate(
            worldCellCoord,
            level,
            faceId,
            neighborCell,
            neighborOrigin,
            neighborReservoir,
            targetM,
            sourceM,
            spatialCandidate,
            spatialReuseWeight,
            spatialMInc
        )) {
            float randSpatial = hash_uintToFloat(hash_41_q5(uvec4(entryIndex, faceId, frameCounter, 0x27D4EB2Du)));
            float sourceCorrection = rc_spatialSourceCorrection(neighborReservoir);
            float spatialStrength = SETTING_RC_SPATIAL_STRENGTH;
            float spatialUpdateWeight =
                spatialCandidate.targetWeight *
                sourceCorrection *
                spatialMInc *
                spatialStrength;
            float spatialEffectiveMInc = spatialMInc * spatialStrength;
            selectedSpatial = rc_reservoirUpdateWeighted(
                reservoir,
                wSum,
                spatialCandidate,
                spatialUpdateWeight,
                spatialEffectiveMInc,
                randSpatial
            );
        } else {
            spatialNeighborValid = false;
        }
    #endif

    if (selectedCandidate) {
        selectedAge = 0u;
        selectedFlags = candidate.flags;
    }
    if (selectedParentBootstrap) {
        selectedAge = 0u;
        selectedFlags = RC_RES_FLAG_PARENT_BOOTSTRAP;
    }
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

    if (spatialNeighborValid) {
        reservoir.meta |= 1u;
    }

    rc_reservoirStore(rc_currentSide(), reservoirIndex, reservoir);
}

void main() {
    voxel_initShared();

    if (gl_GlobalInvocationID.x < rc_entryCounter) {
        uint data = rc_updateEntryIndices[gl_GlobalInvocationID.x];
        uint entryIndex = bitfieldExtract(data, 0, 26);
        uint faceId = bitfieldExtract(data, 26, 6);
        if (entryIndex < RC_ENTRY_COUNT) {
            uint level = rc_entryLevel(entryIndex);
            ivec3 worldCellCoord = rc_worldCellCoordFromEntryIndex(entryIndex);
            uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), entryIndex);
            uvec4 entry = rc_indirection[bufferIndex];
            if (entry.x != RC_INVALID && entry.z == rc_worldKeyHash(level, worldCellCoord) && rc_entryMetaValid(entry.w) && rc_entryMetaLevel(entry.w) == level) {
                uint faceMask = entry.y & 0x3fu;
                if (rc_hasFace(faceMask, faceId)) {
                    rc_updateFace(entryIndex, entry, worldCellCoord, level, faceId);
                }
            }
        }
    }
}
