#define RC_DATA_MODIFIER restrict buffer

layout(local_size_x = 128) in;

// Indirect dispatch dimensions are written by RadianceCacheAllocate.

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/gi/RadianceCache.glsl"
#include "/techniques/gi/ResampleMaterial.glsl"
#include "/techniques/voxel/VoxelTrace.glsl"
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"
#include "/util/Colors2.glsl"
#include "/util/Fresnel.glsl"
#include "/util/HardcodedPBR.glsl"
#include "/util/MaterialIDConst.glsl"
#include "/util/Rand.glsl"


vec3 rcHemisphereDirection(vec3 normal, vec3 localDir) {
    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, normal));
    vec3 B = cross(normal, T);
    return normalize(T * localDir.x + B * localDir.y + normal * localDir.z);
}

void rcTouchHit(VoxelHit hit) {
    uint faceId = rcFaceIdFromNormal(hit.normal);
    vec3 faceNormal = rcFaceNormal(faceId);
    vec3 surfacePos = hit.hitPos - faceNormal * 0.02;
    for (uint level = 0u; level < RC_CLIP_LEVELS; level++) {
        ivec3 worldCellCoord = rcWorldCellCoord(surfacePos, level);
        rcTouchFace(level, worldCellCoord, faceId);
    }
}

bool rcLoadPreviousHitReservoir(
    VoxelHit hit,
    out RCReservoir reservoir,
    out vec3 faceNormal
) {
    uint faceId = rcFaceIdFromNormal(hit.normal);
    faceNormal = rcFaceNormal(faceId);
    vec3 surfacePos = hit.hitPos - faceNormal * 0.02;
    uint level = rcSelectLevel(surfacePos);
    ivec3 worldCellCoord = rcWorldCellCoord(surfacePos, level);
    return rcLoadFaceReservoir(rcPreviousSide(), level, worldCellCoord, faceId, reservoir);
}

struct RCHitSurface {
    vec3 albedo;
    vec3 emissive;
    ResampleMaterial material;
    bool valid;
};

RCHitSurface rcHitSurfaceInit() {
    RCHitSurface surface;
    surface.albedo = vec3(0.0);
    surface.emissive = vec3(0.0);
    surface.material = resampleMaterial_init();
    surface.valid = false;
    return surface;
}

RCHitSurface rcSampleHitSurface(VoxelHit hit) {
    RCHitSurface surface = rcHitSurfaceInit();
    if (!hit.hit || hit.materialID == 0u || hit.materialID == MATERIAL_ID_WATER) {
        return surface;
    }

    HardcodedPBR hardcoded = hardcodedpbr_decode(hit.materialID);
    uint faceId = voxel_faceIndexFromNormal(hit.normal);
    uvec2 tcData = voxel_faceTexcoords[voxel_faceTexcoordIndex(hit.materialID, faceId)];
    vec4 tc = unpackUnorm4x16(tcData);
    if (all(equal(tc, vec4(0.0)))) {
        return surface;
    }

    vec2 localUV = voxel_faceLocalUV(faceId, hit.hitPos);
    vec2 atlasUV = mix(tc.xw, tc.zy, localUV);
    vec3 baseColor = colors2_material_toWorkSpace(texture(usam_blockAtlasColor, atlasUV).rgb);
    if (any(isnan(baseColor))) {
        return surface;
    }

    surface.albedo = baseColor;
    surface.material.f0 = fresnel_iorToF0(max(hardcoded.ior, AIR_IOR));
    surface.material.dielectric = 1.0;
    surface.material.roughness = max(pow2(hardcoded.roughness), 0.001);

    float emissiveScale = hardcoded.emissive * exp2(float(hardcoded.emissiveMultiplier));
    surface.emissive = baseColor * emissiveScale;
    surface.valid = true;
    return surface;
}

vec3 rcSampleMissRadiance(vec3 rayDir) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, rayDir);
    return atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
}

vec3 rcSampleHitRadiance(VoxelHit hit, vec3 outgoingDir, out bool valid) {
    valid = false;
    if (!hit.hit) {
        vec3 missRadiance = rcSampleMissRadiance(normalize(-outgoingDir));
        valid = rcLuminance(missRadiance) > 0.0 && !any(isnan(missRadiance));
        return valid ? missRadiance : vec3(0.0);
    }

    RCHitSurface surface = rcSampleHitSurface(hit);
    if (!surface.valid) {
        return vec3(0.0);
    }

    vec3 radiance = surface.emissive;
    valid = rcLuminance(radiance) > 0.0 && !any(isnan(radiance));

    RCReservoir prevReservoir;
    vec3 faceNormal;
    if (!rcLoadPreviousHitReservoir(hit, prevReservoir, faceNormal)) {
        return radiance;
    }

    vec3 incomingDir = normalize(prevReservoir.sampleDir);
    vec3 viewDir = normalize(outgoingDir);
    float NDotL = dot(faceNormal, incomingDir);
    float NDotV = dot(faceNormal, viewDir);
    if (NDotL <= 0.0 || NDotV <= 0.0) {
        return radiance;
    }

    vec3 incomingRadiance = rcReservoirEstimateRadiance(prevReservoir);
    if (rcLuminance(incomingRadiance) <= 0.0 || any(isnan(incomingRadiance)) || any(isnan(incomingDir))) {
        return radiance;
    }

    vec3 H = incomingDir + viewDir;
    float invHLen = inversesqrt(max(dot(H, H), 1e-6));
    float NDotH = saturate(dot(faceNormal, H * invHLen));
    float LDotH = saturate(dot(incomingDir, H * invHLen));
    ResampleBRDF brdf = resampleMaterial_evalBRDF(surface.material, NDotL, NDotV, NDotH, LDotH);
    if (brdf.full <= 0.0) {
        return radiance;
    }

    vec3 bounceFactor = surface.albedo * brdf.diffuse + vec3(brdf.specular);
    vec3 bounceRadiance = incomingRadiance * bounceFactor;
    if (rcLuminance(bounceRadiance) <= 0.0 || any(isnan(bounceRadiance))) {
        return radiance;
    }

    radiance += bounceRadiance;
    valid = true;
    return radiance;
}

bool rcRevalidateHistoryReservoir(
    ivec3 worldCellCoord,
    uint level,
    uint faceId,
    inout RCReservoir reservoir,
    out float targetWeight
) {
    targetWeight = 0.0;

    if (!rcReservoirValid(reservoir)) {
        return false;
    }

    vec3 sampleDir = normalize(reservoir.sampleDir);
    if (any(isnan(sampleDir))) {
        return false;
    }

    vec3 rayOrigin = rcFaceRayOrigin(worldCellCoord, level, faceId);
    VoxelRay voxelRay = voxelray_setup(rayOrigin, sampleDir, 0u);
    VoxelHit hit = voxel_traceRay(voxelRay, 128);

    uint flags = rcReservoirMetaFlags(reservoir.meta);
    bool expectSurfaceHit = (flags & RC_RES_FLAG_SURFACE_HIT) != 0u;
    bool expectSkyMiss = (flags & RC_RES_FLAG_SKY_MISS) != 0u;

    if (expectSurfaceHit) {
        if (!hit.hit) {
            return false;
        }

        float hitThreshold = max(float(rcVoxelSize(level)) * 0.25, 0.1);
        if (length(hit.hitPos - reservoir.hitPos) > hitThreshold) {
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
    vec3 radiance = rcSampleHitRadiance(hit, -sampleDir, radianceValid);
    targetWeight = rcLuminance(radiance);
    if (
        !radianceValid
        || targetWeight <= 0.0
        || any(isnan(radiance))
        || isnan(targetWeight)
    ) {
        return false;
    }

    reservoir.radiance = radiance;
    if (hit.hit) {
        reservoir.hitPos = hit.hitPos;
        flags = RC_RES_FLAG_SURFACE_HIT;
    } else {
        flags = RC_RES_FLAG_SKY_MISS;
    }
    reservoir.meta = rcPackReservoirMeta(rcReservoirMetaAge(reservoir.meta), true, flags);
    return true;
}

bool rcLoadRandomSpatialNeighbor(
    uint entryIndex,
    ivec3 worldCellCoord,
    uint level,
    uint faceId,
    out ivec3 neighborCell,
    out vec3 neighborOrigin,
    out RCReservoir neighborReservoir
) {
    neighborCell = worldCellCoord;
    neighborOrigin = rcFaceRayOrigin(worldCellCoord, level, faceId);
    neighborReservoir = rcReservoirInit();

    uint neighborIndex = hash_41_q3(uvec4(entryIndex, faceId, frameCounter, 0xC2B2AE35u)) & 7u;
    ivec2 neighborOffset = rcNeighborOffset8(neighborIndex);
    neighborCell = worldCellCoord + rcNeighborPlaneOffset(faceId, neighborOffset.x, neighborOffset.y);

    vec3 targetCenter = rcFaceCenter(worldCellCoord, level, faceId);
    vec3 neighborCenter = rcFaceCenter(neighborCell, level, faceId);
    float maxDistance = max(float(rcVoxelSize(level)) * SETTING_RC_SPATIAL_MAX_DIST, 1e-3);
    if (length(neighborCenter - targetCenter) > maxDistance) {
        return false;
    }

    if (!rcLoadFaceReservoir(rcPreviousSide(), level, neighborCell, faceId, neighborReservoir)) {
        return false;
    }
    if (!rcReservoirIsSurfaceHit(neighborReservoir)) {
        return false;
    }

    neighborOrigin = rcFaceRayOrigin(neighborCell, level, faceId);
    return true;
}

float rcPairwiseSpatialMIS(
    vec3 targetOrigin,
    vec3 targetNormal,
    vec3 neighborOrigin,
    vec3 neighborNormal,
    vec3 hitPos,
    vec3 hitNormal
) {
    #ifndef SETTING_RC_SPATIAL_USE_MIS
        return 1.0;
    #else
        float pTarget = 0.0;
        float pNeighbor = 0.0;

        #ifdef SETTING_RC_SPATIAL_USE_JACOBIAN
            pTarget = rcAreaPdfCosineConnection(targetOrigin, targetNormal, hitPos, hitNormal);
            pNeighbor = rcAreaPdfCosineConnection(neighborOrigin, neighborNormal, hitPos, hitNormal);
        #else
            vec3 targetToHit = hitPos - targetOrigin;
            float targetDistanceSq = dot(targetToHit, targetToHit);
            if (targetDistanceSq > 1e-6) {
                pTarget = max(dot(targetNormal, normalize(targetToHit)), 0.0) * RCP_PI;
            }

            vec3 neighborToHit = hitPos - neighborOrigin;
            float neighborDistanceSq = dot(neighborToHit, neighborToHit);
            if (neighborDistanceSq > 1e-6) {
                pNeighbor = max(dot(neighborNormal, normalize(neighborToHit)), 0.0) * RCP_PI;
            }
        #endif

        if (pTarget <= 0.0) {
            return 0.0;
        }

        float pSpatial = pNeighbor * 0.125;
        float pSum = pTarget + pSpatial;
        if (pSum <= 1e-6) {
            return 0.0;
        }

        return pTarget * safeRcp(pSum);
    #endif
}

float rcPairwiseSpatialMIS_MAware(
    vec3 targetOrigin,
    vec3 targetNormal,
    vec3 neighborOrigin,
    vec3 neighborNormal,
    vec3 hitPos,
    vec3 hitNormal,
    float targetM,
    float sourceM
) {
    #ifndef SETTING_RC_SPATIAL_USE_MIS
        return 1.0;
    #else
        float pTarget = 0.0;
        float pNeighbor = 0.0;

        #ifdef SETTING_RC_SPATIAL_USE_JACOBIAN
            pTarget = rcAreaPdfCosineConnection(targetOrigin, targetNormal, hitPos, hitNormal);
            pNeighbor = rcAreaPdfCosineConnection(neighborOrigin, neighborNormal, hitPos, hitNormal);
        #else
            vec3 targetToHit = hitPos - targetOrigin;
            float targetDistanceSq = dot(targetToHit, targetToHit);
            if (targetDistanceSq > 1e-6) {
                pTarget = max(dot(targetNormal, normalize(targetToHit)), 0.0) * RCP_PI;
            }

            vec3 neighborToHit = hitPos - neighborOrigin;
            float neighborDistanceSq = dot(neighborToHit, neighborToHit);
            if (neighborDistanceSq > 1e-6) {
                pNeighbor = max(dot(neighborNormal, normalize(neighborToHit)), 0.0) * RCP_PI;
            }
        #endif

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
    #endif
}

float rcSpatialEffectiveSourceM(RCReservoir neighborReservoir) {
    float m = neighborReservoir.m;
    if (isnan(m) || m <= 0.0) {
        return 0.0;
    }

    float maxSpatialM = min(float(SETTING_RC_M_MAX), 8.0);
    return clamp(m, 1.0, maxSpatialM);
}

float rcSpatialSourceCorrection(RCReservoir neighborReservoir) {
    float wy = neighborReservoir.avgWY;
    if (isnan(wy) || wy <= 0.0) {
        return 0.0;
    }

    return clamp(wy, 0.0, 2.0);
}

RCCandidate rcGenerateCandidate(uint entryIndex, ivec3 worldCellCoord, uint level, uint faceId) {
    RCCandidate candidate;
    candidate.radiance = vec3(0.0);
    candidate.dir = rcFaceNormal(faceId);
    candidate.hitPos = rcFaceCenter(worldCellCoord, level, faceId);
    candidate.hitNormal = vec3(0.0);
    candidate.targetWeight = 0.0;
    candidate.flags = 0u;
    candidate.valid = false;

    vec3 faceNormal = rcFaceNormal(faceId);
    uvec4 randHash = hash_44_q3(uvec4(entryIndex, faceId, frameCounter, 0x9E3779B9u));
    vec2 randValue = hash_uintToFloat(randHash.xy);
    vec4 localSample = rand_sampleInCosineWeightedHemisphere(randValue);
    vec3 worldDir = rcHemisphereDirection(faceNormal, localSample.xyz);
    float cosTheta = max(dot(faceNormal, worldDir), 0.0);
    if (cosTheta <= 0.0 || localSample.w <= 0.0) {
        return candidate;
    }

    vec3 rayOrigin = rcFaceRayOrigin(worldCellCoord, level, faceId);
    VoxelRay voxelRay = voxelray_setup(rayOrigin, worldDir, 0u);
    VoxelHit hit = voxel_traceRay(voxelRay, 128);
    if (hit.hit) {
        //rcTouchHit(hit); TODO: move to another pass?
    }

    bool radianceValid = false;
    vec3 radiance = rcSampleHitRadiance(hit, -worldDir, radianceValid);
    float targetWeight = rcLuminance(radiance);
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

bool rcGenerateSpatialCandidate(
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
    candidate.dir = rcFaceNormal(faceId);
    candidate.hitPos = rcFaceCenter(worldCellCoord, level, faceId);
    candidate.hitNormal = vec3(0.0);
    candidate.targetWeight = 0.0;
    candidate.flags = 0u;
    candidate.valid = false;
    spatialReuseWeight = 0.0;
    spatialMInc = 0.0;

    #ifndef SETTING_RC_SPATIAL_ENABLE
        return false;
    #else
        vec3 targetNormal = rcFaceNormal(faceId);
        vec3 targetOrigin = rcFaceRayOrigin(worldCellCoord, level, faceId);

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
        float distanceSq = dot(toHit, toHit);
        if (distanceSq <= 1e-6) {
            return false;
        }

        vec3 shiftedDir = toHit * inversesqrt(distanceSq);
        float targetCos = dot(targetNormal, shiftedDir);
        if (targetCos <= 0.05) {
            return false;
        }

        VoxelRay ray = voxelray_setup(targetOrigin, shiftedDir, 0u);
        VoxelHit hit = voxel_traceRay(ray, 128);
        if (!hit.hit) {
            return false;
        }

        float hitThreshold = max(float(rcVoxelSize(level)) * 0.25, 0.1);
        if (length(hit.hitPos - hitPos) > hitThreshold) {
            return false;
        }
        if (dot(hit.normal, -shiftedDir) <= 0.0) {
            return false;
        }

        bool radianceValid = false;
        vec3 radiance = rcSampleHitRadiance(hit, -shiftedDir, radianceValid);
        float targetWeight = rcLuminance(radiance);
        if (
            !radianceValid
            || targetWeight <= 0.0
            || any(isnan(radiance))
            || isnan(targetWeight)
        ) {
            return false;
        }

        float misWeight = rcPairwiseSpatialMIS_MAware(
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

void rcUpdateFace(uint entryIndex, uvec4 entry, ivec3 worldCellCoord, uint level, uint faceId) {
    uint reservoirIndex = rcFaceReservoirIndex(entry.x, entry.y, faceId);
    if (reservoirIndex >= uint(SETTING_RC_POOL_SIZE)) {
        return;
    }

    RCCandidate candidate = rcGenerateCandidate(entryIndex, worldCellCoord, level, faceId);
    RCReservoir reservoir = rcReservoirInit();

    uint prevBufferIndex = rcBufferEntryIndex(rcPreviousSide(), entryIndex);
    uvec4 prevEntry = rc_indirection[prevBufferIndex];
    bool historyValid = prevEntry.x != RC_INVALID
        && prevEntry.z == entry.z
        && rcEntryMetaValid(prevEntry.w)
        && rcEntryMetaLevel(prevEntry.w) == level
        && rcHasFace(prevEntry.y, faceId);

    uint historyAge = 0u;
    float reservoirTargetWeight = 0.0;
    if (historyValid) {
        uint prevReservoirIndex = rcFaceReservoirIndex(prevEntry.x, prevEntry.y, faceId);
        if (prevReservoirIndex < uint(SETTING_RC_POOL_SIZE)) {
            reservoir = rcReservoirLoad(rcPreviousSide(), prevReservoirIndex);
            historyValid = rcReservoirValid(reservoir);
            if (historyValid) {
//                reservoir.m *= global_historyResetFactor;
                historyAge = rcReservoirMetaAge(reservoir.meta);
                reservoirTargetWeight = rcLuminance(reservoir.radiance);
                historyValid = reservoir.avgWY > 0.0
                    && reservoir.m > 0.0
                    && reservoirTargetWeight > 0.0
                    && !isnan(reservoir.avgWY)
                    && !isnan(reservoir.m)
                    && !isnan(reservoirTargetWeight);
            }
        } else {
            historyValid = false;
        }
    }
    float randKill = hash_uintToFloat(hash_41_q3(uvec4(entryIndex, faceId, frameCounter, 0x1145CA6Bu)));
    // 100% chance to kill reservoir at each frame on max age.
    if (historyValid && randKill * 65536.0 < pow2(float(historyAge))) {
        reservoir.m *= 0.1;
        historyAge = 0u;
        historyValid = rcRevalidateHistoryReservoir(
            worldCellCoord,
            level,
            faceId,
            reservoir,
            reservoirTargetWeight
        );
        if (!historyValid) {
            reservoir = rcReservoirInit();
            reservoirTargetWeight = 0.0;
        }
    }

    float wSum = 0.0;
    float selectedTargetWeight = 0.0;
    uint selectedFlags = historyValid ? rcReservoirMetaFlags(reservoir.meta) : 0u;
    uint selectedAge = historyValid ? min(historyAge + 1u, 255u) : 0u;
    bool selectedCandidate = false;
    bool selectedSpatial = false;

    ivec3 neighborCell = worldCellCoord;
    vec3 neighborOrigin = rcFaceRayOrigin(worldCellCoord, level, faceId);
    RCReservoir neighborReservoir = rcReservoirInit();
    bool spatialNeighborValid = rcLoadRandomSpatialNeighbor(
        entryIndex,
        worldCellCoord,
        level,
        faceId,
        neighborCell,
        neighborOrigin,
        neighborReservoir
    );

    if (historyValid) {
        float randValue = hash_uintToFloat(hash_41_q3(uvec4(entryIndex, faceId, frameCounter, 0x85EBCA6Bu)));
        wSum = reservoir.avgWY * reservoir.m * reservoirTargetWeight;
        selectedCandidate = rcReservoirUpdateWeighted(
            reservoir,
            wSum,
            candidate,
            candidate.targetWeight,
            1.0,
            randValue
        );
        selectedTargetWeight = selectedCandidate ? candidate.targetWeight : reservoirTargetWeight;
    } else {
        rcReservoirInitFromCandidate(reservoir, candidate);
        if (rcReservoirValid(reservoir)) {
            wSum = candidate.targetWeight;
            selectedTargetWeight = candidate.targetWeight;
            selectedFlags = candidate.flags;
        }
    }

    #ifdef SETTING_RC_SPATIAL_ENABLE
        RCCandidate spatialCandidate;
        float spatialReuseWeight;
        float spatialMInc;
        float sourceM = rcSpatialEffectiveSourceM(neighborReservoir);
        float targetM = clamp(max(reservoir.m, 1.0), 1.0, float(SETTING_RC_M_MAX));
        if (spatialNeighborValid && sourceM > 0.0 && SETTING_RC_SPATIAL_STRENGTH > 0.0 && rcGenerateSpatialCandidate(
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
            float randSpatial = hash_uintToFloat(hash_41_q3(uvec4(entryIndex, faceId, frameCounter, 0x27D4EB2Du)));
            float sourceCorrection = rcSpatialSourceCorrection(neighborReservoir);
            float spatialStrength = SETTING_RC_SPATIAL_STRENGTH;
            float spatialUpdateWeight =
                spatialCandidate.targetWeight *
                sourceCorrection *
                spatialMInc *
                spatialStrength;
            float spatialEffectiveMInc = spatialMInc * spatialStrength;
            selectedSpatial = rcReservoirUpdateWeighted(
                reservoir,
                wSum,
                spatialCandidate,
                spatialUpdateWeight,
                spatialEffectiveMInc,
                randSpatial
            );
            if (selectedSpatial) {
                selectedTargetWeight = spatialCandidate.targetWeight;
            }
        }
    #endif

    if (selectedCandidate) {
        selectedAge = 0u;
        selectedFlags = candidate.flags;
    }
    #ifdef SETTING_RC_SPATIAL_ENABLE
    if (selectedSpatial) {
        selectedAge = 0u;
        selectedFlags = spatialCandidate.flags;
    }
    #endif

    float unclampedM = reservoir.m;
    float clampedM = clamp(unclampedM, 0.0, float(SETTING_RC_M_MAX));
    if (unclampedM > clampedM && unclampedM > 0.0) {
        wSum *= clampedM * safeRcp(unclampedM);
    }
    reservoir.m = clampedM;

    bool reservoirValid = reservoir.m > 0.0
        && selectedTargetWeight > 0.0
        && wSum > 0.0
        && !isnan(wSum);
    reservoir.avgWY = reservoirValid ? wSum * safeRcp(reservoir.m) * safeRcp(selectedTargetWeight) : 0.0;
    reservoir.meta = rcPackReservoirMeta(selectedAge, reservoirValid, selectedFlags);

    rcReservoirStore(rcCurrentSide(), reservoirIndex, reservoir);
}

void main() {
    voxel_initShared();

    uint entryIndex = gl_GlobalInvocationID.x;
    if (entryIndex >= RC_ENTRY_COUNT) {
        return;
    }

    uint level = rcEntryLevel(entryIndex);
    ivec3 worldCellCoord = rcWorldCellCoordFromEntryIndex(entryIndex);
    uint bufferIndex = rcBufferEntryIndex(rcCurrentSide(), entryIndex);
    uvec4 entry = rc_indirection[bufferIndex];
    if (entry.x == RC_INVALID || entry.z != rcWorldKeyHash(level, worldCellCoord) || !rcEntryMetaValid(entry.w) || rcEntryMetaLevel(entry.w) != level) {
        return;
    }

    uint faceMask = entry.y & 0x3fu;
    for (uint faceId = 0u; faceId < 6u; faceId++) {
        if (rcHasFace(faceMask, faceId)) {
            rcUpdateFace(entryIndex, entry, worldCellCoord, level, faceId);
        }
    }
}
