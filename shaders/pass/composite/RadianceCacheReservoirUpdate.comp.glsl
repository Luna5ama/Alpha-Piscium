#define RC_DATA_MODIFIER restrict buffer

layout(local_size_x = 256) in;

#include "/Base.glsl"
#include "/techniques/gi/RadianceCache.glsl"
#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/ResampleMaterial.glsl"
#include "/techniques/voxel/VoxelTrace.glsl"
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"
#include "/util/Colors2.glsl"
#include "/util/Fresnel.glsl"
#include "/util/HardcodedPBR.glsl"
#include "/util/MaterialIDConst.glsl"
#include "/util/Rand.glsl"

const ivec3 workGroups = ivec3(5120, 1, 1);

vec3 rcWorldToViewDir(vec3 worldDir) {
    return normalize(mat3(gbufferModelView) * worldDir);
}

vec3 rcHemisphereDirection(vec3 normal, vec3 localDir) {
    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, normal));
    vec3 B = cross(normal, T);
    return normalize(T * localDir.x + B * localDir.y + normal * localDir.z);
}

vec2 rcFaceLocalUV(uint faceId, vec3 hitPos) {
    vec3 f = fract(hitPos);
    uint faceAxis = faceId >> 1u;
    bool positiveFace = (faceId & 1u) == 0u;
    if (faceAxis == 0u) {
        return vec2(positiveFace ? 1.0 - f.z : f.z, f.y);
    } else if (faceAxis == 1u) {
        return vec2(f.x, positiveFace ? 1.0 - f.z : f.z);
    }
    return vec2(positiveFace ? f.x : 1.0 - f.x, f.y);
}

ReSTIRReservoir rcLoadPreviousTemporalReservoir(ivec2 texelPos) {
    uvec4 prevTemporalReservoirData = bool(frameCounter & 1)
        ? history_restir_reservoirTemporal2_fetch(texelPos)
        : history_restir_reservoirTemporal1_fetch(texelPos);
    return restir_reservoir_unpack(prevTemporalReservoirData);
}

vec3 rcDecodeHistoryNormal(vec4 packedData, vec3 fallbackNormal) {
    vec3 normal = packedData.xyz * 2.0 - 1.0;
    float normalLen2 = dot(normal, normal);
    return normalLen2 > 1e-6 ? normal * inversesqrt(normalLen2) : fallbackNormal;
}

void rcTouchFace(uint level, ivec3 worldCellCoord, uint faceId) {
    uint entryIndex = rcEntryIndex(level, worldCellCoord);
    uint bufferIndex = rcBufferEntryIndex(rcCurrentSide(), entryIndex);
    uint worldKeyHash = rcWorldKeyHash(level, worldCellCoord);
    uint oldKey = atomicCompSwap(rc_indirection[bufferIndex].z, RC_INVALID, worldKeyHash);
    if (oldKey == RC_INVALID || oldKey == worldKeyHash) {
        uvec4 entry = rc_indirection[bufferIndex];
        uint oldFaceMask = entry.y & 0x3fu;
        uint newFaceMask = oldFaceMask | rcFaceBit(faceId);
        bool canGrowFaceMask = entry.x == RC_INVALID || newFaceMask == oldFaceMask;
        if (!canGrowFaceMask) {
            uint allocatedClassSize = rcAllocClassSize(bitCount(oldFaceMask));
            canGrowFaceMask = bitCount(newFaceMask) <= allocatedClassSize;
        }
        if (!canGrowFaceMask) {
            return;
        }

        atomicOr(rc_indirection[bufferIndex].y, rcFaceBit(faceId));
        rc_indirection[bufferIndex].w = rcPackEntryMeta(level, 0u, true);
    } else {
        atomicAdd(rc_keyMismatchCounter, 1u);
    }
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

bool rcProjectHitToPrevFrame(
    VoxelHit hit,
    out ivec2 prevTexelPos,
    out vec3 prevViewHitPos,
    out vec3 prevViewGeomNormal,
    out vec3 prevViewNormal
) {
    vec3 hitScenePos = hit.hitPos - cameraPosition;
    vec3 prevScenePos = hitScenePos + uval_cameraDelta;
    prevViewHitPos = (gbufferPrevModelView * vec4(prevScenePos, 1.0)).xyz;

    vec4 prevClipPos = global_prevCamProj * vec4(prevViewHitPos, 1.0);
    uint clipFlag = uint(prevClipPos.z > 0.0);
    clipFlag &= uint(all(lessThan(abs(prevClipPos.xy), prevClipPos.ww)));
    vec2 prevScreenPos = prevClipPos.xy / prevClipPos.w * 0.5 + 0.5;
    clipFlag &= uint(all(equal(saturate(prevScreenPos), prevScreenPos)));
    if (!bool(clipFlag)) {
        return false;
    }

    prevTexelPos = ivec2(clamp(prevScreenPos * uval_mainImageSize, vec2(0.0), uval_mainImageSize - 1.0));
    float historyViewZ = history_viewZ_fetch(prevTexelPos).x;
    if (historyViewZ <= -65536.0) {
        return false;
    }

    vec2 historyScreenPos = coords_texelToUV(prevTexelPos, uval_mainImageSizeRcp);
    vec3 historyViewPos = coords_toViewCoord(historyScreenPos, historyViewZ, global_prevCamProjInverse);
    prevViewGeomNormal = coords_dir_worldToViewPrev(hit.normal);
    vec3 historyGeomNormal = rcDecodeHistoryNormal(history_geomViewNormal_fetch(prevTexelPos), prevViewGeomNormal);
    if (dot(prevViewGeomNormal, historyGeomNormal) < 0.75) {
        return false;
    }

    float planeDistance = gi_planeDistance(prevViewHitPos, prevViewGeomNormal, historyViewPos, historyGeomNormal);
    float planeThreshold = max(0.25, abs(prevViewHitPos.z) * 0.01);
    if (planeDistance > planeThreshold) {
        return false;
    }

    prevViewGeomNormal = historyGeomNormal;
    prevViewNormal = rcDecodeHistoryNormal(history_viewNormal_fetch(prevTexelPos), prevViewGeomNormal);
    if (dot(prevViewNormal, prevViewGeomNormal) <= 0.0) {
        prevViewNormal = prevViewGeomNormal;
    }
    return true;
}

ResampleMaterial rcResampleMaterialFromVoxelHit(VoxelHit hit) {
    HardcodedPBR hardcoded = hardcodedpbr_decode(hit.materialID);
    ResampleMaterial material = resampleMaterial_init();
    material.f0 = fresnel_iorToF0(max(hardcoded.ior, AIR_IOR));
    material.dielectric = 1.0;
    material.roughness = max(pow2(hardcoded.roughness), 0.001);
    return material;
}

vec3 rcSamplePreviousReservoirBounce(VoxelHit hit, vec3 outgoingDir, out bool valid) {
    valid = false;
    if (hit.materialID == 0u || hit.materialID == MATERIAL_ID_WATER) {
        return vec3(0.0);
    }

    ivec2 prevTexelPos = ivec2(0);
    vec3 prevViewHitPos = vec3(0.0);
    vec3 prevViewGeomNormal = vec3(0.0);
    vec3 prevViewNormal = vec3(0.0);
    if (!rcProjectHitToPrevFrame(hit, prevTexelPos, prevViewHitPos, prevViewGeomNormal, prevViewNormal)) {
        return vec3(0.0);
    }

    ReSTIRReservoir prevReservoir = rcLoadPreviousTemporalReservoir(prevTexelPos);
    if (!restir_isReservoirValid(prevReservoir) || prevReservoir.avgWY <= 0.0) {
        return vec3(0.0);
    }

    vec4 prevSample = history_restir_prevSample_fetch(prevTexelPos);
    if (prevSample.w <= 0.0 || any(isnan(prevSample.rgb))) {
        return vec3(0.0);
    }

    vec3 outgoingDirPrevView = coords_dir_worldToViewPrev(outgoingDir);
    float NDotL = dot(prevViewNormal, prevReservoir.Y.xyz);
    float NDotV = dot(prevViewNormal, outgoingDirPrevView);
    if (NDotL <= 0.0 || NDotV <= 0.0) {
        return vec3(0.0);
    }

    vec3 H = prevReservoir.Y.xyz + outgoingDirPrevView;
    float invHLen = inversesqrt(max(dot(H, H), 1e-6));
    float NDotH = saturate(dot(prevViewNormal, H * invHLen));
    float LDotH = saturate(dot(prevReservoir.Y.xyz, H * invHLen));
    ResampleBRDF brdf = resampleMaterial_evalBRDF(rcResampleMaterialFromVoxelHit(hit), NDotL, NDotV, NDotH, LDotH);
    if (brdf.full <= 0.0) {
        return vec3(0.0);
    }

    vec3 bounceRadiance = prevSample.rgb * brdf.full * prevReservoir.avgWY;
    valid = rcLuminance(bounceRadiance) > 0.0 && !any(isnan(bounceRadiance));
    return bounceRadiance;
}

vec3 rcSampleVoxelRadiance(VoxelHit hit, out bool valid) {
    valid = false;
    if (!hit.hit || hit.materialID == 0u || hit.materialID == MATERIAL_ID_WATER) {
        return vec3(0.0);
    }

    HardcodedPBR hardcoded = hardcodedpbr_decode(hit.materialID);
    if (hardcoded.emissive <= 0.0 && hardcoded.emissiveMultiplier <= 0) {
        return vec3(0.0);
    }

    uint faceId = voxel_faceIndexFromNormal(hit.normal);
    uvec2 tcData = voxel_faceTexcoords[voxel_faceTexcoordIndex(hit.materialID, faceId)];
    vec4 tc = unpackUnorm4x16(tcData);
    if (all(equal(tc, vec4(0.0)))) {
        return vec3(0.0);
    }

    vec2 localUV = rcFaceLocalUV(faceId, hit.hitPos);
    vec2 atlasUV = mix(tc.xw, tc.zy, localUV);
    vec3 baseColor = texture(usam_blockAtlasColor, atlasUV).rgb;
    float emissiveScale = hardcoded.emissive * exp2(float(hardcoded.emissiveMultiplier));
    vec3 result = baseColor * emissiveScale;
    valid = rcLuminance(result) > 0.0 && !any(isnan(result));
    return result;
}

RCCandidate rcGenerateCandidate(uint entryIndex, ivec3 worldCellCoord, uint level, uint faceId) {
    RCCandidate candidate;
    candidate.radiance = vec3(0.0);
    candidate.dir = rcFaceNormal(faceId);
    candidate.hitPos = rcFaceCenter(worldCellCoord, level, faceId);
    candidate.targetWeight = 0.0;
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

    vec3 rayOrigin = rcFaceCenter(worldCellCoord, level, faceId) + faceNormal * 0.05;
    VoxelRay voxelRay = voxelray_setup(rayOrigin, worldDir, 0u);
    VoxelHit hit = voxel_traceRay(voxelRay, 128);
    if (!hit.hit) {
        return candidate;
    }
    rcTouchHit(hit);

    bool radianceValid = false;
    vec3 radiance = rcSampleVoxelRadiance(hit, radianceValid);
    bool bounceValid = false;
    vec3 bounceRadiance = rcSamplePreviousReservoirBounce(hit, -worldDir, bounceValid);
    radiance += bounceRadiance;
    float targetWeight = rcLuminance(radiance) * cosTheta * safeRcp(localSample.w);
    bool candidateValid = (radianceValid || bounceValid)
        && targetWeight > 0.0
        && !any(isnan(radiance))
        && !isnan(targetWeight);
    if (!candidateValid) {
        return candidate;
    }

    candidate.radiance = radiance;
    candidate.dir = worldDir;
    candidate.hitPos = hit.hitPos;
    candidate.targetWeight = targetWeight;
    candidate.valid = true;
    return candidate;
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
    if (historyValid) {
        uint prevReservoirIndex = rcFaceReservoirIndex(prevEntry.x, prevEntry.y, faceId);
        if (prevReservoirIndex < uint(SETTING_RC_POOL_SIZE)) {
            reservoir = rcReservoirLoad(rcPreviousSide(), prevReservoirIndex);
            historyValid = rcReservoirValid(reservoir);
            historyAge = rcReservoirMetaAge(reservoir.meta);
        } else {
            historyValid = false;
        }
    }

    if (historyValid) {
        float randValue = hash_uintToFloat(hash_41_q3(uvec4(entryIndex, faceId, frameCounter, 0x85EBCA6Bu)));
        rcReservoirUpdate(reservoir, candidate, randValue);
        uint M = rcReservoirMetaM(reservoir.meta);
        reservoir.meta = rcPackReservoirMeta(M, min(historyAge + 1u, 255u), rcReservoirValid(reservoir), 0u);
    } else {
        rcReservoirInitFromCandidate(reservoir, candidate);
    }

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
