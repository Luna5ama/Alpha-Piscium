#define RC_DATA_MODIFIER restrict buffer

layout(local_size_x = 256) in;

#include "/techniques/gi/RadianceCache.glsl"
#include "/techniques/voxel/VoxelTrace.glsl"
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"
#include "/util/Colors2.glsl"
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

    bool radianceValid = false;
    vec3 radiance = rcSampleVoxelRadiance(hit, radianceValid);
    float targetWeight = rcLuminance(radiance) * cosTheta * safeRcp(localSample.w);
    if (!radianceValid || targetWeight <= 0.0 || any(isnan(radiance)) || isnan(targetWeight)) {
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
