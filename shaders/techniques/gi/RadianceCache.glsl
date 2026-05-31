#ifndef INCLUDE_techniques_gi_RadianceCache_glsl
#define INCLUDE_techniques_gi_RadianceCache_glsl a

#include "/util/Hash.glsl"
#include "/util/Math.glsl"

#define RC_CLIP_LEVELS 5u
#define RC_CLIP_SIZE 64u
#define RC_CLIP_MASK 63u
#define RC_INVALID 0xffffffffu
#define RC_ENTRY_COUNT 1310720u
#define RC_RESERVOIR_RECORDS 3u

#define RC_FACE_POS_X 0u
#define RC_FACE_NEG_X 1u
#define RC_FACE_POS_Y 2u
#define RC_FACE_NEG_Y 3u
#define RC_FACE_POS_Z 4u
#define RC_FACE_NEG_Z 5u

#define RC_ENTRY_META_VALID 0x80000000u
#define RC_ENTRY_META_LEVEL_MASK 0x00000007u
#define RC_ENTRY_META_AGE_SHIFT 8u
#define RC_ENTRY_META_AGE_MASK 0x0000ff00u

#define RC_RES_META_VALID 0x80000000u
#define RC_RES_META_M_MASK 0x000000ffu
#define RC_RES_META_AGE_SHIFT 8u
#define RC_RES_META_AGE_MASK 0x0000ff00u

#ifndef RC_DATA_MODIFIER
#define RC_DATA_MODIFIER restrict buffer
#endif

layout(std430, binding = 12) RC_DATA_MODIFIER RadianceCacheMetaData {
    uint rc_allocationCounter;
    uint rc_keyMismatchCounter;
    uint rc_poolOverflowCounter;
    uint rc_cacheHitCounter;
    uint rc_cacheMissCounter;
    uint rc_paddingCounters[11];
};

layout(std430, binding = 13) RC_DATA_MODIFIER RadianceCacheIndirectionData {
    uvec4 rc_indirection[];
};

layout(std430, binding = 14) RC_DATA_MODIFIER RadianceCacheReservoirData {
    uvec4 rc_reservoirs[];
};

struct RCReservoir {
    vec3 radiance;
    float W;
    vec3 sampleDir;
    float targetWeight;
    vec3 hitPos;
    uint meta;
};

struct RCCandidate {
    vec3 radiance;
    vec3 dir;
    vec3 hitPos;
    float targetWeight;
    bool valid;
};

struct RCLookupResult {
    vec3 radiance;
    float weight;
    uint hits;
    uint misses;
    uint levelMask;
    uint faceMask;
    uint m;
    uint age;
};

uint rcCurrentSide() {
    return frameCounter & 1u;
}

uint rcPreviousSide() {
    return 1u - rcCurrentSide();
}

uint rcVoxelSize(uint level) {
    return 1u << level;
}

ivec3 rcWorldCellCoord(vec3 worldPos, uint level) {
    return ivec3(floor(worldPos / float(rcVoxelSize(level))));
}

uvec3 rcClipTexel(ivec3 worldCellCoord) {
    return uvec3(worldCellCoord) & uvec3(RC_CLIP_MASK);
}

uint rcEntryIndex(uint level, ivec3 worldCellCoord) {
    uvec3 clipTexel = rcClipTexel(worldCellCoord);
    return level * (RC_CLIP_SIZE * RC_CLIP_SIZE * RC_CLIP_SIZE)
        + clipTexel.z * (RC_CLIP_SIZE * RC_CLIP_SIZE)
        + clipTexel.y * RC_CLIP_SIZE
        + clipTexel.x;
}

uint rcEntryLevel(uint entryIndex) {
    return entryIndex / (RC_CLIP_SIZE * RC_CLIP_SIZE * RC_CLIP_SIZE);
}

uvec3 rcEntryClipTexel(uint entryIndex) {
    uint localIndex = entryIndex % (RC_CLIP_SIZE * RC_CLIP_SIZE * RC_CLIP_SIZE);
    return uvec3(
        localIndex & RC_CLIP_MASK,
        (localIndex >> 6u) & RC_CLIP_MASK,
        (localIndex >> 12u) & RC_CLIP_MASK
    );
}

int rcNearestClipCoord(int cameraCellCoord, uint clipTexelCoord) {
    int result = (cameraCellCoord & ~int(RC_CLIP_MASK)) | int(clipTexelCoord);
    int delta = result - cameraCellCoord;
    if (delta > int(RC_CLIP_SIZE / 2u)) result -= int(RC_CLIP_SIZE);
    if (delta < -int(RC_CLIP_SIZE / 2u)) result += int(RC_CLIP_SIZE);
    return result;
}

ivec3 rcWorldCellCoordFromEntryIndex(uint entryIndex) {
    uint level = rcEntryLevel(entryIndex);
    uvec3 clipTexel = rcEntryClipTexel(entryIndex);
    ivec3 cameraCell = rcWorldCellCoord(cameraPosition, level);
    return ivec3(
        rcNearestClipCoord(cameraCell.x, clipTexel.x),
        rcNearestClipCoord(cameraCell.y, clipTexel.y),
        rcNearestClipCoord(cameraCell.z, clipTexel.z)
    );
}

uint rcBufferEntryIndex(uint side, uint entryIndex) {
    return side * RC_ENTRY_COUNT + entryIndex;
}

uint rcReservoirRecordIndex(uint side, uint reservoirIndex) {
    return (side * SETTING_RC_POOL_SIZE + reservoirIndex) * RC_RESERVOIR_RECORDS;
}

uint rcWorldKeyHash(uint level, ivec3 worldCellCoord) {
    return hash_41_q3(uvec4(uvec3(worldCellCoord), level));
}

uint rcFaceBit(uint faceId) {
    return 1u << faceId;
}

bool rcHasFace(uint faceMask, uint faceId) {
    return (faceMask & rcFaceBit(faceId)) != 0u;
}

uint rcFaceLocalOffset(uint faceMask, uint faceId) {
    uint faceBit = rcFaceBit(faceId);
    return bitCount(faceMask & (faceBit - 1u));
}

uint rcFaceReservoirIndex(uint baseIndex, uint faceMask, uint faceId) {
    return baseIndex + rcFaceLocalOffset(faceMask, faceId);
}

uint rcAllocClassSize(uint faceCount) {
    if (faceCount <= 1u) return 1u;
    if (faceCount <= 2u) return 2u;
    if (faceCount <= 4u) return 4u;
    return 6u;
}

uint rcPackEntryMeta(uint level, uint age, bool valid) {
    return (valid ? RC_ENTRY_META_VALID : 0u)
        | (level & RC_ENTRY_META_LEVEL_MASK)
        | ((min(age, 255u) << RC_ENTRY_META_AGE_SHIFT) & RC_ENTRY_META_AGE_MASK);
}

bool rcEntryMetaValid(uint meta) {
    return (meta & RC_ENTRY_META_VALID) != 0u;
}

uint rcEntryMetaLevel(uint meta) {
    return meta & RC_ENTRY_META_LEVEL_MASK;
}

uint rcEntryMetaAge(uint meta) {
    return (meta & RC_ENTRY_META_AGE_MASK) >> RC_ENTRY_META_AGE_SHIFT;
}

uint rcPackReservoirMeta(uint M, uint age, bool valid, uint flags) {
    return (valid ? RC_RES_META_VALID : 0u)
        | (min(M, uint(SETTING_RC_M_MAX)) & RC_RES_META_M_MASK)
        | ((min(age, 255u) << RC_RES_META_AGE_SHIFT) & RC_RES_META_AGE_MASK)
        | (flags & 0x7fff0000u);
}

bool rcReservoirMetaValid(uint meta) {
    return (meta & RC_RES_META_VALID) != 0u;
}

uint rcReservoirMetaM(uint meta) {
    return meta & RC_RES_META_M_MASK;
}

uint rcReservoirMetaAge(uint meta) {
    return (meta & RC_RES_META_AGE_MASK) >> RC_RES_META_AGE_SHIFT;
}

vec3 rcFaceNormal(uint faceId) {
    if (faceId == RC_FACE_POS_X) return vec3(1.0, 0.0, 0.0);
    if (faceId == RC_FACE_NEG_X) return vec3(-1.0, 0.0, 0.0);
    if (faceId == RC_FACE_POS_Y) return vec3(0.0, 1.0, 0.0);
    if (faceId == RC_FACE_NEG_Y) return vec3(0.0, -1.0, 0.0);
    if (faceId == RC_FACE_POS_Z) return vec3(0.0, 0.0, 1.0);
    return vec3(0.0, 0.0, -1.0);
}

ivec3 rcFaceNormalI(uint faceId) {
    return ivec3(rcFaceNormal(faceId));
}

uint rcFaceIdFromNormal(vec3 normal) {
    vec3 a = abs(normal);
    if (a.x >= a.y && a.x >= a.z) return normal.x >= 0.0 ? RC_FACE_POS_X : RC_FACE_NEG_X;
    if (a.y >= a.z) return normal.y >= 0.0 ? RC_FACE_POS_Y : RC_FACE_NEG_Y;
    return normal.z >= 0.0 ? RC_FACE_POS_Z : RC_FACE_NEG_Z;
}

vec3 rcFaceCenter(ivec3 worldCellCoord, uint level, uint faceId) {
    float voxelSize = float(rcVoxelSize(level));
    vec3 cellMin = vec3(worldCellCoord) * voxelSize;
    vec3 center = cellMin + vec3(voxelSize * 0.5);
    center += rcFaceNormal(faceId) * (voxelSize * 0.5);
    return center;
}

RCReservoir rcReservoirInit() {
    RCReservoir reservoir;
    reservoir.radiance = vec3(0.0);
    reservoir.W = 0.0;
    reservoir.sampleDir = vec3(0.0, 1.0, 0.0);
    reservoir.targetWeight = 0.0;
    reservoir.hitPos = vec3(0.0);
    reservoir.meta = 0u;
    return reservoir;
}

bool rcReservoirValid(RCReservoir reservoir) {
    return rcReservoirMetaValid(reservoir.meta) && rcReservoirMetaM(reservoir.meta) > 0u;
}

RCReservoir rcReservoirLoad(uint side, uint reservoirIndex) {
    uint recordIndex = rcReservoirRecordIndex(side, reservoirIndex);
    uvec4 r0 = rc_reservoirs[recordIndex + 0u];
    uvec4 r1 = rc_reservoirs[recordIndex + 1u];
    uvec4 r2 = rc_reservoirs[recordIndex + 2u];
    RCReservoir reservoir;
    reservoir.radiance = uintBitsToFloat(r0.xyz);
    reservoir.W = uintBitsToFloat(r0.w);
    reservoir.sampleDir = uintBitsToFloat(r1.xyz);
    reservoir.targetWeight = uintBitsToFloat(r1.w);
    reservoir.hitPos = uintBitsToFloat(r2.xyz);
    reservoir.meta = r2.w;
    return reservoir;
}

void rcReservoirStore(uint side, uint reservoirIndex, RCReservoir reservoir) {
    uint recordIndex = rcReservoirRecordIndex(side, reservoirIndex);
    rc_reservoirs[recordIndex + 0u] = uvec4(floatBitsToUint(reservoir.radiance), floatBitsToUint(reservoir.W));
    rc_reservoirs[recordIndex + 1u] = uvec4(floatBitsToUint(reservoir.sampleDir), floatBitsToUint(reservoir.targetWeight));
    rc_reservoirs[recordIndex + 2u] = uvec4(floatBitsToUint(reservoir.hitPos), reservoir.meta);
}

float rcLuminance(vec3 radiance) {
    return length(radiance);
}

void rcReservoirInitFromCandidate(inout RCReservoir reservoir, RCCandidate candidate) {
    if (candidate.valid && candidate.targetWeight > 0.0) {
        reservoir.radiance = candidate.radiance;
        reservoir.W = candidate.targetWeight;
        reservoir.sampleDir = candidate.dir;
        reservoir.targetWeight = candidate.targetWeight;
        reservoir.hitPos = candidate.hitPos;
        reservoir.meta = rcPackReservoirMeta(1u, 0u, true, 0u);
    } else {
        reservoir = rcReservoirInit();
    }
}

void rcReservoirUpdate(inout RCReservoir reservoir, RCCandidate candidate, float randValue) {
    if (!candidate.valid || candidate.targetWeight <= 0.0) {
        return;
    }

    uint M = min(rcReservoirMetaM(reservoir.meta) + 1u, uint(SETTING_RC_M_MAX));
    reservoir.W += candidate.targetWeight;
    float p = candidate.targetWeight / max(reservoir.W, 1e-6);
    if (randValue < p) {
        reservoir.radiance = candidate.radiance;
        reservoir.sampleDir = candidate.dir;
        reservoir.hitPos = candidate.hitPos;
        reservoir.targetWeight = candidate.targetWeight;
    }
    reservoir.meta = rcPackReservoirMeta(M, 0u, true, 0u);
}

RCLookupResult rcLookupInit() {
    RCLookupResult result;
    result.radiance = vec3(0.0);
    result.weight = 0.0;
    result.hits = 0u;
    result.misses = 0u;
    result.levelMask = 0u;
    result.faceMask = 0u;
    result.m = 0u;
    result.age = 0u;
    return result;
}

uint rcSelectLevel(float queryRadiusBlocks) {
    if (queryRadiusBlocks < 16.0) return 0u;
    if (queryRadiusBlocks < 32.0) return 1u;
    if (queryRadiusBlocks < 64.0) return 2u;
    if (queryRadiusBlocks < 128.0) return 3u;
    return 4u;
}

void rcLookupSampleFace(
    inout RCLookupResult result,
    vec3 P,
    vec3 N,
    uint level,
    ivec3 worldCellCoord,
    uint faceId
) {
    uint entryIndex = rcEntryIndex(level, worldCellCoord);
    uint bufferIndex = rcBufferEntryIndex(rcCurrentSide(), entryIndex);
    uvec4 entry = rc_indirection[bufferIndex];
    uint worldKeyHash = rcWorldKeyHash(level, worldCellCoord);

    if (entry.x == RC_INVALID || entry.z != worldKeyHash || !rcEntryMetaValid(entry.w) || rcEntryMetaLevel(entry.w) != level || !rcHasFace(entry.y, faceId)) {
        result.misses++;
        return;
    }

    uint reservoirIndex = rcFaceReservoirIndex(entry.x, entry.y, faceId);
    if (reservoirIndex >= uint(SETTING_RC_POOL_SIZE)) {
        result.misses++;
        return;
    }

    RCReservoir reservoir = rcReservoirLoad(rcCurrentSide(), reservoirIndex);
    if (!rcReservoirValid(reservoir)) {
        result.misses++;
        return;
    }

    vec3 faceNormal = rcFaceNormal(faceId);
    float normalWeight = max(dot(N, faceNormal), 0.0);
    if (normalWeight <= 0.0) {
        result.misses++;
        return;
    }

    vec3 faceCenter = rcFaceCenter(worldCellCoord, level, faceId);
    float thickness = float(rcVoxelSize(level)) * 0.25;
    float side = dot(P - faceCenter, faceNormal);
    if (side < -thickness) {
        result.misses++;
        return;
    }

    vec3 d = P - faceCenter;
    float dist2 = dot(d, d);
    float radius = float(rcVoxelSize(level)) * 2.0;
    float distanceWeight = exp(-dist2 / max(radius * radius, 1e-4));
    float ageWeight = exp2(-float(rcReservoirMetaAge(reservoir.meta)) * 0.125);
    float w = normalWeight * distanceWeight * ageWeight;

    result.radiance += reservoir.radiance * w;
    result.weight += w;
    result.hits++;
    result.levelMask |= 1u << level;
    result.faceMask |= rcFaceBit(faceId);
    result.m = max(result.m, rcReservoirMetaM(reservoir.meta));
    result.age = max(result.age, rcReservoirMetaAge(reservoir.meta));
}

RCLookupResult rcLookupDiffuseGI(vec3 P, vec3 N, float queryRadiusBlocks) {
    RCLookupResult result = rcLookupInit();
    uint level = rcSelectLevel(queryRadiusBlocks);
    ivec3 baseCell = rcWorldCellCoord(P, level);

    const int searchRadius = 0;
    for (int z = -searchRadius; z <= searchRadius; z++) {
        for (int y = -searchRadius; y <= searchRadius; y++) {
            for (int x = -searchRadius; x <= searchRadius; x++) {
                ivec3 cell = baseCell + ivec3(x, y, z);
                for (uint faceId = 0u; faceId < 6u; faceId++) {
                    rcLookupSampleFace(result, P, N, level, cell, faceId);
                }
            }
        }
    }

    if (result.weight > 0.0) {
        result.radiance /= result.weight;
    }
    return result;
}

#endif
