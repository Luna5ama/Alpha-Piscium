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
    float avgWY;
    vec3 sampleDir;
    float m;
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
    float m;
    uint age;
};

uint rcCurrentSide() {
    return uint(frameCounter) & 1u;
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

uint rcPackReservoirMeta(uint age, bool valid, uint flags) {
    return (valid ? RC_RES_META_VALID : 0u)
        | ((min(age, 255u) << RC_RES_META_AGE_SHIFT) & RC_RES_META_AGE_MASK)
        | (flags & 0x7fff0000u);
}

bool rcReservoirMetaValid(uint meta) {
    return (meta & RC_RES_META_VALID) != 0u;
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
    reservoir.avgWY = 0.0;
    reservoir.sampleDir = vec3(0.0, 1.0, 0.0);
    reservoir.m = 0.0;
    reservoir.hitPos = vec3(0.0);
    reservoir.meta = 0u;
    return reservoir;
}

bool rcReservoirValid(RCReservoir reservoir) {
    return rcReservoirMetaValid(reservoir.meta) && reservoir.m > 0.0;
}

RCReservoir rcReservoirLoad(uint side, uint reservoirIndex) {
    uint recordIndex = rcReservoirRecordIndex(side, reservoirIndex);
    uvec4 r0 = rc_reservoirs[recordIndex + 0u];
    uvec4 r1 = rc_reservoirs[recordIndex + 1u];
    uvec4 r2 = rc_reservoirs[recordIndex + 2u];
    RCReservoir reservoir;
    reservoir.radiance = uintBitsToFloat(r0.xyz);
    reservoir.avgWY = uintBitsToFloat(r0.w);
    reservoir.sampleDir = uintBitsToFloat(r1.xyz);
    reservoir.m = uintBitsToFloat(r1.w);
    reservoir.hitPos = uintBitsToFloat(r2.xyz);
    reservoir.meta = r2.w;
    return reservoir;
}

void rcReservoirStore(uint side, uint reservoirIndex, RCReservoir reservoir) {
    uint recordIndex = rcReservoirRecordIndex(side, reservoirIndex);
    rc_reservoirs[recordIndex + 0u] = uvec4(floatBitsToUint(reservoir.radiance), floatBitsToUint(reservoir.avgWY));
    rc_reservoirs[recordIndex + 1u] = uvec4(floatBitsToUint(reservoir.sampleDir), floatBitsToUint(reservoir.m));
    rc_reservoirs[recordIndex + 2u] = uvec4(floatBitsToUint(reservoir.hitPos), reservoir.meta);
}

float rcLuminance(vec3 radiance) {
    return length(radiance);
}

float rcReservoirTargetWeight(RCReservoir reservoir) {
    return rcLuminance(reservoir.radiance) * PI;
}

void rcReservoirInitFromCandidate(inout RCReservoir reservoir, RCCandidate candidate) {
    if (candidate.valid && candidate.targetWeight > 0.0) {
        reservoir.radiance = candidate.radiance;
        reservoir.avgWY = 1.0;
        reservoir.sampleDir = candidate.dir;
        reservoir.m = 1.0;
        reservoir.hitPos = candidate.hitPos;
        reservoir.meta = rcPackReservoirMeta(0u, true, 0u);
    } else {
        reservoir = rcReservoirInit();
    }
}

bool rcReservoirUpdate(inout RCReservoir reservoir, inout float wSum, RCCandidate candidate, float randValue) {
    if (!candidate.valid || candidate.targetWeight <= 0.0) {
        return false;
    }

    wSum += candidate.targetWeight;
    reservoir.m = clamp(reservoir.m + 1.0, 1.0, float(SETTING_RC_M_MAX));
    float p = candidate.targetWeight / max(wSum, 1e-6);
    if (randValue < p) {
        reservoir.radiance = candidate.radiance;
        reservoir.sampleDir = candidate.dir;
        reservoir.hitPos = candidate.hitPos;
        return true;
    }

    return false;
}

RCLookupResult rcLookupInit() {
    RCLookupResult result;
    result.radiance = vec3(0.0);
    result.weight = 0.0;
    result.hits = 0u;
    result.misses = 0u;
    result.levelMask = 0u;
    result.faceMask = 0u;
    result.m = 0.0;
    result.age = 0u;
    return result;
}

uint rcSelectLevel(vec3 P) {
    vec3 d = abs(P - cameraPositionInt);
    float maxDistF = max(max(d.x, d.y), d.z);

    // Conservative integer distance in blocks.
    uint maxDist = uint(ceil(maxDistF));

    uint safeRadius = RC_CLIP_SIZE / 2u - 1u; // 31

    if (maxDist <= safeRadius) {
        return 0u;
    }

    // Need smallest level such that:
    // safeRadius * 2^level >= maxDist
    uint q = (maxDist + safeRadius - 1u) / safeRadius; // ceil(maxDist / safeRadius)

    // ceil(log2(q))
    uint level = uint(findMSB(q - 1u) + 1);

    return min(level, RC_CLIP_LEVELS - 1u);
}

uint rcDominantFaceId(vec3 N) {
    vec3 a = abs(N);

    if (a.x >= a.y && a.x >= a.z) {
        return N.x >= 0.0 ? RC_FACE_POS_X : RC_FACE_NEG_X;
    }

    if (a.y >= a.x && a.y >= a.z) {
        return N.y >= 0.0 ? RC_FACE_POS_Y : RC_FACE_NEG_Y;
    }

    return N.z >= 0.0 ? RC_FACE_POS_Z : RC_FACE_NEG_Z;
}

uint rcFaceAxis(uint faceId) {
    if (faceId == RC_FACE_POS_X || faceId == RC_FACE_NEG_X) return 0u;
    if (faceId == RC_FACE_POS_Y || faceId == RC_FACE_NEG_Y) return 1u;
    return 2u;
}

void rcFaceTangentAxes(uint faceId, out uint axis0, out uint axis1) {
    uint nAxis = rcFaceAxis(faceId);

    if (nAxis == 0u) {
        axis0 = 1u; // Y
        axis1 = 2u; // Z
    } else if (nAxis == 1u) {
        axis0 = 0u; // X
        axis1 = 2u; // Z
    } else {
        axis0 = 0u; // X
        axis1 = 1u; // Y
    }
}

void rcLookupSampleFaceWeighted(
    inout RCLookupResult result,
    vec3 P,
    vec3 N,
    uint level,
    ivec3 worldCellCoord,
    uint faceId,
    float interpWeight
) {
    if (interpWeight <= 0.0) {
        return;
    }

    uint entryIndex = rcEntryIndex(level, worldCellCoord);
    uint bufferIndex = rcBufferEntryIndex(rcCurrentSide(), entryIndex);
    uvec4 entry = rc_indirection[bufferIndex];

    uint worldKeyHash = rcWorldKeyHash(level, worldCellCoord);

    if (
    entry.x == RC_INVALID ||
    entry.z != worldKeyHash ||
    !rcEntryMetaValid(entry.w) ||
    rcEntryMetaLevel(entry.w) != level ||
    !rcHasFace(entry.y, faceId)
    ) {
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

    float voxelSize = float(rcVoxelSize(level));
    float thickness = voxelSize * 0.25;

    float side = dot(P - faceCenter, faceNormal);
    if (side < -thickness) {
        result.misses++;
        return;
    }

    uint age = rcReservoirMetaAge(reservoir.meta);
    float ageWeight = exp2(-float(age) * 0.125);

    // Bilinear interpolation already provides the geometric tangent-plane weight.
    // Do not multiply by the old exp(-dist²/r²), otherwise it stops being proper bilinear filtering.
    float w = interpWeight * normalWeight * ageWeight;

    if (w <= 0.0) {
        result.misses++;
        return;
    }

    result.radiance += reservoir.radiance * w;
    result.weight += w;

    result.hits++;
    result.levelMask |= 1u << level;
    result.faceMask |= rcFaceBit(faceId);
    result.m = max(result.m, reservoir.m);
    result.age = max(result.age, age);
}

RCLookupResult rcLookupDiffuseGIBilinear(vec3 P, vec3 N) {
    RCLookupResult result = rcLookupInit();

    uint level = rcSelectLevel(P);
    float voxelSize = float(rcVoxelSize(level));

    uint faceId = rcDominantFaceId(N);
    vec3 faceNormal = rcFaceNormal(faceId);

    uint axis0;
    uint axis1;
    rcFaceTangentAxes(faceId, axis0, axis1);

    uint normalAxis = rcFaceAxis(faceId);

    // Move slightly behind the queried surface so the owner cell is stable.
    // For +Y face, this moves into the solid cell below the face.
    // For -Y face, this moves into the solid cell above the face.
    float surfaceEpsilon = max(voxelSize * 1e-3, 1e-3);
    vec3 ownerP = P - faceNormal * surfaceEpsilon;

    ivec3 ownerCell = rcWorldCellCoord(ownerP, level);

    // Face centers lie at cell + 0.5 along tangent axes.
    // Therefore the bilinear coordinate over face centers is:
    //
    //     P_t / voxelSize - 0.5
    //
    // The integer part selects the lower face-center cell,
    // and the fractional part is the interpolation weight.
    vec3 cellSpace = P / voxelSize - vec3(0.5);

    float u = cellSpace[int(axis0)];
    float v = cellSpace[int(axis1)];

    int u0 = int(floor(u));
    int v0 = int(floor(v));

    float fu = fract(u);
    float fv = fract(v);

    float w00 = (1.0 - fu) * (1.0 - fv);
    float w10 = fu * (1.0 - fv);
    float w01 = (1.0 - fu) * fv;
    float w11 = fu * fv;

    ivec3 cell00 = ownerCell;
    ivec3 cell10 = ownerCell;
    ivec3 cell01 = ownerCell;
    ivec3 cell11 = ownerCell;

    cell00[int(axis0)] = u0;
    cell00[int(axis1)] = v0;

    cell10[int(axis0)] = u0 + 1;
    cell10[int(axis1)] = v0;

    cell01[int(axis0)] = u0;
    cell01[int(axis1)] = v0 + 1;

    cell11[int(axis0)] = u0 + 1;
    cell11[int(axis1)] = v0 + 1;

    // The normal-axis coordinate remains fixed from ownerCell.
    // This is what turns the lookup from 2x2x2 into 2x2x1.
    cell00[int(normalAxis)] = ownerCell[int(normalAxis)];
    cell10[int(normalAxis)] = ownerCell[int(normalAxis)];
    cell01[int(normalAxis)] = ownerCell[int(normalAxis)];
    cell11[int(normalAxis)] = ownerCell[int(normalAxis)];

    rcLookupSampleFaceWeighted(result, P, N, level, cell00, faceId, w00);
    rcLookupSampleFaceWeighted(result, P, N, level, cell10, faceId, w10);
    rcLookupSampleFaceWeighted(result, P, N, level, cell01, faceId, w01);
    rcLookupSampleFaceWeighted(result, P, N, level, cell11, faceId, w11);

    if (result.weight > 0.0) {
        result.radiance /= result.weight;
    }

    return result;
}

void rcLookupSampleFace1x1(
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

    if (
    entry.x == RC_INVALID ||
    entry.z != worldKeyHash ||
    !rcEntryMetaValid(entry.w) ||
    rcEntryMetaLevel(entry.w) != level ||
    !rcHasFace(entry.y, faceId)
    ) {
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

    float voxelSize = float(rcVoxelSize(level));
    float thickness = voxelSize * 0.25;

    float side = dot(P - faceCenter, faceNormal);
    if (side < -thickness) {
        result.misses++;
        return;
    }

    uint age = rcReservoirMetaAge(reservoir.meta);
    float ageWeight = exp2(-float(age) * 0.125);

    // 1x1 lookup: no bilinear and no tangent distance filter.
    // Weight only by normal compatibility and history freshness.
    float w = normalWeight * ageWeight;

    if (w <= 0.0) {
        result.misses++;
        return;
    }

    result.radiance += reservoir.radiance * w;
    result.weight += w;

    result.hits++;
    result.levelMask |= 1u << level;
    result.faceMask |= rcFaceBit(faceId);
    result.m = max(result.m, reservoir.m);
    result.age = max(result.age, age);
}

RCLookupResult rcLookupDiffuseGI(vec3 P, vec3 N) {
    RCLookupResult result = rcLookupInit();

    uint level = rcSelectLevel(P);
    float voxelSize = float(rcVoxelSize(level));

    uint faceId = rcDominantFaceId(N);
    vec3 faceNormal = rcFaceNormal(faceId);

    // Move into the owner voxel so the face owner is stable.
    // For +Y face, this moves slightly below the surface.
    // For -Y face, this moves slightly above the surface.
    float surfaceEpsilon = max(voxelSize * 1e-3, 1e-3);
    vec3 ownerP = P - faceNormal * surfaceEpsilon;

    ivec3 ownerCell = rcWorldCellCoord(ownerP, level);

    rcLookupSampleFace1x1(
        result,
        P,
        N,
        level,
        ownerCell,
        faceId
    );

    if (result.weight > 0.0) {
        result.radiance /= result.weight;
    }

    return result;
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

#endif
