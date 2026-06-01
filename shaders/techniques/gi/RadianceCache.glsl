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
#define RC_ENTRY_META_PENDING_FACE_SHIFT 16u
#define RC_ENTRY_META_PENDING_FACE_MASK 0x003f0000u

#define RC_RES_META_VALID 0x80000000u
#define RC_RES_META_AGE_SHIFT 8u
#define RC_RES_META_AGE_MASK 0x0000ff00u
#define RC_RES_META_FLAGS_MASK 0x7fff0000u

#define RC_RES_FLAG_SURFACE_HIT 0x00010000u
#define RC_RES_FLAG_SKY_MISS 0x00020000u
#define RC_RES_FLAG_PARENT_BOOTSTRAP 0x00040000u

#define RC_FEEDBACK_SCREEN_SHIFT 0u
#define RC_FEEDBACK_HIT_SHIFT 6u
#define RC_FEEDBACK_FACE_MASK 0x3fu

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

// x: reservoir base index, or RC_INVALID if no reservoir allocated for this entry
// y: bitmask of valid faces
// z: world key hash
// w: meta (bits 0-2: level, bit 31: valid flag, bits 16-23: pending visible face mask)
layout(std430, binding = 13) RC_DATA_MODIFIER RadianceCacheIndirectionData {
    uvec4 rc_indirection[];
};

layout(std430, binding = 14) RC_DATA_MODIFIER RadianceCacheReservoirData {
    uvec4 rc_reservoirs[];
};

// x: world key hash, or RC_INVALID
// y bits 0-5: screen touched faces
// y bits 6-11: next-frame hit feedback faces
layout(std430, binding = 15) RC_DATA_MODIFIER RadianceCacheFeedbackData {
    uvec2 rc_feedback[];
};

struct RCReservoir {
    vec3 radiance;
    float avgWY;
    vec3 sampleDir;
    float m;
    vec3 hitPos;
    uint meta;
};

RCReservoir rc_reservoirInit() {
    RCReservoir reservoir;
    reservoir.radiance = vec3(0.0);
    reservoir.avgWY = 0.0;
    reservoir.sampleDir = vec3(0.0, 1.0, 0.0);
    reservoir.m = 0.0;
    reservoir.hitPos = vec3(0.0);
    reservoir.meta = 0u;
    return reservoir;
}

bool _rcReservoirMetaValid(uint meta) {
    return (meta & RC_RES_META_VALID) != 0u;
}


bool rc_reservoirValid(RCReservoir reservoir) {
    return _rcReservoirMetaValid(reservoir.meta) && reservoir.m > 0.0;
}

uint rc_reservoirRecordIndex(uint side, uint reservoirIndex) {
    return (side * SETTING_RC_POOL_SIZE + reservoirIndex) * RC_RESERVOIR_RECORDS;
}

RCReservoir rc_reservoirLoad(uint side, uint reservoirIndex) {
    uint recordIndex = rc_reservoirRecordIndex(side, reservoirIndex);
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

struct RCCandidate {
    vec3 radiance;
    vec3 dir;
    vec3 hitPos;
    vec3 hitNormal;
    float targetWeight;
    uint flags;
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
    uint debug;
};

RCLookupResult rc_lookupInit() {
    RCLookupResult result;
    result.radiance = vec3(0.0);
    result.weight = 0.0;
    result.hits = 0u;
    result.misses = 0u;
    result.levelMask = 0u;
    result.faceMask = 0u;
    result.m = 0.0;
    result.age = 0u;
    result.debug = 0;
    return result;
}

uint rc_currentSide() {
    return uint(frameCounter) & 1u;
}

uint rc_previousSide() {
    return 1u - rc_currentSide();
}

uint rc_voxelSize(uint level) {
    return 1u << level;
}

ivec3 rc_worldCellCoord(vec3 worldPos, uint level) {
    return ivec3(floor(ldexp(worldPos, ivec3(-int(level)))));
}

uvec3 rc_clipTexel(ivec3 worldCellCoord) {
    return uvec3(worldCellCoord) & uvec3(RC_CLIP_MASK);
}

uint rc_entryIndex(uint level, ivec3 worldCellCoord) {
    uvec3 clipTexel = rc_clipTexel(worldCellCoord);
    return level * (RC_CLIP_SIZE * RC_CLIP_SIZE * RC_CLIP_SIZE)
        + clipTexel.z * (RC_CLIP_SIZE * RC_CLIP_SIZE)
        + clipTexel.y * RC_CLIP_SIZE
        + clipTexel.x;
}

uint rc_entryLevel(uint entryIndex) {
    return entryIndex / (RC_CLIP_SIZE * RC_CLIP_SIZE * RC_CLIP_SIZE);
}

uvec3 rc_entryClipTexel(uint entryIndex) {
    uint localIndex = entryIndex % (RC_CLIP_SIZE * RC_CLIP_SIZE * RC_CLIP_SIZE);
    return uvec3(
        localIndex & RC_CLIP_MASK,
        (localIndex >> 6u) & RC_CLIP_MASK,
        (localIndex >> 12u) & RC_CLIP_MASK
    );
}

int rc_nearestClipCoord(int cameraCellCoord, uint clipTexelCoord) {
    int result = (cameraCellCoord & ~int(RC_CLIP_MASK)) | int(clipTexelCoord);
    int delta = result - cameraCellCoord;
    if (delta > int(RC_CLIP_SIZE / 2u)) result -= int(RC_CLIP_SIZE);
    if (delta < -int(RC_CLIP_SIZE / 2u)) result += int(RC_CLIP_SIZE);
    return result;
}

ivec3 rc_worldCellCoordFromEntryIndex(uint entryIndex) {
    uint level = rc_entryLevel(entryIndex);
    uvec3 clipTexel = rc_entryClipTexel(entryIndex);
    ivec3 cameraCell = rc_worldCellCoord(cameraPosition, level);
    return ivec3(
        rc_nearestClipCoord(cameraCell.x, clipTexel.x),
        rc_nearestClipCoord(cameraCell.y, clipTexel.y),
        rc_nearestClipCoord(cameraCell.z, clipTexel.z)
    );
}

bool rc_worldCellInCurrentClip(uint level, ivec3 worldCellCoord) {
    ivec3 cameraCell = rc_worldCellCoord(cameraPosition, level);
    ivec3 delta = worldCellCoord - cameraCell;
    return all(lessThanEqual(abs(delta), ivec3(31)));
}

uint rc_bufferEntryIndex(uint side, uint entryIndex) {
    return side * RC_ENTRY_COUNT + entryIndex;
}

uint rc_feedbackRecordIndex(uint side, uint entryIndex) {
    return rc_bufferEntryIndex(side, entryIndex);
}

uint rc_worldKeyHash(uint level, ivec3 worldCellCoord) {
    return hash_41_q5(uvec4(uvec3(worldCellCoord), level));
}

uint rc_faceBit(uint faceId) {
    return 1u << faceId;
}

bool rc_hasFace(uint faceMask, uint faceId) {
    return (faceMask & rc_faceBit(faceId)) != 0u;
}

uint rc_faceLocalOffset(uint faceMask, uint faceId) {
    uint faceBit = rc_faceBit(faceId);
    return bitCount(faceMask & (faceBit - 1u));
}

uint rc_faceReservoirIndex(uint baseIndex, uint faceMask, uint faceId) {
    return baseIndex + rc_faceLocalOffset(faceMask, faceId);
}

uint rc_allocClassSize(uint faceCount) {
    if (faceCount <= 1u) return 1u;
    if (faceCount <= 2u) return 2u;
    if (faceCount <= 4u) return 4u;
    return 6u;
}

uint rc_packEntryMeta(uint level, bool valid) {
    return (valid ? RC_ENTRY_META_VALID : 0u)
        | (level & RC_ENTRY_META_LEVEL_MASK);
}

bool rc_entryMetaValid(uint meta) {
    return (meta & RC_ENTRY_META_VALID) != 0u;
}

uint rc_entryMetaLevel(uint meta) {
    return meta & RC_ENTRY_META_LEVEL_MASK;
}

uint rc_entryMetaPendingFaceMask(uint meta) {
    return (meta & RC_ENTRY_META_PENDING_FACE_MASK) >> RC_ENTRY_META_PENDING_FACE_SHIFT;
}

uint rc_entryMetaPendingFaceBits(uint faceMask) {
    return (faceMask & 0x3fu) << RC_ENTRY_META_PENDING_FACE_SHIFT;
}

uint rc_entryMetaClearPendingFaces(uint meta) {
    return meta & ~RC_ENTRY_META_PENDING_FACE_MASK;
}

uint rc_feedbackScreenBits(uint faceMask) {
    return (faceMask & RC_FEEDBACK_FACE_MASK) << RC_FEEDBACK_SCREEN_SHIFT;
}

uint rc_feedbackHitBits(uint faceMask) {
    return (faceMask & RC_FEEDBACK_FACE_MASK) << RC_FEEDBACK_HIT_SHIFT;
}

uint rc_feedbackScreenFaceMask(uint side, uint entryIndex) {
    uint recordIndex = rc_feedbackRecordIndex(side, entryIndex);
    return (rc_feedback[recordIndex].y >> RC_FEEDBACK_SCREEN_SHIFT) & RC_FEEDBACK_FACE_MASK;
}

uint rc_feedbackHitFaceMask(uint side, uint entryIndex) {
    uint recordIndex = rc_feedbackRecordIndex(side, entryIndex);
    return (rc_feedback[recordIndex].y >> RC_FEEDBACK_HIT_SHIFT) & RC_FEEDBACK_FACE_MASK;
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

uint rc_packReservoirMeta(uint age, bool valid, uint flags) {
    return (valid ? RC_RES_META_VALID : 0u)
        | ((min(age, 255u) << RC_RES_META_AGE_SHIFT) & RC_RES_META_AGE_MASK)
        | (flags & RC_RES_META_FLAGS_MASK);
}

uint rc_reservoirMetaAge(uint meta) {
    return (meta & RC_RES_META_AGE_MASK) >> RC_RES_META_AGE_SHIFT;
}

uint rc_reservoirMetaFlags(uint meta) {
    return meta & RC_RES_META_FLAGS_MASK;
}

bool rc_reservoirIsSurfaceHit(RCReservoir reservoir) {
    return (rc_reservoirMetaFlags(reservoir.meta) & RC_RES_FLAG_SURFACE_HIT) != 0u;
}

bool rc_reservoirIsSkyMiss(RCReservoir reservoir) {
    return (rc_reservoirMetaFlags(reservoir.meta) & RC_RES_FLAG_SKY_MISS) != 0u;
}

bool rc_reservoirIsParentBootstrap(RCReservoir reservoir) {
    return (rc_reservoirMetaFlags(reservoir.meta) & RC_RES_FLAG_PARENT_BOOTSTRAP) != 0u;
}

vec3 rc_faceNormal(uint faceId) {
    uint axis = faceId >> 1u;
    float signValue = 1.0 - 2.0 * float(faceId & 1u);

    vec3 axisMask = vec3(
        float(axis == 0u),
        float(axis == 1u),
        float(axis == 2u)
    );

    return axisMask * signValue;
}

ivec3 rc_faceNormalI(uint faceId) {
    uint axis = faceId >> 1u;
    int signValue = 1 - 2 * int(faceId & 1u);

    ivec3 axisMask = ivec3(
        int(axis == 0u),
        int(axis == 1u),
        int(axis == 2u)
    );

    return axisMask * signValue;
}

uint rc_faceIdFromNormal(vec3 normal) {
    vec3 a = abs(normal);

    uint yWins = uint(a.y > a.x) & uint(a.y >= a.z);
    uint zWins = uint(a.z > a.x) & uint(a.z >  a.y);

    uint axis = yWins + zWins * 2u;

    float axisValue =
    normal.x * float(axis == 0u) +
    normal.y * float(axis == 1u) +
    normal.z * float(axis == 2u);

    uint signBit = uint(axisValue < 0.0);

    return axis * 2u + signBit;
}

vec3 rc_faceCenter(ivec3 worldCellCoord, uint level, uint faceId) {
    float voxelSize = float(rc_voxelSize(level));
    vec3 cellMin = ldexp(worldCellCoord, ivec3(level));
    vec3 center = cellMin + vec3(voxelSize * 0.5);
    center += rc_faceNormal(faceId) * (voxelSize * 0.5);
    return center;
}

vec3 rc_faceRayOrigin(ivec3 worldCellCoord, uint level, uint faceId) {
    float halfVoxel = float(rc_voxelSize(level)) * 0.5;
    return ldexp(worldCellCoord, ivec3(level))
        + vec3(halfVoxel)
        + rc_faceNormal(faceId) * (halfVoxel + 0.05);
}

ivec3 rc_neighborPlaneOffset(uint faceId, int offset0, int offset1) {
    if (faceId == RC_FACE_POS_X || faceId == RC_FACE_NEG_X) {
        return ivec3(0, offset0, offset1);
    }

    if (faceId == RC_FACE_POS_Y || faceId == RC_FACE_NEG_Y) {
        return ivec3(offset0, 0, offset1);
    }

    return ivec3(offset0, offset1, 0);
}

ivec2 rc_neighborOffset8(uint index) {
    if (index == 0u) return ivec2(-1, -1);
    if (index == 1u) return ivec2(0, -1);
    if (index == 2u) return ivec2(1, -1);
    if (index == 3u) return ivec2(-1, 0);
    if (index == 4u) return ivec2(1, 0);
    if (index == 5u) return ivec2(-1, 1);
    if (index == 6u) return ivec2(0, 1);
    return ivec2(1, 1);
}

float rc_areaPdfCosineConnection(
    vec3 origin,
    vec3 originNormal,
    vec3 hitPos,
    vec3 hitNormal
) {
    vec3 delta = hitPos - origin;
    float distanceSq = dot(delta, delta);
    if (distanceSq <= 1e-6) {
        return 0.0;
    }

    vec3 wi = delta * inversesqrt(distanceSq);
    float cosOrigin = dot(originNormal, wi);
    float cosHit = abs(dot(hitNormal, -wi));
    if (cosOrigin <= 0.0 || cosHit <= 0.0) {
        return 0.0;
    }

    float pdfOmega = cosOrigin * RCP_PI;
    return pdfOmega * cosHit / distanceSq;
}

bool rc_loadFaceReservoir(
    uint side,
    uint level,
    ivec3 worldCellCoord,
    uint faceId,
    out RCReservoir reservoir
) {
    reservoir = rc_reservoirInit();

    uint entryIndex = rc_entryIndex(level, worldCellCoord);
    uint bufferIndex = rc_bufferEntryIndex(side, entryIndex);
    uvec4 entry = rc_indirection[bufferIndex];
    uint worldKeyHash = rc_worldKeyHash(level, worldCellCoord);
    if (
        entry.x == RC_INVALID
        || entry.z != worldKeyHash
        || !rc_entryMetaValid(entry.w)
        || rc_entryMetaLevel(entry.w) != level
        || !rc_hasFace(entry.y, faceId)
    ) {
        return false;
    }

    uint reservoirIndex = rc_faceReservoirIndex(entry.x, entry.y, faceId);
    if (reservoirIndex >= uint(SETTING_RC_POOL_SIZE)) {
        return false;
    }

    reservoir = rc_reservoirLoad(side, reservoirIndex);
    return rc_reservoirValid(reservoir);
}

bool rc_loadParentFaceReservoir(
    uint level,
    ivec3 worldCellCoord,
    uint faceId,
    out uint parentLevel,
    out ivec3 parentCell,
    out RCReservoir parentReservoir
) {
    parentLevel = level;
    parentCell = worldCellCoord;
    parentReservoir = rc_reservoirInit();

    if (level + 1u >= RC_CLIP_LEVELS) {
        return false;
    }

    parentLevel = level + 1u;

    vec3 faceNormal = rc_faceNormal(faceId);
    vec3 faceCenter = rc_faceCenter(worldCellCoord, level, faceId);
    float voxelSize = float(rc_voxelSize(level));
    float epsilon = max(voxelSize * 1e-3, 1e-3);
    vec3 ownerP = faceCenter - faceNormal * epsilon;
    parentCell = rc_worldCellCoord(ownerP, parentLevel);

    return rc_loadFaceReservoir(
        rc_previousSide(),
        parentLevel,
        parentCell,
        faceId,
        parentReservoir
    );
}
void rc_reservoirStore(uint side, uint reservoirIndex, RCReservoir reservoir) {
    uint recordIndex = rc_reservoirRecordIndex(side, reservoirIndex);
    rc_reservoirs[recordIndex + 0u] = uvec4(floatBitsToUint(reservoir.radiance), floatBitsToUint(reservoir.avgWY));
    rc_reservoirs[recordIndex + 1u] = uvec4(floatBitsToUint(reservoir.sampleDir), floatBitsToUint(reservoir.m));
    rc_reservoirs[recordIndex + 2u] = uvec4(floatBitsToUint(reservoir.hitPos), reservoir.meta);
}

float rc_luminance(vec3 radiance) {
    return length(radiance);
}

vec3 rc_reservoirEstimateRadiance(RCReservoir reservoir) {
    vec3 result = vec3(0.0);

    if (rc_reservoirValid(reservoir)) {
        result = max(reservoir.radiance * reservoir.avgWY, 0.0);
    }

    return result;
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
    } else {
        reservoir = rc_reservoirInit();
    }
    return reservoir;
}

bool rc_reservoirUpdate(inout RCReservoir reservoir, inout float wSum, RCCandidate candidate, float randValue) {
    if (!candidate.valid || candidate.targetWeight <= 0.0) {
        return false;
    }

    wSum += candidate.targetWeight;
    reservoir.m += 1.0;
    float p = candidate.targetWeight * safeRcp(wSum);
    if (randValue < p) {
        reservoir.radiance = candidate.radiance;
        reservoir.sampleDir = candidate.dir;
        reservoir.hitPos = candidate.hitPos;
        return true;
    }

    return false;
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

uint rc_selectLevel(vec3 P) {
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

uint rc_dominantFaceId(vec3 N) {
    vec3 a = abs(N);

    if (a.x >= a.y && a.x >= a.z) {
        return N.x >= 0.0 ? RC_FACE_POS_X : RC_FACE_NEG_X;
    }

    if (a.y >= a.x && a.y >= a.z) {
        return N.y >= 0.0 ? RC_FACE_POS_Y : RC_FACE_NEG_Y;
    }

    return N.z >= 0.0 ? RC_FACE_POS_Z : RC_FACE_NEG_Z;
}

uint rc_faceAxis(uint faceId) {
    return faceId / 2u;
}

void rc_faceTangentAxes(uint faceId, out uint axis0, out uint axis1) {
    uint nAxis = rc_faceAxis(faceId);

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

void rc_lookupSampleFaceWeighted(
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

    uint entryIndex = rc_entryIndex(level, worldCellCoord);
    uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), entryIndex);
    uvec4 entry = rc_indirection[bufferIndex];

    uint worldKeyHash = rc_worldKeyHash(level, worldCellCoord);

    if (
    entry.x == RC_INVALID ||
    entry.z != worldKeyHash ||
    !rc_entryMetaValid(entry.w) ||
    rc_entryMetaLevel(entry.w) != level ||
    !rc_hasFace(entry.y, faceId)
    ) {
        result.misses++;
        return;
    }

    uint reservoirIndex = rc_faceReservoirIndex(entry.x, entry.y, faceId);

    if (reservoirIndex >= uint(SETTING_RC_POOL_SIZE)) {
        result.misses++;
        return;
    }

    RCReservoir reservoir = rc_reservoirLoad(rc_currentSide(), reservoirIndex);

    if (!rc_reservoirValid(reservoir)) {
        result.misses++;
        return;
    }

    vec3 faceNormal = rc_faceNormal(faceId);

    float normalWeight = saturate(dot(N, faceNormal));
    if (normalWeight <= 0.0) {
        result.misses++;
        return;
    }

    vec3 faceCenter = rc_faceCenter(worldCellCoord, level, faceId);

    float thickness = ldexp(1.0, int(level));

    float side = dot(P - faceCenter, faceNormal);
    if (side < -thickness) {
        result.misses++;
        return;
    }

    uint age = rc_reservoirMetaAge(reservoir.meta);

    float w = interpWeight * normalWeight;

    if (w <= 0.0) {
        result.misses++;
        return;
    }

    vec3 estimatedRadiance = rc_reservoirEstimateRadiance(reservoir);
    if (rc_luminance(estimatedRadiance) <= 0.0 || any(isnan(estimatedRadiance))) {
        result.misses++;
        return;
    }

    result.radiance += estimatedRadiance * w;
    result.weight += w;

    result.hits++;
    result.levelMask |= 1u << level;
    result.faceMask |= rc_faceBit(faceId);
    result.m = max(result.m, reservoir.m);
    result.age = max(result.age, age);
    result.debug = reservoir.meta & 0xFF;
}

RCLookupResult rc_lookupDiffuseGISmooth(vec3 P, vec3 N, vec3 geomN) {
    RCLookupResult result = rc_lookupInit();

    uint level = rc_selectLevel(P);
    float voxelSize = float(rc_voxelSize(level));

    uint faceId = rc_dominantFaceId(geomN);
    vec3 faceNormal = rc_faceNormal(faceId);

    uint axis0;
    uint axis1;
    rc_faceTangentAxes(faceId, axis0, axis1);

    uint normalAxis = rc_faceAxis(faceId);

    // Move slightly behind the queried surface so the owner cell is stable.
    // For +Y face, this moves into the solid cell below the face.
    // For -Y face, this moves into the solid cell above the face.
    float surfaceEpsilon = max(voxelSize * 1e-3, 1e-3);
    vec3 ownerP = P - faceNormal * surfaceEpsilon;

    ivec3 ownerCell = rc_worldCellCoord(ownerP, level);

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

    // Cubic smoothstep interpolation weights:
    //
    //   smoothstep01(t) = t * t * (3 - 2 * t)
    //
    // This keeps the same 2x2 footprint as bilinear interpolation, but makes
    // the transition C1-continuous inside each cell interval.
    float su = fu * fu * (3.0 - 2.0 * fu);
    float sv = fv * fv * (3.0 - 2.0 * fv);

    float w00 = (1.0 - su) * (1.0 - sv);
    float w10 = su * (1.0 - sv);
    float w01 = (1.0 - su) * sv;
    float w11 = su * sv;

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

    rc_lookupSampleFaceWeighted(result, P, N, level, cell00, faceId, w00);
    rc_lookupSampleFaceWeighted(result, P, N, level, cell10, faceId, w10);
    rc_lookupSampleFaceWeighted(result, P, N, level, cell01, faceId, w01);
    rc_lookupSampleFaceWeighted(result, P, N, level, cell11, faceId, w11);

    if (result.weight > 0.0) {
        result.radiance /= result.weight;
    }

    return result;
}
void rc_lookupSampleFace1x1(
    inout RCLookupResult result,
    vec3 P,
    vec3 N,
    uint level,
    ivec3 worldCellCoord,
    uint faceId
) {
    uint entryIndex = rc_entryIndex(level, worldCellCoord);
    uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), entryIndex);
    uvec4 entry = rc_indirection[bufferIndex];

    uint worldKeyHash = rc_worldKeyHash(level, worldCellCoord);

    if (
    entry.x == RC_INVALID ||
    entry.z != worldKeyHash ||
    !rc_entryMetaValid(entry.w) ||
    rc_entryMetaLevel(entry.w) != level ||
    !rc_hasFace(entry.y, faceId)
    ) {
        result.misses++;
        return;
    }

    uint reservoirIndex = rc_faceReservoirIndex(entry.x, entry.y, faceId);

    if (reservoirIndex >= uint(SETTING_RC_POOL_SIZE)) {
        result.misses++;
        return;
    }

    RCReservoir reservoir = rc_reservoirLoad(rc_currentSide(), reservoirIndex);

    if (!rc_reservoirValid(reservoir)) {
        result.misses++;
        return;
    }

    vec3 faceNormal = rc_faceNormal(faceId);

    float normalWeight = max(dot(N, faceNormal), 0.0);
    if (normalWeight <= 0.0) {
        result.misses++;
        return;
    }

    vec3 faceCenter = rc_faceCenter(worldCellCoord, level, faceId);

    float thickness = ldexp(2.0, int(level));

    float side = dot(P - faceCenter, faceNormal);
    if (side < -thickness) {
        result.misses++;
        return;
    }

    uint age = rc_reservoirMetaAge(reservoir.meta);

    // 1x1 lookup: no bilinear and no tangent distance filter.
    // Weight only by normal compatibility and history freshness.
    float w = normalWeight;

    if (w <= 0.0) {
        result.misses++;
        return;
    }

    vec3 estimatedRadiance = rc_reservoirEstimateRadiance(reservoir);
    if (rc_luminance(estimatedRadiance) <= 0.0 || any(isnan(estimatedRadiance))) {
        result.misses++;
        return;
    }

    result.radiance += estimatedRadiance * w;
    result.weight += w;

    result.hits++;
    result.levelMask |= 1u << level;
    result.faceMask |= rc_faceBit(faceId);
    result.m = max(result.m, reservoir.m);
    result.age = max(result.age, age);
    result.debug = reservoir.meta & 0xFF;
}

RCLookupResult rc_lookupDiffuseGI(vec3 P, vec3 N, vec3 geomN) {
    RCLookupResult result = rc_lookupInit();

    uint level = rc_selectLevel(P);
    float voxelSize = float(rc_voxelSize(level));

    uint faceId = rc_dominantFaceId(geomN);
    vec3 faceNormal = rc_faceNormal(faceId);

    // Move into the owner voxel so the face owner is stable.
    // For +Y face, this moves slightly below the surface.
    // For -Y face, this moves slightly above the surface.
    float surfaceEpsilon = max(voxelSize * 1e-3, 1e-3);
    vec3 ownerP = P - faceNormal * surfaceEpsilon;

    ivec3 ownerCell = rc_worldCellCoord(ownerP, level);

    rc_lookupSampleFace1x1(
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

bool rc_touchFace(uint level, ivec3 worldCellCoord, uint faceId) {
    uint entryIndex = rc_entryIndex(level, worldCellCoord);
    uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), entryIndex);
    uint worldKeyHash = rc_worldKeyHash(level, worldCellCoord);
    uint oldKey = atomicCompSwap(rc_indirection[bufferIndex].z, RC_INVALID, worldKeyHash);
    if (oldKey == RC_INVALID || oldKey == worldKeyHash) {
        uvec4 entry = rc_indirection[bufferIndex];
        uint oldFaceMask = entry.y & 0x3fu;
        uint newFaceMask = oldFaceMask | rc_faceBit(faceId);
        bool canGrowFaceMask = entry.x == RC_INVALID || newFaceMask == oldFaceMask;
        if (!canGrowFaceMask) {
            uint allocatedClassSize = rc_allocClassSize(bitCount(oldFaceMask));
            canGrowFaceMask = bitCount(newFaceMask) <= allocatedClassSize;
        }
        if (!canGrowFaceMask) {
            return false;
        }

        atomicOr(rc_indirection[bufferIndex].y, rc_faceBit(faceId));
        uint pendingFaceBits = rc_indirection[bufferIndex].w & RC_ENTRY_META_PENDING_FACE_MASK;
        rc_indirection[bufferIndex].w = rc_packEntryMeta(level, true) | pendingFaceBits;
        return true;
    } else {
        atomicAdd(rc_keyMismatchCounter, 1u);
        return false;
    }
}

#endif
