#ifndef INCLUDE_techniques_gi_RadianceCache_glsl
#define INCLUDE_techniques_gi_RadianceCache_glsl a

#include "/util/Hash.glsl"
#include "/util/Math.glsl"

#define RC_CLIP_LEVELS 5u
#define RC_CLIP_SIZE 64u
#define RC_CLIP_MASK 63u
#define RC_INVALID 0xffffffffu
#define RC_ENTRY_COUNT 1310720u
#define RC_RESERVOIR_RECORDS 4u

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

#define RC_FEEDBACK_SCREEN_SHIFT 0u
#define RC_FEEDBACK_HIT_SHIFT 6u
#define RC_FEEDBACK_FACE_MASK 0x3fu

#ifndef RC_DATA_MODIFIER
#define RC_DATA_MODIFIER restrict buffer
#endif

const float RC_MAX_ROUGHNESS = 0.25;

// x: reservoir base index, or RC_INVALID if no reservoir allocated for this entry
// y: bitmask of valid faces
// z: world key hash
// w: meta (bits 0-2: level, bit 31: valid flag, bits 16-23: pending visible face mask)
layout(std430, binding = 10) RC_DATA_MODIFIER RadianceCacheIndirectionData {
    uvec4 rc_indirection[];
};

layout(std430, binding = 11) RC_DATA_MODIFIER RadianceCacheReservoirData {
    uvec4 rc_reservoirs[];
};

struct RCReservoir {
    vec3 radiance;
    float avgWY;
    vec3 sampleDir;
    float m;
    vec3 hitPos;
    uint meta;
    vec3 estimate;
};

RCReservoir rc_reservoirInit() {
    RCReservoir reservoir;
    reservoir.radiance = vec3(0.0);
    reservoir.avgWY = 0.0;
    reservoir.sampleDir = vec3(0.0, 1.0, 0.0);
    reservoir.m = 0.0;
    reservoir.hitPos = vec3(0.0);
    reservoir.meta = 0u;
    reservoir.estimate = vec3(0.0);
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
    uvec4 r3 = rc_reservoirs[recordIndex + 3u];
    RCReservoir reservoir;
    reservoir.radiance = uintBitsToFloat(r0.xyz);
    reservoir.avgWY = uintBitsToFloat(r0.w);
    reservoir.sampleDir = uintBitsToFloat(r1.xyz);
    reservoir.m = uintBitsToFloat(r1.w);
    reservoir.hitPos = uintBitsToFloat(r2.xyz);
    reservoir.meta = r2.w;
    reservoir.estimate = uintBitsToFloat(r3.xyz);
    return reservoir;
}

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
    float halfVoxel = ldexp(0.5, int(level));
    vec3 cellMin = ldexp(worldCellCoord, ivec3(level));
    vec3 center = cellMin + halfVoxel;
    center += rc_faceNormal(faceId) * halfVoxel;
    return center;
}

vec3 rc_faceRayOrigin(ivec3 worldCellCoord, uint level, uint faceId) {
    float halfVoxel = ldexp(0.5, int(level));
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

void rc_reservoirStore(uint side, uint reservoirIndex, RCReservoir reservoir) {
    uint recordIndex = rc_reservoirRecordIndex(side, reservoirIndex);
    rc_reservoirs[recordIndex + 0u] = uvec4(floatBitsToUint(reservoir.radiance), floatBitsToUint(reservoir.avgWY));
    rc_reservoirs[recordIndex + 1u] = uvec4(floatBitsToUint(reservoir.sampleDir), floatBitsToUint(reservoir.m));
    rc_reservoirs[recordIndex + 2u] = uvec4(floatBitsToUint(reservoir.hitPos), reservoir.meta);
    rc_reservoirs[recordIndex + 3u] = uvec4(floatBitsToUint(reservoir.estimate), 0u);
}

float rc_luminance(vec3 radiance) {
    return length(radiance);
}

vec3 rc_reservoirEstimateRadiance(RCReservoir reservoir) {
    vec3 result = vec3(0.0);

    if (rc_reservoirValid(reservoir)) {
        result = reservoir.estimate;
    }

    return result;
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

#endif
