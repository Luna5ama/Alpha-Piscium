#define RC_DATA_MODIFIER restrict buffer
#include "/Base.glsl"
#include "/techniques/gi/RadianceCache.glsl"
#include "/techniques/voxel/Voxelization.glsl"
#include "/util/MaterialIDConst.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(5120, 1, 1);

bool rcCarryFaceValid(uint level, ivec3 worldCellCoord, uint faceId) {
    vec3 faceNormal = rcFaceNormal(faceId);
    ivec3 faceNormalI = ivec3(faceNormal);
    ivec3 ownerBlock = ivec3(floor(rcFaceCenter(worldCellCoord, level, faceId) - faceNormal * 0.02));
    return voxel_opaqueAtBlock(ownerBlock) && !voxel_opaqueAtBlock(ownerBlock + faceNormalI);
}

uint rcCarryFaceMask(uint level, ivec3 worldCellCoord, uint faceMask) {
    uint carriedFaceMask = 0u;
    for (uint faceId = 0u; faceId < 6u; faceId++) {
        if (rcHasFace(faceMask, faceId) && rcCarryFaceValid(level, worldCellCoord, faceId)) {
            carriedFaceMask |= rcFaceBit(faceId);
        }
    }
    return carriedFaceMask;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx == 0u) {
        rc_allocationCounter = 0u;
        rc_keyMismatchCounter = 0u;
        rc_poolOverflowCounter = 0u;
        rc_cacheHitCounter = 0u;
        rc_cacheMissCounter = 0u;
    }

    if (idx >= RC_ENTRY_COUNT) {
        return;
    }

    uint currentBufferIndex = rcBufferEntryIndex(rcCurrentSide(), idx);
    uint previousBufferIndex = rcBufferEntryIndex(rcPreviousSide(), idx);
    uvec4 previousEntry = rc_indirection[previousBufferIndex];
    uint level = rcEntryLevel(idx);
    ivec3 worldCellCoord = rcWorldCellCoordFromEntryIndex(idx);
    uint worldKeyHash = rcWorldKeyHash(level, worldCellCoord);
    uint faceMask = previousEntry.y & 0x3fu;

    if (
        previousEntry.z == worldKeyHash
        && rcEntryMetaValid(previousEntry.w)
        && rcEntryMetaLevel(previousEntry.w) == level
        && faceMask != 0u
    ) {
        uint carriedFaceMask = rcCarryFaceMask(level, worldCellCoord, faceMask);
        if (carriedFaceMask != 0u) {
            uint age = min(rcEntryMetaAge(previousEntry.w) + 1u, 255u);
            rc_indirection[currentBufferIndex] = uvec4(
                RC_INVALID,
                carriedFaceMask,
                worldKeyHash,
                rcPackEntryMeta(level, age, true)
            );
            return;
        }
    }

    rc_indirection[currentBufferIndex] = uvec4(RC_INVALID, 0u, RC_INVALID, 0u);
}
