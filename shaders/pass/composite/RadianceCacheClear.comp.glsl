#define RC_DATA_MODIFIER restrict buffer
#include "/techniques/gi/RadianceCache.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(5120, 1, 1);

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx == 0u) {
        rc_allocationCounter = 0u;
        rc_keyMismatchCounter = 0u;
        rc_poolOverflowCounter = 0u;
        rc_cacheHitCounter = 0u;
        rc_cacheMissCounter = 0u;
    }

    if (idx < RC_ENTRY_COUNT) {
        uint currentBufferIndex = rcBufferEntryIndex(rcCurrentSide(), idx);
        uint previousBufferIndex = rcBufferEntryIndex(rcPreviousSide(), idx);
        uvec4 currentEntry = rc_indirection[currentBufferIndex];
        uvec4 previousEntry = rc_indirection[previousBufferIndex];
        uint level = rcEntryLevel(idx);
        ivec3 worldCellCoord = rcWorldCellCoordFromEntryIndex(idx);
        uint worldKeyHash = rcWorldKeyHash(level, worldCellCoord);
        uint previousFaceMask = previousEntry.y & 0x3fu;
        uint pendingVisibleFaceMask = rcEntryMetaPendingFaceMask(currentEntry.w);

        if (previousEntry.z == worldKeyHash
            && rcEntryMetaValid(previousEntry.w)
            && rcEntryMetaLevel(previousEntry.w) == level
            && previousFaceMask != 0u
        ) {
            uint carriedFaceMask = previousFaceMask & pendingVisibleFaceMask;
            if (carriedFaceMask != 0u) {
                uint age = min(rcEntryMetaAge(previousEntry.w) + 1u, 255u);
                rc_indirection[currentBufferIndex] = uvec4(
                    RC_INVALID,
                    carriedFaceMask,
                    worldKeyHash,
                    rcEntryMetaClearPendingFaces(rcPackEntryMeta(level, age, true))
                );
                return;
            }
        }

        rc_indirection[currentBufferIndex] = uvec4(RC_INVALID, 0u, RC_INVALID, rcEntryMetaClearPendingFaces(0u));
    }
}
