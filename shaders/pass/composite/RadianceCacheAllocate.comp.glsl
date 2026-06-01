#define RC_DATA_MODIFIER restrict buffer
#include "/techniques/gi/RadianceCache.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(5120, 1, 1);

void main() {
    uint entryIndex = gl_GlobalInvocationID.x;
    if (entryIndex >= RC_ENTRY_COUNT) {
        return;
    }

    uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), entryIndex);
    uvec4 entry = rc_indirection[bufferIndex];
    uint faceMask = entry.y & 0x3fu;
    if (faceMask == 0u || entry.z == RC_INVALID || !rc_entryMetaValid(entry.w)) {
        return;
    }

    uint faceCount = bitCount(faceMask);
    uint classSize = rc_allocClassSize(faceCount);
    uint reservoirBaseIndex = atomicAdd(rc_allocationCounter, classSize);
    if (reservoirBaseIndex + classSize > uint(SETTING_RC_POOL_SIZE)) {
        rc_indirection[bufferIndex].x = RC_INVALID;
        rc_indirection[bufferIndex].y = 0u;
        atomicAdd(rc_poolOverflowCounter, 1u);
        return;
    }

    rc_indirection[bufferIndex].x = reservoirBaseIndex;

    uint writeIndex = atomicAdd(rc_entryCounter, faceCount);
    uint offset = 0u;
    for (uint i = 0; i < 6u; i++) {
        uint faceBit = 1u << i;
        if ((faceMask & faceBit) != 0u) {
            rc_updateEntryIndices[writeIndex++] = bitfieldInsert(entryIndex, i, 26, 6);
        }
    }
}
