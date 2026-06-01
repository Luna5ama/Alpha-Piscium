#define RC_DATA_MODIFIER restrict buffer
#include "/techniques/gi/RadianceCache.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(5120, 1, 1);

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= RC_ENTRY_COUNT) {
        return;
    }

    uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), idx);
    uvec4 entry = rc_indirection[bufferIndex];
    uint faceMask = entry.y & 0x3fu;
    if (faceMask == 0u || entry.z == RC_INVALID || !rc_entryMetaValid(entry.w)) {
        return;
    }

    uint faceCount = bitCount(faceMask);
    uint classSize = rc_allocClassSize(faceCount);
    uint baseIndex = atomicAdd(rc_allocationCounter, classSize);
    if (baseIndex + classSize > uint(SETTING_RC_POOL_SIZE)) {
        rc_indirection[bufferIndex].x = RC_INVALID;
        rc_indirection[bufferIndex].y = 0u;
        atomicAdd(rc_poolOverflowCounter, 1u);
        return;
    }

    rc_indirection[bufferIndex].x = baseIndex;
}
