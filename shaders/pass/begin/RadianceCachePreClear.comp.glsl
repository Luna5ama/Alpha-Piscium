#define RC_DATA_MODIFIER restrict buffer

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(256, 1, 1);
#include "/techniques/gi/RadianceCache.glsl"

void rc_clearEntry(uint listIndex) {
    uint bufferIndex;
    if (rc_activeClearCount > RC_ENTRY_COUNT) {
        bufferIndex = listIndex;
    } else {
        uint currentSide = rc_currentSide();
        uint entryIndex = rc_activeEntryIndices[rc_activeEntryListIndex(currentSide, listIndex)];
        bufferIndex = rc_bufferEntryIndex(currentSide, entryIndex);
    }
    rc_indirection[bufferIndex] = uvec4(RC_INVALID, 0u, RC_INVALID, 0u);
}

void main() {
    for (uint listIndex = gl_GlobalInvocationID.x; listIndex < rc_activeClearCount; listIndex += 65536u) {
        rc_clearEntry(listIndex);
    }
}
