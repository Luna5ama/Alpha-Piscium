#define RC_DATA_MODIFIER restrict buffer
#include "/Base.glsl"
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

    if (idx >= RC_ENTRY_COUNT) {
        return;
    }

    rc_indirection[rcBufferEntryIndex(rcCurrentSide(), idx)] = uvec4(RC_INVALID, 0u, RC_INVALID, 0u);
}
