layout(local_size_x = 128) in;

#define VOXEL_BLOCK_MODEL_LINEAR_AABB_TEXELS
#include "/techniques/gi/RadianceCacheUpdate.glsl"

void main() {
    voxel_initShared();

    if (gl_GlobalInvocationID.x < rc_entryCounter) {
        uint data = rc_updateEntryIndices[gl_GlobalInvocationID.x];
        uint entryIndex = bitfieldExtract(data, 0, 26);
        uint faceId = bitfieldExtract(data, 26, 6);
        if (entryIndex < RC_ENTRY_COUNT) {
            uint level = rc_entryLevel(entryIndex);
            ivec3 worldCellCoord = rc_worldCellCoordFromEntryIndex(entryIndex);
            uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), entryIndex);
            uvec4 entry = rc_indirection[bufferIndex];
            if (entry.x != RC_INVALID && entry.z == rc_worldKeyHash(level, worldCellCoord) && rc_entryMetaValid(entry.w) && rc_entryMetaLevel(entry.w) == level) {
                uint faceMask = entry.y & 0x3fu;
                if (rc_hasFace(faceMask, faceId)) {
                    rc_updateFace(entryIndex, entry, worldCellCoord, level, faceId);
                }
            }
        }
    }
}
