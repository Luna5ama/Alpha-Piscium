// Carries each RC face's path guide statistics from its previous-side slot to its current slot,
// with exponential decay and a sample-count cap. Faces without history start empty.
#define PG_DATA_MODIFIER restrict buffer

layout(local_size_x = 128) in;

#include "/techniques/gi/RadianceCacheUpdate.glsl"
#include "/techniques/gi/PathGuiding.glsl"

void main() {
    if (gl_GlobalInvocationID.x >= rc_entryCounter) {
        return;
    }
    uint data = rc_updateEntryIndices[gl_GlobalInvocationID.x];
    uint entryIndex = bitfieldExtract(data, 0, 26);
    uint faceId = bitfieldExtract(data, 26, 6);
    if (entryIndex >= RC_ENTRY_COUNT) {
        return;
    }
    uint level = rc_entryLevel(entryIndex);
    uint worldKeyHash = rc_worldKeyHash(level, rc_worldCellCoordFromEntryIndex(entryIndex));
    uvec4 entry = rc_indirection[rc_bufferEntryIndex(rc_currentSide(), entryIndex)];
    if (
        entry.x == RC_INVALID
        || entry.z != worldKeyHash
        || !rc_entryMetaValid(entry.w)
        || rc_entryMetaLevel(entry.w) != level
        || !rc_hasFace(entry.y, faceId)
    ) {
        return;
    }
    uint slot = rc_faceReservoirIndex(entry.x, entry.y, faceId);
    if (slot >= uint(SETTING_RC_POOL_SIZE)) {
        return;
    }

    uint statsIndex = pathGuide_statsIndex(rc_currentSide(), slot);
    uvec4 stats0;
    uvec4 stats1;
    pathGuide_previousStats(entryIndex, worldKeyHash, level, faceId, stats0, stats1);
    pg_stats[statsIndex] = stats0;
    pg_stats[statsIndex + 1u] = stats1;
}
