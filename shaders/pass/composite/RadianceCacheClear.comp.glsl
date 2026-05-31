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
        uint currentBufferIndex = rc_bufferEntryIndex(rc_currentSide(), idx);
        uint previousBufferIndex = rc_bufferEntryIndex(rc_previousSide(), idx);
        uint previousFeedbackIndex = rc_feedbackRecordIndex(rc_previousSide(), idx);
        uvec4 currentEntry = rc_indirection[currentBufferIndex];
        uvec4 previousEntry = rc_indirection[previousBufferIndex];
        uint level = rc_entryLevel(idx);
        ivec3 worldCellCoord = rc_worldCellCoordFromEntryIndex(idx);
        uint worldKeyHash = rc_worldKeyHash(level, worldCellCoord);
        uint previousFaceMask = previousEntry.y & 0x3fu;
        uint pendingVisibleFaceMask = rc_entryMetaPendingFaceMask(currentEntry.w);
        uvec2 previousFeedback = rc_feedback[previousFeedbackIndex];

        uint carriedFaceMask = 0u;
        if (previousEntry.z == worldKeyHash
            && rc_entryMetaValid(previousEntry.w)
            && rc_entryMetaLevel(previousEntry.w) == level
            && previousFaceMask != 0u
        ) {
            carriedFaceMask = previousFaceMask;
        }

        uint feedbackFaceMask = 0u;
        if (previousFeedback.x == worldKeyHash) {
            feedbackFaceMask = (previousFeedback.y >> RC_FEEDBACK_HIT_SHIFT) & RC_FEEDBACK_FACE_MASK;
        }

        uint newFaceMask = (carriedFaceMask | feedbackFaceMask) & pendingVisibleFaceMask;
        if (newFaceMask != 0u) {
            rc_indirection[currentBufferIndex] = uvec4(
                RC_INVALID,
                newFaceMask,
                worldKeyHash,
                rc_entryMetaClearPendingFaces(rc_packEntryMeta(level, true))
            );
        } else {
            rc_indirection[currentBufferIndex] = uvec4(RC_INVALID, 0u, RC_INVALID, rc_entryMetaClearPendingFaces(0u));
        }

        rc_feedbackClearRecord(rc_currentSide(), idx);
    }
}
