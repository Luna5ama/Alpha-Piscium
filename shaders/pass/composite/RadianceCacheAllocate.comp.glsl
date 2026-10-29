#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#define RC_DATA_MODIFIER restrict buffer

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(256, 1, 1);
#include "/techniques/gi/RadianceCacheUpdate.glsl"

void rc_allocateEntry(uint listIndex) {
    uint currentSide = rc_currentSide();
    bool listed = listIndex < rc_activeEntryCounts[currentSide];
    uint entryIndex = listed
        ? rc_activeEntryIndices[rc_activeEntryListIndex(currentSide, listIndex)]
        : RC_ENTRY_COUNT;
    uint poolSize = uint(SETTING_RC_POOL_SIZE);

    uint bufferIndex = 0u;
    uvec4 entry = uvec4(RC_INVALID, 0u, RC_INVALID, 0u);
    bool preparedEntry = false;
    if (listed) {
        bufferIndex = rc_bufferEntryIndex(currentSide, entryIndex);
        entry = rc_indirection[bufferIndex];

        uint pendingVisibleFaceMask = rc_entryMetaPendingFaceMask(entry.w);
        if (pendingVisibleFaceMask != 0u) {
            preparedEntry = true;
            uint level = rc_entryLevel(entryIndex);
            ivec3 worldCellCoord = rc_worldCellCoordFromEntryIndex(entryIndex);
            uint worldKeyHash = rc_worldKeyHash(level, worldCellCoord);

            uvec4 previousEntry = rc_indirection[rc_bufferEntryIndex(rc_previousSide(), entryIndex)];
            uint previousFaceMask = previousEntry.y & 0x3fu;
            bool previousKeyMatch = previousEntry.z == worldKeyHash;
            bool previousValid = previousKeyMatch
                && rc_entryMetaValid(previousEntry.w)
                && rc_entryMetaLevel(previousEntry.w) == level;
            uint carriedFaceMask = previousValid ? previousFaceMask : 0u;
            uint feedbackFaceMask = previousKeyMatch ? rc_entryMetaHitFaceMask(previousEntry.w) : 0u;

            bool currentKeyMatch = entry.z == RC_INVALID || entry.z == worldKeyHash;
            uint touchedFaceMask = currentKeyMatch ? entry.y & 0x3fu : 0u;
            uint faceMask = ((carriedFaceMask | feedbackFaceMask) & pendingVisibleFaceMask) | touchedFaceMask;
            uint screenFaceBits = currentKeyMatch ? entry.w & RC_ENTRY_META_SCREEN_FACE_MASK : 0u;

            // Keep list membership until PreClear, even when no faces need allocation.
            entry = uvec4(RC_INVALID, 0u, RC_INVALID, RC_ENTRY_META_LISTED);
            if (faceMask != 0u) {
                entry = uvec4(
                    RC_INVALID,
                    faceMask,
                    worldKeyHash,
                    rc_packEntryMeta(level, true) | screenFaceBits | RC_ENTRY_META_LISTED
                );
            }
        }
    }

    uint faceMask = entry.y & 0x3fu;
    bool activeFlag = listed
        && faceMask != 0u
        && entry.z != RC_INVALID
        && rc_entryMetaValid(entry.w);

    if (subgroupAny(activeFlag)) {
        uint faceCount = activeFlag ? bitCount(faceMask) : 0u;
        uint classSize = activeFlag ? rc_allocClassSize(faceCount) : 0u;

        uint classSizePrefix = subgroupExclusiveAdd(classSize);

        uint reservoirBaseIndex = 0u;
        if (gl_SubgroupInvocationID == gl_SubgroupSize - 1) {
            uint totalClassSize = classSizePrefix + classSize;
            if (totalClassSize != 0u) {
                reservoirBaseIndex = atomicAdd(rc_allocationCounter, totalClassSize);
            }
        }

        reservoirBaseIndex = subgroupBroadcast(reservoirBaseIndex, gl_SubgroupSize - 1) + classSizePrefix;
        bool overflow = activeFlag && reservoirBaseIndex + classSize > poolSize;
        if (subgroupAny(overflow)) {
            uvec4 overflowMask = subgroupBallot(overflow);
            uint subgroupOverflowCount = uint(bitCount(overflowMask.x) + bitCount(overflowMask.y) + bitCount(overflowMask.z) + bitCount(overflowMask.w));
            if (subgroupElect()) {
                atomicAdd(rc_poolOverflowCounter, subgroupOverflowCount);
            }
        }


        uint validFaceCount = activeFlag && !overflow ? faceCount : 0u;
        uint validFaceCountPrefix = subgroupExclusiveAdd(validFaceCount);
        uint updateEntryIndicesBaseIndex = 0u;
        if (gl_SubgroupInvocationID == gl_SubgroupSize - 1) {
            uint totalFaceCount = validFaceCountPrefix + validFaceCount;
            if (totalFaceCount != 0u) {
                updateEntryIndicesBaseIndex = atomicAdd(rc_entryCounter, totalFaceCount);
            }
        }
        updateEntryIndicesBaseIndex = subgroupBroadcast(updateEntryIndicesBaseIndex, gl_SubgroupSize - 1) + validFaceCountPrefix;

        if (activeFlag) {
            if (overflow) {
                entry.x = RC_INVALID;
                entry.y = 0u;
            } else {
                entry.x = reservoirBaseIndex;

                for (uint i = 0; i < 6u; i++) {
                    uint faceBit = 1u << i;
                    if ((faceMask & faceBit) != 0u) {
                        rc_updateEntryIndices[updateEntryIndicesBaseIndex++] = bitfieldInsert(entryIndex, i, 26, 6);
                    }
                }
            }

            if (!preparedEntry) {
                rc_indirection[bufferIndex].x = entry.x;
                if (overflow) {
                    rc_indirection[bufferIndex].y = 0u;
                }
            }
        }
    }

    if (preparedEntry) {
        rc_indirection[bufferIndex] = entry;
    }
}

void main() {
    uint count = rc_activeEntryCounts[rc_currentSide()];
    for (uint groupBase = gl_WorkGroupID.x * 256u; groupBase < count; groupBase += 65536u) {
        rc_allocateEntry(groupBase + gl_LocalInvocationID.x);
    }
}
