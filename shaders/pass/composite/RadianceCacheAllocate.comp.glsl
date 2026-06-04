#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#define RC_DATA_MODIFIER restrict buffer

layout(local_size_x = 256) in;
#include "/techniques/gi/RadianceCacheUpdate.glsl"

const ivec3 workGroups = ivec3(5120, 1, 1);

void main() {
    uint entryIndex = gl_GlobalInvocationID.x;
    uint currentSide = rc_currentSide();
    uint poolSize = uint(SETTING_RC_POOL_SIZE);

    uint bufferIndex = 0u;
    uvec4 entry = uvec4(RC_INVALID, 0u, RC_INVALID, 0u);
    if (entryIndex < RC_ENTRY_COUNT) {
        bufferIndex = rc_bufferEntryIndex(currentSide, entryIndex);
        entry = rc_indirection[bufferIndex];
    }

    uint faceMask = entry.y & 0x3fu;
    bool activeFlag = entryIndex < RC_ENTRY_COUNT
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
        uint subgroupOverflowCount = subgroupAdd(overflow ? 1u : 0u);
        if (subgroupOverflowCount != 0u) {
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
                rc_indirection[bufferIndex].x = RC_INVALID;
                rc_indirection[bufferIndex].y = 0u;
            } else {
                rc_indirection[bufferIndex].x = reservoirBaseIndex;

                for (uint i = 0; i < 6u; i++) {
                    uint faceBit = 1u << i;
                    if ((faceMask & faceBit) != 0u) {
                        rc_updateEntryIndices[updateEntryIndicesBaseIndex++] = bitfieldInsert(entryIndex, i, 26, 6);
                    }
                }
            }
        }
    }
}
