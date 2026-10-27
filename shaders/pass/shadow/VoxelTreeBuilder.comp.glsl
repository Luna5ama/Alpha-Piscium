// 64-Tree Builder – runs after the shadow pass (prepare2).
//
// Builds the bottom 2 levels of the dense 64-tree for each allocated brick:
//   Level 1 (leaf, 64 uvec2 per brick): bit k = 1 if block k in sub-region is solid.
//   Level 2 (brick root, 1 uvec2 per brick): bit j = 1 if sub-region j has any solid block.
//
// Tree nodes are written to the dense layout (indexed by brickMorton, not allocID).
// Material data is still read via allocID (sparse pool).
//
// Dispatch: one workgroup per four slots, capped by the grid capacity.
// Threads per workgroup: 256 (64 per brick × 4 bricks, one thread per sub-region).
//
// This pass owns L2 for every brick and writes L1 only for non-empty sub-regions.
// Stale L1 slots are unreachable because traversal tests the corresponding L2 bit first.
// Upper levels (L3–L5) are always written unconditionally by the propagator passes,
// so they are self-clearing when all children below them are zero.
//
// Block Morton contiguity: within sub-region S, blockMorton = S*64 + blockInSr.
// All 64 blocks are contiguous in voxel_materials[], so we read 4 at a time as uvec4.
// Base index (allocID*4096 + subRegion*64) is always divisible by 4.
//
// Root mask accumulation uses parallel reduction (subgroupOr) instead of atomicOr.
// subgroupElect() thread per subgroup → 2 atomicOrs per brick into rootMaskLo/Hi.

#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable

#define VOXEL_BRICK_DATA_MODIFIER restrict readonly buffer
#define VOXEL_MATERIAL_VEC4 a
#define VOXEL_MATERIAL_DATA_MODIFIER restrict readonly buffer
#define VOXEL_TREE_DATA_MODIFIER buffer
#define RC_DATA_MODIFIER restrict buffer
#include "/techniques/voxel/Voxelization.glsl"
#include "/techniques/gi/RadianceCache.glsl"

layout(std430, binding = 8) VOXEL_TREE_DATA_MODIFIER VoxelTreeData {
    uvec2 voxel_tree[];       // VOXEL_TREE_TOTAL uvec2 entries
};

layout(local_size_x = 256) in;
// One workgroup per four possible allocated bricks.
#if VOXEL_POOL_SIZE == 256
const ivec3 workGroups = ivec3(64, 1, 1);
#elif VOXEL_POOL_SIZE == 512
const ivec3 workGroups = ivec3(128, 1, 1);
#elif VOXEL_POOL_SIZE == 1024
const ivec3 workGroups = ivec3(256, 1, 1);
#elif VOXEL_POOL_SIZE == 2048
const ivec3 workGroups = ivec3(512, 1, 1);
#elif VOXEL_POOL_SIZE == 4096 || VOXEL_GRID_SIZE == 16
const ivec3 workGroups = ivec3(1024, 1, 1);
#elif VOXEL_POOL_SIZE == 8192
const ivec3 workGroups = ivec3(2048, 1, 1);
#elif VOXEL_POOL_SIZE == 16384
const ivec3 workGroups = ivec3(4096, 1, 1);
#elif VOXEL_POOL_SIZE == 32768 || VOXEL_GRID_SIZE == 32
const ivec3 workGroups = ivec3(8192, 1, 1);
#elif VOXEL_POOL_SIZE == 65536
const ivec3 workGroups = ivec3(16384, 1, 1);
#endif

shared uint rootMaskLo[4];
shared uint rootMaskHi[4];
shared uint adjacentBrickAllocIDs[24];
shared uvec2 brickOpaqueMasks[256];

bool voxel_isGIOpaqueMaterial(uint materialData) {
    return materialData >= 4u;
}

uint voxel_loadMaterialData(uint allocID, uint blockMorton) {
    uint packedIndex = allocID * 1024u + (blockMorton >> 2u);
    uvec4 packedMaterialData = voxel_materials_v4[packedIndex];
    uint lane = blockMorton & 3u;
    if (lane == 0u) return packedMaterialData.x;
    if (lane == 1u) return packedMaterialData.y;
    if (lane == 2u) return packedMaterialData.z;
    return packedMaterialData.w;
}

void rc_markPendingVisibleFace(uint level, vec3 ownerBlockCenter, uint faceBits) {
    ivec3 worldCellCoord = rc_worldCellCoord(ownerBlockCenter, level);
    if (rc_worldCellInCurrentClip(level, worldCellCoord)) {
        uint entryIndex = rc_entryIndex(level, worldCellCoord);
        uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), entryIndex);
        atomicOr(rc_indirection[bufferIndex].w, rc_entryMetaPendingFaceBits(faceBits));
    }
}

void main() {
    for (uint brick = gl_GlobalInvocationID.x; brick < uint(VOXEL_GRID_BRICKS);
            brick += gl_NumWorkGroups.x * gl_WorkGroupSize.x) {
        if (voxel_brickAllocID[brick] == VOXEL_UNALLOCATED) {
            voxel_tree[uint(VOXEL_TREE_OFFSET_L2) + brick] = uvec2(0u);
        }
    }

    uint groupBase = gl_WorkGroupID.x * 4u;
    uint allocationCount = voxel_brickAllocCounter;
    if (groupBase >= allocationCount) return;

    uint localID    = gl_LocalInvocationID.x;
    uint groupBrick = localID >> 6u;        // 0..3 – which of the 4 bricks in this workgroup
    uint subRegion  = localID & 63u;        // 0..63 – sub-region index within the brick
    uint allocID = groupBase + groupBrick;
    uint brickMorton = 0u;
    if (allocID < allocationCount) {
        brickMorton = voxel_allocatedBricks[allocID];
    } else {
        allocID = VOXEL_UNALLOCATED;
    }

    if (subRegion == 0u) {
        rootMaskLo[groupBrick] = 0u;
        rootMaskHi[groupBrick] = 0u;
    }
    uint leafLow  = 0u;
    uint leafHigh = 0u;
    uint opaqueLow = 0u;
    uint opaqueHigh = 0u;

    if (localID < 24u) {
        uint adjacentGroupBrick = localID / 6u;
        uint faceId = localID - adjacentGroupBrick * 6u;
        uint adjacentSlot = groupBase + adjacentGroupBrick;
        uint adjacentAllocID = VOXEL_UNALLOCATED;
        if (adjacentSlot < allocationCount) {
            uint adjacentBrickMorton = voxel_allocatedBricks[adjacentSlot];
            ivec3 adjacentBrickCoord = ivec3(morton3D_30bDecode(adjacentBrickMorton));
            uint axis = faceId >> 1u;
            adjacentBrickCoord[axis] += (faceId & 1u) != 0u ? -1 : 1;
            if (all(greaterThanEqual(adjacentBrickCoord, ivec3(0))) &&
                all(lessThan(adjacentBrickCoord, ivec3(VOXEL_GRID_SIZE)))) {
                adjacentAllocID = voxel_brickAllocID[voxel_brickMorton(adjacentBrickCoord)];
            }
        }
        adjacentBrickAllocIDs[localID] = adjacentAllocID;
    }

    if (allocID != VOXEL_UNALLOCATED) {
        // Within this sub-region, blockMorton = subRegion * 64 + blockInSr.
        // All 64 entries are contiguous, so read 4-at-a-time as uvec4.
        uint baseIdx = allocID * 1024u + subRegion * 16u;

        for (uint i = 0u; i < 16u; i++) {
            uvec4 mats = voxel_materials_v4[baseIdx + i];
            uvec4 bits4 = uvec4(notEqual(mats, uvec4(0u))) << uvec4(0u, 1u, 2u, 3u);
            uint bits = bits4.x + bits4.y + bits4.z + bits4.w;
            uvec4 opaqueBits4 = uvec4(greaterThanEqual(mats, uvec4(4u))) << uvec4(0u, 1u, 2u, 3u);
            uint opaqueBits = opaqueBits4.x + opaqueBits4.y + opaqueBits4.z + opaqueBits4.w;
            if (i < 8u) {
                leafLow |= bits << (i * 4u);
                opaqueLow |= opaqueBits << (i * 4u);
            } else {
                leafHigh |= bits << ((i - 8u) * 4u);
                opaqueHigh |= opaqueBits << ((i - 8u) * 4u);
            }
        }
    }

    brickOpaqueMasks[localID] = uvec2(opaqueLow, opaqueHigh);
    barrier();

    if (allocID != VOXEL_UNALLOCATED) {
        ivec3 cameraBrick = cameraPositionInt >> 4;
        ivec3 gridOrigin = (cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4;
        uvec3 brickCoordU = morton3D_30bDecode(brickMorton);
        ivec3 brickBlockBase = ivec3(brickCoordU << 4u);
        ivec3 subRegionCoord = ivec3(morton3D_6bDecode(subRegion));
        ivec3 subRegionBlockBase = subRegionCoord << 2;
        uint coarseFaceBits = 0u;
        for (uint wordIndex = 0u; wordIndex < 2u; wordIndex++) {
            uint opaqueMask = wordIndex == 0u ? opaqueLow : opaqueHigh;
            while (opaqueMask != 0u) {
                uint blockInSubRegion = uint(findLSB(opaqueMask)) + wordIndex * 32u;
                opaqueMask &= opaqueMask - 1u;
                ivec3 blockInSubRegionCoord = ivec3(morton3D_6bDecode(blockInSubRegion));
                ivec3 blockInBrick = subRegionBlockBase + blockInSubRegionCoord;
                ivec3 gridBlockPos = brickBlockBase + blockInBrick;
                ivec3 worldBlockPos = gridOrigin + gridBlockPos;

                uint faceBits = 0u;
                for (uint faceId = 0u; faceId < 6u; faceId++) {
                    uint axis = faceId >> 1u;
                    bool negative = (faceId & 1u) != 0u;
                    int axisCoord = blockInSubRegionCoord[axis];
                    bool localNeighbor = negative ? axisCoord > 0 : axisCoord < 3;
                    bool neighborOpaque;
                    if (localNeighbor) {
                        uint axisBit = 1u << axis;
                        uint carryBit = axisBit << 3u;
                        bool carry = ((uint(axisCoord) & 1u) != 0u) != negative;
                        uint neighborBit = blockInSubRegion ^ (axisBit | (carry ? carryBit : 0u));
                        neighborOpaque = neighborBit < 32u
                            ? (opaqueLow & (1u << neighborBit)) != 0u
                            : (opaqueHigh & (1u << (neighborBit - 32u))) != 0u;
                    } else {
                        int subRegionAxisCoord = subRegionCoord[axis];
                        bool brickNeighbor = negative ? subRegionAxisCoord > 0 : subRegionAxisCoord < 3;
                        if (brickNeighbor) {
                            uint axisBit = 1u << axis;
                            uint carryBit = axisBit << 3u;
                            bool carry = ((uint(subRegionAxisCoord) & 1u) != 0u) != negative;
                            uint neighborSubRegion = subRegion ^ (axisBit | (carry ? carryBit : 0u));
                            uint neighborBit = blockInSubRegion ^ (axisBit | carryBit);
                            uvec2 neighborMask = brickOpaqueMasks[groupBrick * 64u + neighborSubRegion];
                            neighborOpaque = neighborBit < 32u
                                ? (neighborMask.x & (1u << neighborBit)) != 0u
                                : (neighborMask.y & (1u << (neighborBit - 32u))) != 0u;
                        } else {
                            uint neighborAllocID = adjacentBrickAllocIDs[groupBrick * 6u + faceId];
                            uint blockMorton = subRegion * 64u + blockInSubRegion;
                            uint neighborBlockMorton = blockMorton ^ (0x249u << axis);
                            neighborOpaque = neighborAllocID != VOXEL_UNALLOCATED &&
                                voxel_isGIOpaqueMaterial(voxel_loadMaterialData(neighborAllocID, neighborBlockMorton));
                        }
                    }
                    if (!neighborOpaque) {
                        faceBits |= 1u << faceId;
                    }
                }
                if (faceBits != 0u) {
                    vec3 ownerBlockCenter = vec3(worldBlockPos) + vec3(0.5);
                    for (uint level = 0u; level < 2u; level++) {
                        rc_markPendingVisibleFace(level, ownerBlockCenter, faceBits);
                    }
                    coarseFaceBits |= faceBits;
                }
            }
        }

        if (coarseFaceBits != 0u) {
            vec3 subRegionCenter = vec3(gridOrigin + brickBlockBase + subRegionBlockBase) + vec3(0.5);
            for (uint level = 2u; level < RC_CLIP_LEVELS; level++) {
                rc_markPendingVisibleFace(level, subRegionCenter, coarseFaceBits);
            }
        }

        // Parallel reduction: compute bit(s) this thread contributes to the root mask
        bool subRegionNonEmpty = (leafLow | leafHigh) != 0u;
        uint bitLo = 0u, bitHi = 0u;
        if (subRegionNonEmpty) {
            uint leafIdx = uint(VOXEL_TREE_OFFSET_L1) + brickMorton * 64u + subRegion;
            voxel_tree[leafIdx] = uvec2(leafLow, leafHigh);
            if (subRegion < 32u) {
                bitLo = 1u << subRegion;
            } else {
                bitHi = 1u << (subRegion - 32u);
            }
        }

        // Reduce within subgroup using subgroupOr
        uint reducedLo = subgroupOr(bitLo);
        uint reducedHi = subgroupOr(bitHi);

        // One thread per subgroup writes result to temporary shared storage
        if (subgroupElect()) {
            atomicOr(rootMaskLo[groupBrick], reducedLo);
            atomicOr(rootMaskHi[groupBrick], reducedHi);
        }
    }

    barrier();

    // Allocated roots are disjoint from the unallocated roots cleared above.
    if (subRegion == 0u && allocID != VOXEL_UNALLOCATED) {
        uint rootIdx = uint(VOXEL_TREE_OFFSET_L2) + brickMorton;
        voxel_tree[rootIdx] = uvec2(rootMaskLo[groupBrick], rootMaskHi[groupBrick]);
    }
}
