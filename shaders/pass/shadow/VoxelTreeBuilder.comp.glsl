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
// Each workgroup uses 64 lanes per brick for four bricks.
//
// This pass owns L2 for every brick and writes L1 only for non-empty sub-regions.
// An empty L2 bit leaves its backing L1 slot unspecified.
// Upper levels (L3–L5) are always written unconditionally by the propagator passes,
// so they are self-clearing when all children below them are zero.
//
// Block Morton contiguity: within sub-region S, blockMorton = S*64 + blockInSr.
// All 64 blocks are contiguous in voxel_materials[], so we read 4 at a time as uvec4.
// Base index (allocID*4096 + subRegion*64) is always divisible by 4.
//
// Lanes cooperate on adjacent material quads. A subgroup ballot packs the final L2 bits.

#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_basic : enable

#extension GL_KHR_shader_subgroup_shuffle : enable
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
shared uvec2 occupiedMasks[256];

bool voxel_isGIOpaqueMaterial(uint materialData) {
    return materialData >= 4u;
}

uint voxel_materialNibble(bvec4 occupied) {
    uvec4 bits = uvec4(occupied) << uvec4(0u, 1u, 2u, 3u);
    return bits.x | bits.y | bits.z | bits.w;
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

uvec2 voxel_shiftMaskRight(uvec2 mask, uint count) {
    if (count >= 32u) return uvec2(mask.y >> (count - 32u), 0u);
    return uvec2((mask.x >> count) | (mask.y << (32u - count)), mask.y >> count);
}

uvec2 voxel_shiftMaskLeft(uvec2 mask, uint count) {
    if (count >= 32u) return uvec2(0u, mask.x << (count - 32u));
    return uvec2(mask.x << count, (mask.y << count) | (mask.x >> (32u - count)));
}

// Shift neighbor occupancy onto each owner's Morton bit. Crossing a low coordinate bit
// advances by one Morton bit stride; carrying into the high coordinate bit advances by seven.
uvec2 voxel_openFaces(uvec2 occupied, uint face, uint groupBrick, uint subRegion, ivec3 subRegionCoord) {
    uint axis = face >> 1u;
    uint stride = 1u << axis;
    bool negative = (face & 1u) != 0u;
    uvec2 low = uvec2(axis == 0u ? 0xAAAAAAAAu : (axis == 1u ? 0xCCCCCCCCu : 0xF0F0F0F0u));
    uvec2 high = axis == 0u ? uvec2(0xFF00FF00u) : (axis == 1u ? uvec2(0xFFFF0000u) : uvec2(0u, ~0u));
    uvec2 edge = negative ? ~(low | high) : low & high;
    uvec2 neighbors = negative
        ? (voxel_shiftMaskLeft(occupied, stride) & low) | (voxel_shiftMaskLeft(occupied, 7u * stride) & ~low & high)
        : (voxel_shiftMaskRight(occupied, stride) & ~low) | (voxel_shiftMaskRight(occupied, 7u * stride) & low & ~high);
    int coord = subRegionCoord[axis];
    if (negative ? coord > 0 : coord < 3) {
        bool carry = ((uint(coord) & 1u) != 0u) != negative;
        uint adjacent = subRegion ^ (stride | (carry ? stride << 3u : 0u));
        uvec2 other = brickOpaqueMasks[groupBrick * 64u + adjacent];
        neighbors |= edge & (negative ? voxel_shiftMaskRight(other, 9u * stride) : voxel_shiftMaskLeft(other, 9u * stride));
    } else {
        uint allocID = adjacentBrickAllocIDs[groupBrick * 6u + face];
        if (allocID != VOXEL_UNALLOCATED) {
            uvec2 boundary = occupied & edge;
            for (uint word = 0u; word < 2u; ++word) {
                uint bits = boundary[word];
                while (bits != 0u) {
                    uint bit = uint(findLSB(bits));
                    bits &= bits - 1u;
                    uint morton = subRegion * 64u + word * 32u + bit;
                    if (voxel_isGIOpaqueMaterial(voxel_loadMaterialData(allocID, morton ^ (0x249u << axis)))) {
                        neighbors[word] |= 1u << bit;
                    }
                }
            }
        }
    }
    return occupied & ~neighbors;
}

void rc_markPendingVisibleFace(uint level, vec3 ownerBlockCenter, uint faceBits) {
    ivec3 worldCellCoord = rc_worldCellCoord(ownerBlockCenter, level);
    if (rc_worldCellInCurrentClip(level, worldCellCoord)) {
        uint entryIndex = rc_entryIndex(level, worldCellCoord);
        rc_markEntryListed(rc_currentSide(), entryIndex, rc_entryMetaPendingFaceBits(faceBits));
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

    uint localID = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uint groupBrick = localID >> 6u;
    uint subRegion = localID & 63u;
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
    #ifdef SETTING_RC_ENABLE
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
    #endif

    if (allocID != VOXEL_UNALLOCATED) {
        // Each lane supplies distinct nibbles of the same leaf mask.
        uint lanes = min(gl_SubgroupSize, 4u);
        uint lane = subRegion & (lanes - 1u);
        for (uint tile = 0u; tile < lanes; ++tile) {
            uint leaf = tile * (64u / lanes) + subRegion / lanes;
            uint baseIndex = allocID * 1024u + leaf * 16u;
            uvec4 masks = uvec4(0u);
            for (uint i = lane; i < 8u; i += lanes) {
                uvec4 a = voxel_materials_v4[baseIndex + i];
                uvec4 b = voxel_materials_v4[baseIndex + i + 8u];
                uvec4 nibble = uvec4(voxel_materialNibble(notEqual(a, uvec4(0u))),
                    voxel_materialNibble(notEqual(b, uvec4(0u))), 0u, 0u);
                #ifdef SETTING_RC_ENABLE
                nibble.z = voxel_materialNibble(greaterThanEqual(a, uvec4(4u)));
                nibble.w = voxel_materialNibble(greaterThanEqual(b, uvec4(4u)));
                #endif
                masks |= nibble << (i * 4u);
            }
            for (uint shift = 1u; shift < lanes; shift *= 2u) {
                masks |= subgroupShuffle(masks, gl_SubgroupInvocationID ^ shift);
            }
            if (lane == 0u) {
                occupiedMasks[groupBrick * 64u + leaf] = masks.xy;
                #ifdef SETTING_RC_ENABLE
                brickOpaqueMasks[groupBrick * 64u + leaf] = masks.zw;
                #endif
            }
        }
    }
    barrier();
    if (allocID != VOXEL_UNALLOCATED) {
        uvec2 leafMask = occupiedMasks[localID];
        #ifdef SETTING_RC_ENABLE
        uint opaqueLow = brickOpaqueMasks[localID].x;
        uint opaqueHigh = brickOpaqueMasks[localID].y;
        #endif
        #ifdef SETTING_RC_ENABLE
        if ((opaqueLow | opaqueHigh) != 0u) {
            ivec3 cameraBrick = cameraPositionInt >> 4;
            ivec3 gridOrigin = (cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4;
            uvec3 brickCoordU = morton3D_30bDecode(brickMorton);
            ivec3 brickBlockBase = ivec3(brickCoordU << 4u);
            ivec3 subRegionCoord = ivec3(morton3D_6bDecode(subRegion));
            ivec3 subRegionBlockBase = subRegionCoord << 2;
            uvec2 occupied = uvec2(opaqueLow, opaqueHigh);
            uvec2 posX = voxel_openFaces(occupied, 0u, groupBrick, subRegion, subRegionCoord);
            uvec2 negX = voxel_openFaces(occupied, 1u, groupBrick, subRegion, subRegionCoord);
            uvec2 posY = voxel_openFaces(occupied, 2u, groupBrick, subRegion, subRegionCoord);
            uvec2 negY = voxel_openFaces(occupied, 3u, groupBrick, subRegion, subRegionCoord);
            uvec2 posZ = voxel_openFaces(occupied, 4u, groupBrick, subRegion, subRegionCoord);
            uvec2 negZ = voxel_openFaces(occupied, 5u, groupBrick, subRegion, subRegionCoord);
            uint coarseFaceBits = uint(any(notEqual(posX, uvec2(0u)))) | (uint(any(notEqual(negX, uvec2(0u)))) << 1u)
                | (uint(any(notEqual(posY, uvec2(0u)))) << 2u) | (uint(any(notEqual(negY, uvec2(0u)))) << 3u)
                | (uint(any(notEqual(posZ, uvec2(0u)))) << 4u) | (uint(any(notEqual(negZ, uvec2(0u)))) << 5u);
            vec3 firstCenter = vec3(gridOrigin + brickBlockBase + subRegionBlockBase) + vec3(0.5);
            vec3 lastCenter = vec3(gridOrigin + brickBlockBase + subRegionBlockBase + 3) + vec3(0.5);
            ivec3 cameraCell = rc_worldCellCoord(cameraPosition, 1u);
            ivec3 closestCell = clamp(cameraCell, rc_worldCellCoord(firstCenter, 1u), rc_worldCellCoord(lastCenter, 1u));
            // Only the two finest clips need individual block faces.
            if (rc_worldCellInCurrentClip(1u, closestCell)) {
                uvec2 exposed = posX | negX | posY | negY | posZ | negZ;
                for (uint wordIndex = 0u; wordIndex < 2u; ++wordIndex) {
                    uint visible = exposed[wordIndex];
                    while (visible != 0u) {
                        uint bit = uint(findLSB(visible));
                        visible &= visible - 1u;
                        uint blockInSubRegion = bit + wordIndex * 32u;
                        ivec3 blockInSubRegionCoord = ivec3(morton3D_6bDecode(blockInSubRegion));
                        ivec3 blockInBrick = subRegionBlockBase + blockInSubRegionCoord;
                        ivec3 gridBlockPos = brickBlockBase + blockInBrick;
                        ivec3 worldBlockPos = gridOrigin + gridBlockPos;

                        uint faceBits = ((posX[wordIndex] >> bit) & 1u) | (((negX[wordIndex] >> bit) & 1u) << 1u)
                            | (((posY[wordIndex] >> bit) & 1u) << 2u) | (((negY[wordIndex] >> bit) & 1u) << 3u)
                            | (((posZ[wordIndex] >> bit) & 1u) << 4u) | (((negZ[wordIndex] >> bit) & 1u) << 5u);
                        vec3 ownerBlockCenter = vec3(worldBlockPos) + vec3(0.5);
                        for (uint level = 0u; level < 2u; level++) {
                            rc_markPendingVisibleFace(level, ownerBlockCenter, faceBits);
                        }
                    }
                }
            }

            if (coarseFaceBits != 0u) {
                for (uint level = 2u; level < RC_CLIP_LEVELS; level++) {
                    rc_markPendingVisibleFace(level, firstCenter, coarseFaceBits);
                }
            }
        }
        #endif

        bool nonEmpty = (leafMask.x | leafMask.y) != 0u;
        if (nonEmpty) {
            uint leafIndex = uint(VOXEL_TREE_OFFSET_L1) + brickMorton * 64u + subRegion;
            voxel_tree[leafIndex] = leafMask;
        }

        uvec4 votes = subgroupBallot(nonEmpty);
        if ((subRegion & (min(gl_SubgroupSize, 32u) - 1u)) == 0u) {
            uint bits = votes[gl_SubgroupInvocationID >> 5u] << (subRegion & 31u);
            if (subRegion < 32u) atomicOr(rootMaskLo[groupBrick], bits);
            else atomicOr(rootMaskHi[groupBrick], bits);
        }
    }

    barrier();

    // Allocated roots are disjoint from the unallocated roots cleared above.
    if (subRegion == 0u && allocID != VOXEL_UNALLOCATED) {
        uint rootIdx = uint(VOXEL_TREE_OFFSET_L2) + brickMorton;
        voxel_tree[rootIdx] = uvec2(rootMaskLo[groupBrick], rootMaskHi[groupBrick]);
    }
}
