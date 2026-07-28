// 64-Tree Builder – runs after the shadow pass (shadowcomp2).
//
// Builds the bottom 2 levels of the dense 64-tree for each allocated brick:
//   Level 1 (leaf, 64 uvec2 per brick): bit k = 1 if block k in sub-region is solid.
//   Level 2 (brick root, 1 uvec2 per brick): bit j = 1 if sub-region j has any solid block.
//
// Tree nodes are written to the dense layout (indexed by brickMorton, not allocID).
// Material data is read and updated via allocID (sparse pool).
//
// Dispatch: one workgroup per 4 bricks in the VOXEL_GRID_SIZE^3 grid.
// Threads per workgroup: 256 (64 per brick × 4 bricks, one thread per sub-region).
//
// This pass owns L1 and L2 for every brick — no pre-clearing of SSBO 8 needed:
//   Allocated bricks  → L1 and L2 are computed and written.
//   Unallocated bricks → L1 sub-region slot is zeroed; L2 is written as 0
//                        (rootMaskLo/Hi stay 0 since no atomics fire).
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
#define VOXEL_MATERIAL_DATA_MODIFIER restrict buffer
#define VOXEL_TREE_DATA_MODIFIER buffer
#define RC_DATA_MODIFIER restrict buffer
#include "/techniques/voxel/Voxelization.glsl"
#include "/techniques/gi/RadianceCache.glsl"

layout(std430, binding = 8) VOXEL_TREE_DATA_MODIFIER VoxelTreeData {
    uvec2 voxel_tree[];       // VOXEL_TREE_TOTAL uvec2 entries
};

layout(local_size_x = 256) in;
// One workgroup per 4 bricks in the VOXEL_GRID_SIZE^3 grid
#if VOXEL_GRID_SIZE == 64
const ivec3 workGroups = ivec3(65536, 1, 1);
#elif VOXEL_GRID_SIZE == 32
const ivec3 workGroups = ivec3(8192, 1, 1);
#else
const ivec3 workGroups = ivec3(1024, 1, 1);
#endif

shared uint rootMaskLo[4];
shared uint rootMaskHi[4];

bool voxel_isGIOpaqueMaterial(uint materialID) {
    return materialID != 0u && materialID != MATERIAL_ID_WATER;
}

uint voxel_loadMaterialID(uint allocID, uint blockMorton) {
    uint packedIndex = allocID * 1024u + (blockMorton >> 2u);
    uvec4 packedMaterialData = voxel_materials_v4[packedIndex];
    uint lane = blockMorton & 3u;
    if (lane == 0u) return packedMaterialData.x & 0xFFFFu;
    if (lane == 1u) return packedMaterialData.y & 0xFFFFu;
    if (lane == 2u) return packedMaterialData.z & 0xFFFFu;
    return packedMaterialData.w & 0xFFFFu;
}

bool voxel_opaqueAtGridBlock(ivec3 gridBlockPos) {
    int gridExtent = VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE;
    if (any(lessThan(gridBlockPos, ivec3(0))) || any(greaterThanEqual(gridBlockPos, ivec3(gridExtent)))) {
        return false;
    }

    ivec3 brickCoord = gridBlockPos >> 4;
    uint brickMorton = voxel_brickMorton(brickCoord);
    uint allocID = voxel_brickAllocID[brickMorton];
    if (allocID == VOXEL_UNALLOCATED) {
        return false;
    }

    uint blockMorton = voxel_blockMorton(gridBlockPos & 15);
    return voxel_isGIOpaqueMaterial(voxel_loadMaterialID(allocID, blockMorton));
}

void rc_markPendingVisibleFace(ivec3 worldBlockPos, uint faceBits) {
    vec3 ownerBlockCenter = vec3(worldBlockPos) + vec3(0.5);
    for (uint level = 0u; level < RC_CLIP_LEVELS; level++) {
        ivec3 worldCellCoord = rc_worldCellCoord(ownerBlockCenter, level);
        if (rc_worldCellInCurrentClip(level, worldCellCoord)) {
            uint entryIndex = rc_entryIndex(level, worldCellCoord);
            uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), entryIndex);
            atomicOr(rc_indirection[bufferIndex].w, rc_entryMetaPendingFaceBits(faceBits));
        }
    }
}

void main() {
    uint localID    = gl_LocalInvocationID.x;
    uint groupBrick = localID >> 6u;        // 0..3 – which of the 4 bricks in this workgroup
    uint subRegion  = localID & 63u;        // 0..63 – sub-region index within the brick
    uint brickMorton = gl_WorkGroupID.x * 4u + groupBrick;

    if (subRegion == 0u) {
        rootMaskLo[groupBrick] = 0u;
        rootMaskHi[groupBrick] = 0u;
    }
    barrier();

    uint allocID = voxel_brickAllocID[brickMorton];

    if (allocID != VOXEL_UNALLOCATED) {
        // Within this sub-region, blockMorton = subRegion * 64 + blockInSr.
        // All 64 entries are contiguous, so read 4-at-a-time as uvec4.
        uint baseIdx = allocID * 1024u + subRegion * 16u;

        uint leafLow  = 0u;
        uint leafHigh = 0u;

        // First 8 uvec4 reads → 32 blocks → leafLow (bits 0..31)
        for (uint i = 0u; i < 8u; i++) {
            uvec4 mats = voxel_materials_v4[baseIdx + i] & uvec4(0xFFFFu);
            uvec4 bits4 = uvec4(notEqual(mats, uvec4(0u))) << uvec4(0u, 1u, 2u, 3u);
            uint bits = bits4.x + bits4.y + bits4.z + bits4.w;
            leafLow |= bits << (i * 4u);
        }

        // Next 8 uvec4 reads → 32 blocks → leafHigh (bits 0..31)
        for (uint i = 0u; i < 8u; i++) {
            uvec4 mats = voxel_materials_v4[baseIdx + 8u + i] & uvec4(0xFFFFu);
            uvec4 bits4 = uvec4(notEqual(mats, uvec4(0u))) << uvec4(0u, 1u, 2u, 3u);
            uint bits = bits4.x + bits4.y + bits4.z + bits4.w;
            leafHigh |= bits << (i * 4u);
        }

        // Write Level-1 leaf node
        uint leafIdx = uint(VOXEL_TREE_OFFSET_L1) + brickMorton * 64u + subRegion;
        voxel_tree[leafIdx] = uvec2(leafLow, leafHigh);

        ivec3 cameraBrick = cameraPositionInt >> 4;
        ivec3 gridOrigin = (cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4;
        uvec3 brickCoordU = morton3D_30bDecode(brickMorton);
        ivec3 brickBlockBase = ivec3(brickCoordU << 4u);

        for (uint i = 0u; i < 16u; i++) {
            uvec4 mats = voxel_materials_v4[baseIdx + i];
            for (uint lane = 0u; lane < 4u; lane++) {
                uint blockInSubRegion = i * 4u + lane;
                uint materialID = mats[lane] & 0xFFFFu;
                uint faceBits = 0u;
                if (voxel_isGIOpaqueMaterial(materialID)) {
                    uint blockMorton = subRegion * 64u + blockInSubRegion;
                    ivec3 blockInBrick = ivec3(morton3D_12bDecode(blockMorton));
                    ivec3 gridBlockPos = brickBlockBase + blockInBrick;
                    ivec3 worldBlockPos = gridOrigin + gridBlockPos;

                    for (uint faceId = 0u; faceId < 6u; faceId++) {
                        ivec3 faceNormalI = rc_faceNormalI(faceId);
                        ivec3 neighborGridBlock = gridBlockPos + faceNormalI;
                        if (!voxel_opaqueAtGridBlock(neighborGridBlock)) {
                            faceBits |= rc_faceBit(faceId);
                        }
                    }
                    if (faceBits != 0u) {
                        rc_markPendingVisibleFace(worldBlockPos, faceBits);
                    }
                }
                mats[lane] = (materialID & 0xFFFFu) | (faceBits << 16u);
            }
            voxel_materials_v4[baseIdx + i] = mats;
        }

        // Parallel reduction: compute bit(s) this thread contributes to the root mask
        bool subRegionNonEmpty = (leafLow | leafHigh) != 0u;
        uint bitLo = 0u, bitHi = 0u;
        if (subRegionNonEmpty) {
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
    } else {
        // Unallocated brick — zero out this sub-region's L1 leaf slot
        voxel_tree[uint(VOXEL_TREE_OFFSET_L1) + brickMorton * 64u + subRegion] = uvec2(0u);
    }

    barrier();

    // Always write L2 (computed mask for allocated bricks; 0 for unallocated
    // since rootMaskLo/Hi was initialized to 0 and no atomics fired).
    if (subRegion == 0u) {
        uint rootIdx = uint(VOXEL_TREE_OFFSET_L2) + brickMorton;
        voxel_tree[rootIdx] = uvec2(rootMaskLo[groupBrick], rootMaskHi[groupBrick]);
    }
}
