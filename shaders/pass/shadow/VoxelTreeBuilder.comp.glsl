// 64-Tree Builder – runs after the shadow pass (shadowcomp2).
//
// Builds the bottom 2 levels of the dense 64-tree for each allocated brick:
//   Level 1 (leaf, 64 uvec2 per brick): bit k = 1 if block k in sub-region is solid.
//   Level 2 (brick root, 1 uvec2 per brick): bit j = 1 if sub-region j has any solid block.
//
// Tree nodes are written to the dense layout (indexed by brickMorton, not allocID).
// Material data is still read via allocID (sparse pool).
//
// Dispatch: one workgroup per 4 bricks in the VOXEL_GRID_SIZE^3 grid.
// Threads per workgroup: 256 (64 per brick × 4 bricks, one thread per sub-region).
// SSBO 8 must have been cleared to 0 before this pass (done in begin5_a).
//
// Block Morton contiguity: within sub-region S, blockMorton = S*64 + blockInSr.
// All 64 blocks are contiguous in voxel_materials[], so we read 4 at a time as uvec4.
// Base index (allocID*4096 + subRegion*64) is always divisible by 4.
//
// Root mask accumulation uses parallel reduction (subgroupOr) instead of atomicOr.
// Two subgroups per brick → results stored in tempMaskLo/Hi[gl_SubgroupID] →
// thread 0 of each brick ORs the pair together.

#extension GL_KHR_shader_subgroup_arithmetic : enable

#define VOXEL_BRICK_DATA_MODIFIER restrict readonly buffer
#define VOXEL_MATERIAL_VEC4
#define VOXEL_MATERIAL_DATA_MODIFIER restrict readonly buffer
#define VOXEL_TREE_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

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
            uvec4 mats = voxel_materials_v4[baseIdx + i];
            uvec4 bits4 = uvec4(notEqual(mats, uvec4(0u))) << uvec4(0u, 1u, 2u, 3u);
            uint bits = bits4.x + bits4.y + bits4.z + bits4.w;
            leafLow |= bits << (i * 4u);
        }

        // Next 8 uvec4 reads → 32 blocks → leafHigh (bits 0..31)
        for (uint i = 0u; i < 8u; i++) {
            uvec4 mats = voxel_materials_v4[baseIdx + 8u + i];
            uvec4 bits4 = uvec4(notEqual(mats, uvec4(0u))) << uvec4(0u, 1u, 2u, 3u);
            uint bits = bits4.x + bits4.y + bits4.z + bits4.w;
            leafHigh |= bits << (i * 4u);
        }

        // Write Level-1 leaf node
        uint leafIdx = uint(VOXEL_TREE_OFFSET_L1) + brickMorton * 64u + subRegion;
        voxel_tree[leafIdx] = uvec2(leafLow, leafHigh);

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
    }

    barrier();

    // Write final tree node (brick root, Level 2)
    if (subRegion == 0u && allocID != VOXEL_UNALLOCATED) {
        uint rootIdx = uint(VOXEL_TREE_OFFSET_L2) + brickMorton;
        voxel_tree[rootIdx] = uvec2(rootMaskLo[groupBrick], rootMaskHi[groupBrick]);
    }
}
