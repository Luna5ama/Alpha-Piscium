// 64-Tree Builder – runs after the shadow pass (shadowcomp2).
//
// Builds the bottom 2 levels of the dense 64-tree for each allocated brick:
//   Level 1 (leaf, 64 uvec2 per brick): bit k = 1 if block k in sub-region is solid.
//   Level 2 (brick root, 1 uvec2 per brick): bit j = 1 if sub-region j has any solid block.
//
// Tree nodes are written to the dense layout (indexed by brickMorton, not allocID).
// Material data is still read via allocID (sparse pool).
//
// Dispatch: one workgroup per brick in the VOXEL_GRID_SIZE^3 grid.
// Threads per workgroup: 64 (one per 4^3 sub-region within the brick).
// SSBO 8 must have been cleared to 0 before this pass (done in begin5_a).

#define VOXEL_BRICK_DATA_MODIFIER restrict readonly buffer
#define VOXEL_MATERIAL_DATA_MODIFIER restrict readonly buffer
#define VOXEL_TREE_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 64) in;
// One workgroup per brick in the VOXEL_GRID_SIZE^3 grid
#if VOXEL_GRID_SIZE == 64
const ivec3 workGroups = ivec3(262144, 1, 1);
#elif VOXEL_GRID_SIZE == 32
const ivec3 workGroups = ivec3(32768, 1, 1);
#else
const ivec3 workGroups = ivec3(4096, 1, 1);
#endif

shared uint rootMaskLo;
shared uint rootMaskHi;

void main() {
    uint brickMorton = uint(gl_WorkGroupID.x);   // 0..VOXEL_GRID_BRICKS-1
    uint subRegion   = uint(gl_LocalInvocationID.x); // 0..63

    if (gl_LocalInvocationIndex == 0u) {
        rootMaskLo = 0u;
        rootMaskHi = 0u;
    }
    barrier();

    uint allocID = voxel_brickAllocID[brickMorton];
    if (allocID == VOXEL_UNALLOCATED) return;

    // Decode sub-region Morton to 3D coordinate within the 4^3 sub-region grid
    uvec3 srCoord = morton3D_30bDecode(subRegion); // 0..3 per axis

    // Build the 64-bit leaf mask for this sub-region
    uint leafLow  = 0u;
    uint leafHigh = 0u;
    bool subRegionNonEmpty = false;

    for (uint blockInSr = 0u; blockInSr < 64u; blockInSr++) {
        // 3D coord of this block within the sub-region (0..3 per axis)
        uvec3 bInSrCoord = morton3D_30bDecode(blockInSr);
        // Full block coord in brick (0..15 per axis)
        uvec3 blockCoord = srCoord * 4u + bInSrCoord;
        uint  blockMorton = morton3D_30bEncode(blockCoord);

        uint material = voxel_materials[allocID * 4096u + blockMorton];
        if (material != 0u) {
            subRegionNonEmpty = true;
            if (blockInSr < 32u) {
                leafLow  |= (1u << blockInSr);
            } else {
                leafHigh |= (1u << (blockInSr - 32u));
            }
        }
    }

    // Write leaf node to dense tree layout: Level 1
    // Index = VOXEL_TREE_OFFSET_L1 + brickMorton * 64 + subRegion
    uint leafIdx = uint(VOXEL_TREE_OFFSET_L1) + brickMorton * 64u + subRegion;
    voxel_tree[leafIdx] = uvec2(leafLow, leafHigh);

    // Set the corresponding bit in the shared root mask (32-bit atomics only).
    if (subRegionNonEmpty) {
        if (subRegion < 32u) {
            atomicOr(rootMaskLo, 1u << subRegion);
        } else {
            atomicOr(rootMaskHi, 1u << (subRegion - 32u));
        }
    }

    barrier();

    // Single writer per brick for the Level 2 (brick root) node.
    // Index = VOXEL_TREE_OFFSET_L2 + brickMorton
    if (gl_LocalInvocationIndex == 0u) {
        uint rootIdx = uint(VOXEL_TREE_OFFSET_L2) + brickMorton;
        voxel_tree[rootIdx] = uvec2(rootMaskLo, rootMaskHi);
    }
}
