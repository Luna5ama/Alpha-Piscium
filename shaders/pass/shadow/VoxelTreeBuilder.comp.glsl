// 64-Tree Builder – runs after the shadow pass (shadowcomp2).
//
// Builds a 2-level 64-bit bitmask tree for each allocated brick:
//   Level 0 (root, 1 uvec2): bit j = 1 if 4^3 sub-region j has any solid block.
//   Level 1 (leaves, 64 uvec2): leaf j bit k = 1 if block k in sub-region j is solid.
//
// Dispatch: one workgroup per brick slot in the grid (4096 total).
// Threads per workgroup: 64 (one per 4^3 sub-region within the brick).
// SSBO 8 must have been cleared to 0 before this pass (done in begin5_c).

#define VOXEL_BRICK_DATA_MODIFIER restrict readonly buffer
#define VOXEL_MATERIAL_DATA_MODIFIER restrict readonly buffer
#define VOXEL_TREE_DATA_MODIFIER buffer
#include "/techniques/Voxelization.glsl"

layout(local_size_x = 64) in;
// One workgroup per brick in the 16^3 grid (4096 bricks total)
const ivec3 workGroups = ivec3(4096, 1, 1);

void main() {
    uint brickMorton = uint(gl_WorkGroupID.x);   // 0..4095
    uint subRegion   = uint(gl_LocalInvocationID.x); // 0..63

    uint allocID = voxel_brickAllocID[brickMorton];
    if (allocID == VOXEL_UNALLOCATED) return;

    // Decode sub-region Morton to 3D coordinate within the 4^3 sub-region grid
    uvec3 srCoord = morton3D_decode(subRegion); // 0..3 per axis

    // Build the 64-bit leaf mask for this sub-region
    uint leafLow  = 0u;
    uint leafHigh = 0u;
    bool subRegionNonEmpty = false;

    for (uint blockInSr = 0u; blockInSr < 64u; blockInSr++) {
        // 3D coord of this block within the sub-region (0..3 per axis)
        uvec3 bInSrCoord = morton3D_decode(blockInSr);
        // Full block coord in brick (0..15 per axis)
        uvec3 blockCoord = srCoord * 4u + bInSrCoord;
        uint  blockMorton = morton3D_encode(blockCoord);

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

    // Write leaf node (each sub-region is handled by exactly one thread – no race)
    uint leafIdx = voxel_treeLeafIndex(allocID, subRegion);
    voxel_tree[leafIdx] = uvec2(leafLow, leafHigh);

    // Set the corresponding bit in the root node (atomic, multiple threads write)
    if (subRegionNonEmpty) {
        uint rootIdx = voxel_treeRootIndex(allocID);
        if (subRegion < 32u) {
            atomicOr(voxel_tree[rootIdx].x, 1u << subRegion);
        } else {
            atomicOr(voxel_tree[rootIdx].y, 1u << (subRegion - 32u));
        }
    }
}

