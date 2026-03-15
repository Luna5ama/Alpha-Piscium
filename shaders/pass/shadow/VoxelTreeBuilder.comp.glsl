// 64-Tree Builder – runs after the shadow pass (shadowcomp2).
//
// Builds a 2-level 64-bit bitmask tree for each allocated brick:
//   Level 0 (root, 1 uint64_t): bit j = 1 if 4^3 sub-region j has any solid block.
//   Level 1 (leaves, 64 uint64_t): leaf j bit k = 1 if block k in sub-region j is solid.
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

shared uint rootMaskLo;
shared uint rootMaskHi;

void main() {
    uint brickMorton = uint(gl_WorkGroupID.x);   // 0..4095
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

    // Write leaf node (each sub-region is handled by exactly one thread – no race)
    uint leafIdx = voxel_treeLeafIndex(allocID, subRegion);
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

    // Single writer per brick for the root node to avoid 64-bit atomics.
    if (gl_LocalInvocationIndex == 0u) {
        uint rootIdx = voxel_treeRootIndex(allocID);
        voxel_tree[rootIdx] = uvec2(rootMaskLo, rootMaskHi);
    }
}

