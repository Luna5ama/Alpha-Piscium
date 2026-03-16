// 64-Tree Propagator — Pass 2 of 2 for grid=64 (shadowcomp5).
// Disabled for grid=16/32; those use a single pass (shadowcomp3).
//
// Runs after VoxelTreePropagatorLower (shadowcomp4) has built Level 3.
// Builds Level 4 (64 nodes) from Level 3, then Level 5 (root) from Level 4,
// all within a single workgroup with barriers between levels.
//
// One workgroup × 64 threads: thread tid handles L4 node tid (0..63),
// each reading its 64 L3 children sequentially.  Thread 0 then builds L5.

#define VOXEL_TREE_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 64) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    uint tid = gl_LocalInvocationID.x;  // L4 node index (0..63)

    // ---- Build Level 4 from Level 3 ----
    // L4 has 64 nodes; each aggregates 64 L3 children.
    {
        uint lo = 0u, hi = 0u;
        uint childBase = uint(VOXEL_TREE_OFFSET_L3) + tid * 64u;
        for (uint c = 0u; c < 64u; c++) {
            uvec2 child = voxel_tree[childBase + c];
            if ((child.x | child.y) != 0u) {
                if (c < 32u) lo |= (1u << c);
                else         hi |= (1u << (c - 32u));
            }
        }
        voxel_tree[uint(VOXEL_TREE_OFFSET_L4) + tid] = uvec2(lo, hi);
    }

    memoryBarrierBuffer();
    barrier();

    // ---- Build Level 5 (root) from Level 4 ----
    // L5 has 1 node; all 64 L4 children are valid for grid=64.
    if (tid == 0u) {
        uint lo = 0u, hi = 0u;
        for (uint c = 0u; c < 64u; c++) {
            uvec2 child = voxel_tree[uint(VOXEL_TREE_OFFSET_L4) + c];
            if ((child.x | child.y) != 0u) {
                if (c < 32u) lo |= (1u << c);
                else         hi |= (1u << (c - 32u));
            }
        }
        voxel_tree[uint(VOXEL_TREE_OFFSET_L5)] = uvec2(lo, hi);
    }
}
