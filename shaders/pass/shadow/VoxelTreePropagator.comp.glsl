// 64-Tree Propagator — single pass for grid=16/32 (shadowcomp3).
// Disabled for grid=64; that case uses two passes (shadowcomp4 + shadowcomp5).
//
// Builds L3 → L4 → L5 in one dispatch using 1024 threads and shared barriers.

#define VOXEL_TREE_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 1024) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    uint tid = gl_LocalInvocationID.x;

    // ---- Build Level 3 from Level 2 ----
    #if VOXEL_GRID_SIZE == 16
    #define L3_COUNT 64
    #elif VOXEL_GRID_SIZE == 32
    #define L3_COUNT 512
    #endif

    if (tid < uint(L3_COUNT)) {
        uint parentIdx = uint(VOXEL_TREE_OFFSET_L3) + tid;
        uint childBase = uint(VOXEL_TREE_OFFSET_L2) + tid * 64u;
        uint lo = 0u, hi = 0u;
        for (uint c = 0u; c < 64u; c++) {
            uvec2 child = voxel_tree[childBase + c];
            if ((child.x | child.y) != 0u) {
                if (c < 32u) lo |= (1u << c);
                else         hi |= (1u << (c - 32u));
            }
        }
        voxel_tree[parentIdx] = uvec2(lo, hi);
    }

    memoryBarrierBuffer();
    barrier();

    // ---- Build Level 4 from Level 3 ----
    #if VOXEL_GRID_SIZE == 16
    #define L4_COUNT 1
    #elif VOXEL_GRID_SIZE == 32
    #define L4_COUNT 8
    #endif

    if (tid < uint(L4_COUNT)) {
        uint parentIdx = uint(VOXEL_TREE_OFFSET_L4) + tid;
        uint childBase = uint(VOXEL_TREE_OFFSET_L3) + tid * 64u;
        uint lo = 0u, hi = 0u;
        for (uint c = 0u; c < 64u; c++) {
            uvec2 child = voxel_tree[childBase + c];
            if ((child.x | child.y) != 0u) {
                if (c < 32u) lo |= (1u << c);
                else         hi |= (1u << (c - 32u));
            }
        }
        voxel_tree[parentIdx] = uvec2(lo, hi);
    }

    #if VOXEL_TREE_TOP_LEVEL == 5
    // ---- Build Level 5 (root) from Level 4 ---- (grid=32 only)
    memoryBarrierBuffer();
    barrier();

    if (tid == 0u) {
        uint parentIdx = uint(VOXEL_TREE_OFFSET_L5);
        uint childBase = uint(VOXEL_TREE_OFFSET_L4);
        uint lo = 0u, hi = 0u;
        for (uint c = 0u; c < uint(L4_COUNT); c++) {
            uvec2 child = voxel_tree[childBase + c];
            if ((child.x | child.y) != 0u) {
                if (c < 32u) lo |= (1u << c);
                else         hi |= (1u << (c - 32u));
            }
        }
        voxel_tree[parentIdx] = uvec2(lo, hi);
    }
    #endif
}
