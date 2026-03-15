// 64-Tree Propagator – runs after VoxelTreeBuilder (shadowcomp3).
//
// Propagates the dense 64-tree upward from Level 2 (brick roots, built by
// VoxelTreeBuilder) through Level 3, 4, and optionally 5, up to the single
// root node.  Each parent bit is set if ANY of its 64 children have a
// non-zero mask.
//
// The upper levels are tiny (at most 4096 + 64 + 1 = 4161 nodes for grid=16),
// so a single workgroup with barriers between levels is sufficient.

#define VOXEL_TREE_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 1024) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    uint tid = gl_LocalInvocationID.x;

    // ---- Build Level 3 from Level 2 ----
    // Level 3 has (GRID_BLOCKS/64)^3 nodes.  Each reads 64 Level-2 children.
    #if VOXEL_GRID_SIZE == 16
    // Grid=16: Level 3 = 64 nodes (4^3).  tid < 64 does work.
    #define L3_COUNT 64
    #elif VOXEL_GRID_SIZE == 32
    // Grid=32: Level 3 = 512 nodes (8^3).  tid < 512 does work.
    #define L3_COUNT 512
    #elif VOXEL_GRID_SIZE == 64
    // Grid=64: Level 3 = 4096 nodes (16^3).  All 1024 threads, 4 nodes each.
    #define L3_COUNT 4096
    #endif

    #if L3_COUNT <= 1024
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
    #else
    // L3_COUNT > 1024 (grid=64): each thread handles multiple nodes
    for (uint n = tid; n < uint(L3_COUNT); n += 1024u) {
        uint parentIdx = uint(VOXEL_TREE_OFFSET_L3) + n;
        uint childBase = uint(VOXEL_TREE_OFFSET_L2) + n * 64u;
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
    #endif

    memoryBarrierBuffer();
    barrier();

    // ---- Build Level 4 from Level 3 ----
    // Grid=16: Level 4 = 1 node (root).  Grid=32: 8 nodes.  Grid=64: 64 nodes.
    #if VOXEL_GRID_SIZE == 16
    #define L4_COUNT 1
    #elif VOXEL_GRID_SIZE == 32
    #define L4_COUNT 8
    #elif VOXEL_GRID_SIZE == 64
    #define L4_COUNT 64
    #endif

    if (tid < uint(L4_COUNT)) {
        uint parentIdx = uint(VOXEL_TREE_OFFSET_L4) + tid;
        uint childBase = uint(VOXEL_TREE_OFFSET_L3) + tid * 64u;
        uint lo = 0u, hi = 0u;
        for (uint c = 0u; c < 64u; c++) {
            #if VOXEL_GRID_SIZE == 32
            // Grid=32: Level 3 has only 512 nodes.  Child local index = tid*64+c.
            if (tid * 64u + c >= uint(L3_COUNT)) break;
            #endif
            uvec2 child = voxel_tree[childBase + c];
            if ((child.x | child.y) != 0u) {
                if (c < 32u) lo |= (1u << c);
                else         hi |= (1u << (c - 32u));
            }
        }
        voxel_tree[parentIdx] = uvec2(lo, hi);
    }

    #if VOXEL_TREE_TOP_LEVEL == 5
    // ---- Build Level 5 (root) from Level 4 ----
    // Grid=32: 1 root with 8 valid children.  Grid=64: 1 root with 64 children.
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


