// Clears SSBO 4 (voxel material data) and SSBO 8 (dense 64-tree data) every
// frame, so the shadow pass and tree builder start from a clean slate.
//
// Dispatch: ceil(max(POOL*4096, TREE_TOTAL) / 256) workgroups × 256 threads.
// Each thread clears one voxel_materials entry (if in range) and one
// voxel_tree entry (if in range).

#define VOXEL_MATERIAL_DATA_MODIFIER buffer
#define VOXEL_TREE_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 256) in;

// Compute workgroup count = ceil(max(POOL*4096, TREE_TOTAL) / 256)
// Material threads = POOL * 4096,  Tree threads = VOXEL_TREE_TOTAL
#define _VOXEL_MAT_THREADS  (VOXEL_POOL_SIZE * 4096)

// Iris requires literal integers in workGroups — pre-compute all combos.
// When mat_threads < tree_total: Grid32 → 8323, Grid64 → 66577
// Otherwise: POOL_SIZE * 16
#if VOXEL_GRID_SIZE == 32 && VOXEL_POOL_SIZE < 1024
    #define _VOXEL_CLEAR_WG 8323
#elif VOXEL_GRID_SIZE == 64 && VOXEL_POOL_SIZE < 8192
    #define _VOXEL_CLEAR_WG 66577
#elif VOXEL_POOL_SIZE == 256
    #define _VOXEL_CLEAR_WG 4096
#elif VOXEL_POOL_SIZE == 512
    #define _VOXEL_CLEAR_WG 8192
#elif VOXEL_POOL_SIZE == 1024
    #define _VOXEL_CLEAR_WG 16384
#elif VOXEL_POOL_SIZE == 2048
    #define _VOXEL_CLEAR_WG 32768
#elif VOXEL_POOL_SIZE == 4096
    #define _VOXEL_CLEAR_WG 65536
#elif VOXEL_POOL_SIZE == 8192
    #define _VOXEL_CLEAR_WG 131072
#elif VOXEL_POOL_SIZE == 16384
    #define _VOXEL_CLEAR_WG 262144
#elif VOXEL_POOL_SIZE == 32768
    #define _VOXEL_CLEAR_WG 524288
#elif VOXEL_POOL_SIZE == 65536
    #define _VOXEL_CLEAR_WG 1048576
#endif

const ivec3 workGroups = ivec3(_VOXEL_CLEAR_WG, 1, 1);

void main() {
    uint i = gl_GlobalInvocationID.x;

    // Clear material slot (SSBO 4)
    if (i < uint(_VOXEL_MAT_THREADS)) {
        voxel_materials[i] = 0u;
    }

    // Clear tree slot (SSBO 8)
    if (i < uint(VOXEL_TREE_TOTAL)) {
        voxel_tree[i] = uvec2(0u);
    }
}
