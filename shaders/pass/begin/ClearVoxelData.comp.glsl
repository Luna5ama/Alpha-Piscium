// Clears SSBO 4 (voxel material data) and SSBO 8 (64-tree data) every frame,
// so the shadow pass and tree builder start from a clean slate.
//
// Dispatch: (VOXEL_POOL_SIZE * 16) workgroups × 256 threads = VOXEL_POOL_SIZE * 4096 threads.
// Each thread clears one voxel_materials entry; threads < VOXEL_POOL_SIZE*65 also clear
// their voxel_tree entry (VOXEL_POOL_SIZE bricks x 65 uint64_t total).

#define VOXEL_MATERIAL_DATA_MODIFIER buffer
#define VOXEL_TREE_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 256) in;
// VOXEL_POOL_SIZE * 4096 / 256 = VOXEL_POOL_SIZE * 16 workgroups
#if SETTING_VOXEL_POOL_SIZE == 256
const ivec3 workGroups = ivec3(4096, 1, 1);
#elif SETTING_VOXEL_POOL_SIZE == 512
const ivec3 workGroups = ivec3(8192, 1, 1);
#elif SETTING_VOXEL_POOL_SIZE == 1024
const ivec3 workGroups = ivec3(16384, 1, 1);
#elif SETTING_VOXEL_POOL_SIZE == 2048
const ivec3 workGroups = ivec3(32768, 1, 1);
#elif SETTING_VOXEL_POOL_SIZE == 4096
const ivec3 workGroups = ivec3(65536, 1, 1);
#elif SETTING_VOXEL_POOL_SIZE == 8192
const ivec3 workGroups = ivec3(131072, 1, 1);
#elif SETTING_VOXEL_POOL_SIZE == 16384
const ivec3 workGroups = ivec3(262144, 1, 1);
#elif SETTING_VOXEL_POOL_SIZE == 32768
const ivec3 workGroups = ivec3(524288, 1, 1);
#elif SETTING_VOXEL_POOL_SIZE == 65536
const ivec3 workGroups = ivec3(1048576, 1, 1);
#endif

void main() {
    uint i = gl_GlobalInvocationID.x;

    // Clear material slot (one per thread, covers full SSBO 4)
    voxel_materials[i] = 0u;

    // Clear tree slot for the first 33280 threads (SSBO 8 = 512*65 uint64_t)
    if (i < uint(VOXEL_POOL_SIZE * 65)) {
        voxel_tree[i] = uvec2(0u);
    }
}

