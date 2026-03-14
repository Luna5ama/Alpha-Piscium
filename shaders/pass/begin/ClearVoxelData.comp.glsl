// Clears SSBO 4 (voxel material data) and SSBO 8 (64-tree data) every frame,
// so the shadow pass and tree builder start from a clean slate.
//
// Dispatch: 8192 workgroups × 256 threads = 2,097,152 threads.
// Each thread clears one voxel_materials entry; threads < 33280 also clear
// their voxel_tree entry (512 bricks × 65 uvec2 = 33,280 total).

#define VOXEL_MATERIAL_DATA_MODIFIER buffer
#define VOXEL_TREE_DATA_MODIFIER buffer
#include "/techniques/Voxelization.glsl"

layout(local_size_x = 256) in;
// 512 * 4096 / 256 = 8192 workgroups
const ivec3 workGroups = ivec3(8192, 1, 1);

void main() {
    uint i = gl_GlobalInvocationID.x;

    // Clear material slot (one per thread, covers full SSBO 4)
    voxel_materials[i] = 0u;

    // Clear tree slot for the first 33280 threads (SSBO 8 = 512*65 uvec2)
    if (i < uint(VOXEL_POOL_SIZE * 65)) {
        voxel_tree[i] = uvec2(0u);
    }
}

