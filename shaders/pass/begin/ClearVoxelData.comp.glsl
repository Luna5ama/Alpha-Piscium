// Clears SSBO 4 (voxel material data) every frame so the shadow pass starts
// from a clean slate.  The voxel tree image (uimg_voxelTree) is cleared by
// the separate ClearVoxelTree pass (begin5_b).
//
// Dispatch: VOXEL_POOL_SIZE * 16 workgroups × 256 threads.
// Each thread clears one voxel_materials entry.

#define VOXEL_MATERIAL_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 256) in;

// POOL_SIZE * 4096 / 256 = POOL_SIZE * 16 workgroups
const ivec3 workGroups = ivec3(VOXEL_POOL_SIZE * 16, 1, 1);

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < uint(VOXEL_POOL_SIZE) * 4096u) {
        voxel_materials[i] = 0u;
    }
}
