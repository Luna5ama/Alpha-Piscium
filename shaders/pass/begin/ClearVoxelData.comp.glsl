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
#if VOXEL_POOL_SIZE == 256
const ivec3 workGroups = ivec3(4096, 1, 1);
#elif VOXEL_POOL_SIZE == 512
const ivec3 workGroups = ivec3(8192, 1, 1);
#elif VOXEL_POOL_SIZE == 1024
const ivec3 workGroups = ivec3(16384, 1, 1);
#elif VOXEL_POOL_SIZE == 2048
const ivec3 workGroups = ivec3(32768, 1, 1);
#elif VOXEL_POOL_SIZE == 4096
const ivec3 workGroups = ivec3(65536, 1, 1);
#elif VOXEL_POOL_SIZE == 8192
const ivec3 workGroups = ivec3(131072, 1, 1);
#elif VOXEL_POOL_SIZE == 16384
const ivec3 workGroups = ivec3(262144, 1, 1);
#elif VOXEL_POOL_SIZE == 32768
const ivec3 workGroups = ivec3(524288, 1, 1);
#elif VOXEL_POOL_SIZE == 65536
const ivec3 workGroups = ivec3(1048576, 1, 1);
#else
#error "Unsupported VOXEL_POOL_SIZE"
#endif

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < uint(VOXEL_POOL_SIZE) * 4096u) {
        voxel_materials[i] = 0u;
    }
}
