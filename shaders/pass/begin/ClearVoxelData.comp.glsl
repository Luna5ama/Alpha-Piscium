// Clears SSBO 4 (voxel material data) every frame so the shadow pass starts
// from a clean slate.  The dense 64-tree (SSBO 8) is self-clearing:
//   L1/L2 are always written by VoxelTreeBuilder (zeroed for unallocated bricks).
//   L3–L5 are always written by the propagator passes (zero when children are zero).
//
// Dispatch: (POOL_SIZE * 4096 / 256) workgroups × 256 threads.
// Each thread clears exactly one voxel_materials entry.

#define VOXEL_MATERIAL_DATA_MODIFIER buffer
#define VOXEL_MATERIAL_VEC4 a
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 256) in;

// workGroups = POOL_SIZE * 4096 / 256 = POOL_SIZE * 16
#if VOXEL_POOL_SIZE == 256
    #define _VOXEL_CLEAR_WG 1024
#elif VOXEL_POOL_SIZE == 512
    #define _VOXEL_CLEAR_WG 2048
#elif VOXEL_POOL_SIZE == 1024
    #define _VOXEL_CLEAR_WG 4096
#elif VOXEL_POOL_SIZE == 2048
    #define _VOXEL_CLEAR_WG 8192
#elif VOXEL_POOL_SIZE == 4096
    #define _VOXEL_CLEAR_WG 16384
#elif VOXEL_POOL_SIZE == 8192
    #define _VOXEL_CLEAR_WG 32768
#elif VOXEL_POOL_SIZE == 16384
    #define _VOXEL_CLEAR_WG 65536
#elif VOXEL_POOL_SIZE == 32768
    #define _VOXEL_CLEAR_WG 131072
#elif VOXEL_POOL_SIZE == 65536
    #define _VOXEL_CLEAR_WG 262144
#endif

const ivec3 workGroups = ivec3(_VOXEL_CLEAR_WG, 1, 1);

void main() {
    voxel_materials_v4[gl_GlobalInvocationID.x] = uvec4(0u);
}
