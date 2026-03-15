// Multi-pass Brick Allocator — Pass 1/4: Clear
//
// Clears voxel_brickAllocID[] to 0 (temp remap target used by Pass 2) and
// voxel_bucketCounts[] to 0 (distance bucket counts used by Passes 2–4).
// One thread per brick; total threads = VOXEL_GRID_BRICKS.
//
// Only active when SETTING_VOXEL_GRID_SIZE > 16 (begin7.csh).

#define VOXEL_BRICK_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 256) in;
#if VOXEL_GRID_SIZE == 64
const ivec3 workGroups = ivec3(1024, 1, 1); // 262144 / 256
#else
const ivec3 workGroups = ivec3(128, 1, 1);  // 32768 / 256
#endif

void main() {
    uint gid = gl_GlobalInvocationID.x;
    voxel_brickAllocID[gid] = 0u;
    if (gid < uint(NUM_DIST_BUCKETS)) {
        voxel_bucketCounts[gid] = 0u;
    }
}

