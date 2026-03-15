// Resets the voxel brick allocation counter so VoxelAllocator can re-assign
// brick IDs from 0 each frame.  Runs as a single-thread dispatch before
// VoxelAllocator to guarantee the counter is zero before any atomicAdd.

#define VOXEL_BRICK_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    voxel_brickAllocCounter = 0u;
}

