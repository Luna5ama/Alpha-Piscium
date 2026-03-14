// Brick Allocator – runs every frame in a begin pass, AFTER VoxelCounterReset.
//
// Reads the occupancy flags left by the PREVIOUS frame's shadow pass, assigns
// allocation IDs 0..511 (first-come, first-served by Morton index), then clears
// occupancy so the current frame's shadow pass can write fresh flags.
//
// One thread per brick (4096 threads total, 16 workgroups × 256).

#define VOXEL_BRICK_DATA_MODIFIER buffer
#include "/techniques/Voxelization.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(16, 1, 1); // 16 * 256 = 4096 threads

void main() {
    uint i = gl_GlobalInvocationID.x; // brick Morton index 0..4095

    if (voxel_brickOccupancy[i] == 1u) {
        uint allocID = atomicAdd(voxel_brickAllocCounter, 1u);
        voxel_brickAllocID[i] = (allocID < uint(VOXEL_POOL_SIZE)) ? allocID : VOXEL_UNALLOCATED;
    } else {
        voxel_brickAllocID[i] = VOXEL_UNALLOCATED;
    }

    // Clear occupancy so the shadow pass writes fresh marks this frame.
    // Safe: each thread owns exactly its own index; no cross-thread dependency.
    voxel_brickOccupancy[i] = 0u;
}

