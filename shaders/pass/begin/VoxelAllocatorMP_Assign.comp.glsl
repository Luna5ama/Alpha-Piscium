// Multi-pass Brick Allocator — Pass 4/4: Assign
//
// For each brick with voxel_brickAllocID[gid] == 1u (set by Remap pass),
// atomically claims the next free slot in its distance bucket (whose starting
// ID is the exclusive prefix stored in voxel_bucketCounts[] by the PrefixSum
// pass).  Bricks within pool capacity get their final allocID written back;
// bricks beyond the pool cap get VOXEL_UNALLOCATED.  Unoccupied bricks also
// get VOXEL_UNALLOCATED.  The total allocated count is accumulated into
// voxel_brickAllocCounter via subgroup reduction + global atomic.
//
// One thread per brick; total threads = VOXEL_GRID_BRICKS.
// Only active when SETTING_VOXEL_GRID_SIZE > 16 (begin10.csh).

#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable

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
    vec3 cameraInBrick = vec3(cameraPositionInt & ivec3(VOXEL_BRICK_SIZE - 1)) + cameraPositionFract;

    uint threadAllocCount = 0u;

    if (voxel_brickAllocID[gid] == 1u) {
        ivec3 brickRelCoord = ivec3(morton3D_30bDecode(gid));
        uint dist = brickDistBucket(brickRelCoord, cameraInBrick);
        uint allocID = atomicAdd(voxel_bucketCounts[dist], 1u);
        if (allocID < uint(VOXEL_POOL_SIZE)) {
            voxel_brickAllocID[gid] = allocID;
            threadAllocCount = 1u;
        } else {
            voxel_brickAllocID[gid] = VOXEL_UNALLOCATED;
        }
    } else {
        voxel_brickAllocID[gid] = VOXEL_UNALLOCATED;
    }

    // Accumulate allocated count using subgroup reduction to reduce atomic pressure
    uint sgCount = subgroupAdd(threadAllocCount);
    if (subgroupElect()) {
        atomicAdd(voxel_brickAllocCounter, sgCount);
    }
}
