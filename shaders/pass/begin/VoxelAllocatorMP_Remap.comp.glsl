// Multi-pass Brick Allocator — Pass 2/4: Remap + Count
//
// Reads voxel_brickOccupancy[] written by the previous frame's shadow pass,
// remaps each occupied brick from the old camera's coordinate space to the
// current camera's coordinate space, writes voxel_brickAllocID[newMorton] = 1u
// for in-range bricks, accumulates per-bucket counts into voxel_bucketCounts[]
// via atomicAdd, and clears voxel_brickOccupancy[] for this frame.
// One thread per brick; total threads = VOXEL_GRID_BRICKS.
//
// The remap is a bijection (no two old indices map to the same new index),
// so the voxel_brickAllocID writes need no atomics.
//
// Only active when SETTING_VOXEL_GRID_SIZE > 16 (begin8.csh).

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

    ivec3 currentCameraBrick = cameraPositionInt >> 4;
    ivec3 prevCameraBrick    = previousCameraPositionInt >> 4;
    ivec3 brickDelta         = currentCameraBrick - prevCameraBrick;

    if (voxel_brickOccupancy[gid] == 1u) {
        ivec3 oldRel = ivec3(morton3D_30bDecode(gid));
        ivec3 newRel = oldRel - brickDelta;
        if (all(greaterThanEqual(newRel, ivec3(0))) &&
            all(lessThan(newRel, ivec3(VOXEL_GRID_SIZE)))) {
            uint newMorton = morton3D_30bEncode(uvec3(newRel));
            voxel_brickAllocID[newMorton] = 1u;
            // Count into distance bucket using the new (current-frame) relative coordinate
            vec3 cameraInBrick = vec3(cameraPositionInt & ivec3(VOXEL_BRICK_SIZE - 1)) + cameraPositionFract;
            uint dist = brickDistBucket(newRel, cameraInBrick);
            atomicAdd(voxel_bucketCounts[dist], 1u);
        }
    }
    // Clear occupancy so this frame's shadow pass writes fresh marks
    voxel_brickOccupancy[gid] = 0u;
}

