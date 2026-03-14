// Brick Allocator – single 1024-thread workgroup, distance-prioritised.
//
// Runs every frame in a begin pass, AFTER VoxelCounterReset.
// Reads the occupancy flags left by the PREVIOUS frame's shadow pass, assigns
// allocation IDs 0..511 prioritising bricks closest to the camera, then clears
// occupancy so the current frame's shadow pass can write fresh flags.
//
// Algorithm (all within one workgroup, no atomics to SSBO needed for counting):
//   1. Count occupied bricks per distance bucket.
//   2. Prefix-sum buckets → per-bucket starting alloc IDs.
//   3. Re-scan: each occupied brick claims the next ID in its bucket
//      (or VOXEL_UNALLOCATED if the pool is already full).
//   4. Clear occupancy.
//
// Distance metric: Chebyshev distance in BLOCKS (float) from the camera's
// actual position (including sub-brick fraction) to each brick's centre.
//   camera position within its brick = vec3(cameraPositionInt & 15) + cameraPositionFract
//   brick centre (relative)          = vec3((brickRelCoord - gridCenter) * 16) + 8.0
//   dist = floor( max(|delta.x|, |delta.y|, |delta.z|) )
//
// Grid centre = ivec3(VOXEL_GRID_SIZE/2) = ivec3(8,8,8).
// Max Chebyshev in a 16^3 grid = 136 blocks (corner brick, camera at far edge).
// → 137 buckets (indices 0..136).

#define VOXEL_BRICK_DATA_MODIFIER buffer
#include "/techniques/Voxelization.glsl"

layout(local_size_x = 1024) in;
const ivec3 workGroups = ivec3(1, 1, 1); // single workgroup: 1024 threads × 4 bricks = 4096

#define NUM_DIST_BUCKETS 137  // Chebyshev distances 0..136 blocks
#define BRICKS_PER_THREAD 4   // 4096 / 1024

// s_bucketCount is used for two purposes separated by barriers:
//   Pass 1 → accumulate occupied-brick counts per bucket.
//   Pass 2 → holds per-bucket next-free alloc ID (starts at prefix-sum result).
shared uint s_bucketCount[NUM_DIST_BUCKETS];

// Compute the Chebyshev distance bucket (in whole blocks) from the camera to
// the centre of the brick at relative grid coordinate brickRelCoord.
// cameraInBrick = camera's position within its own brick, in blocks with fraction.
uint brickDistBucket(ivec3 brickRelCoord, vec3 cameraInBrick) {
    const ivec3 gridCenter = ivec3(VOXEL_GRID_SIZE / 2);
    // Centre of the brick in block-space relative to the camera's brick origin.
    vec3 brickCenter = vec3((brickRelCoord - gridCenter) * VOXEL_BRICK_SIZE)
                       + vec3(float(VOXEL_BRICK_SIZE) * 0.5);
    vec3 delta = abs(brickCenter - cameraInBrick);
    uint dist  = uint(max(max(delta.x, delta.y), delta.z)); // floor via truncation
    return min(dist, uint(NUM_DIST_BUCKETS - 1));
}

void main() {
    uint tid = gl_LocalInvocationID.x;

    // Initialise shared bucket counters (137 buckets; threads 0..136 each init one).
    if (tid < uint(NUM_DIST_BUCKETS)) {
        s_bucketCount[tid] = 0u;
    }
    barrier();
    memoryBarrierShared();

    // Camera's sub-brick position in blocks (0..~15.999 per axis).
    vec3 cameraInBrick = vec3(cameraPositionInt & ivec3(VOXEL_BRICK_SIZE - 1))
                         + cameraPositionFract;

    // --- Pass 1: count occupied bricks per block-distance bucket ---
    for (uint k = 0u; k < uint(BRICKS_PER_THREAD); k++) {
        uint i = tid * uint(BRICKS_PER_THREAD) + k;
        if (voxel_brickOccupancy[i] == 1u) {
            ivec3 brickRelCoord = ivec3(morton3D_decode(i));
            uint  dist          = brickDistBucket(brickRelCoord, cameraInBrick);
            atomicAdd(s_bucketCount[dist], 1u);
        }
    }
    barrier();
    memoryBarrierShared();

    // --- Prefix sum (thread 0 only – 137 elements, trivially cheap) ---
    // After this, s_bucketCount[b] holds the *starting* alloc ID for bucket b.
    // These values then double as atomic rank counters in Pass 2.
    if (tid == 0u) {
        uint running = 0u;
        for (uint b = 0u; b < uint(NUM_DIST_BUCKETS); b++) {
            uint cnt         = s_bucketCount[b];
            s_bucketCount[b] = running;
            running         += cnt;
        }
    }
    barrier();
    memoryBarrierShared();

    // --- Pass 2: assign allocation IDs closest-first, clear occupancy ---
    for (uint k = 0u; k < uint(BRICKS_PER_THREAD); k++) {
        uint i = tid * uint(BRICKS_PER_THREAD) + k;
        if (voxel_brickOccupancy[i] == 1u) {
            ivec3 brickRelCoord = ivec3(morton3D_decode(i));
            uint  dist          = brickDistBucket(brickRelCoord, cameraInBrick);

            // Claim the next slot in this bucket; returns the slot index.
            uint allocID = atomicAdd(s_bucketCount[dist], 1u);
            voxel_brickAllocID[i] = (allocID < uint(VOXEL_POOL_SIZE)) ? allocID : VOXEL_UNALLOCATED;
        } else {
            voxel_brickAllocID[i] = VOXEL_UNALLOCATED;
        }

        // Clear occupancy so the shadow pass writes fresh marks this frame.
        voxel_brickOccupancy[i] = 0u;
    }
}

