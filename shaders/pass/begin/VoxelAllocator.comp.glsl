// Brick Allocator – single 1024-thread workgroup, distance-prioritised.
//
// Runs every frame in a begin pass, AFTER VoxelCounterReset.
// Reads the occupancy flags left by the PREVIOUS frame's shadow pass, remaps
// them from the old camera's coordinate space to the current camera's, assigns
// allocation IDs 0..VOXEL_POOL_SIZE-1 prioritising bricks closest to the camera,
// then clears occupancy so the current frame's shadow pass can write fresh flags.
//
// The remapping is critical: occupancy was written using Morton indices relative
// to the previous frame's camera brick position.  The shadow pass this frame
// uses the current camera brick position.  Without remapping, a camera-brick
// shift causes all alloc IDs to be at wrong indices → 1-frame "blink" every
// time the camera crosses a brick boundary.
//
// Algorithm (all within one workgroup, no global atomics needed):
//   0a. Clear voxel_brickAllocID[] to 0 (repurpose as remap temp: 0=empty, 1=occupied).
//   0b. Barrier.
//   1.  Remap occupancy: for each occupied brick at old Morton i, compute the
//       new relative brick coord and store voxel_brickAllocID[newMorton] = 1.
//       The remap is a bijection so no two old indices map to the same new index.
//       Also clears voxel_brickOccupancy for the current frame's shadow pass.
//   1b. Barrier.
//   2.  Count remapped occupied bricks per distance bucket.
//   3.  Prefix-sum buckets → per-bucket starting alloc IDs.
//   4.  Assign alloc IDs closest-first; bricks beyond pool cap get UNALLOCATED.
//
// Distance metric: Chebyshev distance in BLOCKS (float) from the camera's
// actual position (including sub-brick fraction) to each brick's centre.
//
// Grid centre = ivec3(VOXEL_GRID_SIZE/2).
// Max Chebyshev: Grid=16 ~136 blocks, Grid=32 ~264 blocks, Grid=64 ~520 blocks.

#define VOXEL_BRICK_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 1024) in;
const ivec3 workGroups = ivec3(1, 1, 1); // single workgroup: 1024 threads

#define BRICKS_PER_THREAD (VOXEL_GRID_SIZE * VOXEL_GRID_SIZE * VOXEL_GRID_SIZE / 1024)
#define NUM_DIST_BUCKETS 512  // Chebyshev distance buckets (units of 4 blocks), covers all grid sizes

shared ivec3 shared_brickDelta;
shared uint  shared_bucketCount[NUM_DIST_BUCKETS];
shared uint  shared_allocatedCount;

// Compute the Chebyshev distance bucket (in units of 4 blocks) from the camera to
// the centre of the brick at relative grid coordinate brickRelCoord.
uint brickDistBucket(ivec3 brickRelCoord, vec3 cameraInBrick) {
    const ivec3 gridCenter = ivec3(VOXEL_GRID_SIZE / 2);
    vec3 brickCenter = vec3((brickRelCoord - gridCenter) * VOXEL_BRICK_SIZE) + vec3(float(VOXEL_BRICK_SIZE) * 0.5);
    vec3 delta = abs(brickCenter - cameraInBrick);
    uint dist = uint(max(max(delta.x, delta.y), delta.z) / 4); // floor via truncation
    return min(dist, uint(NUM_DIST_BUCKETS - 1));
}

void main() {
    uint tid = gl_LocalInvocationID.x;

    // ---- Phase 0: camera brick delta (thread 0) + shared mem init (all) ----
    if (tid == 0u) {
        ivec3 currentCameraBrick = cameraPositionInt >> 4;
        ivec3 prevCameraBrick = previousCameraPositionInt >> 4;
        shared_brickDelta = currentCameraBrick - prevCameraBrick;
        shared_allocatedCount = 0u;
    }

    if (tid < uint(NUM_DIST_BUCKETS)) {
        shared_bucketCount[tid] = 0u;
    }
    barrier();

    ivec3 brickDelta = shared_brickDelta;

    // ---- Phase 0a: Clear voxel_brickAllocID to 0 (temp remap target) ----
    for (uint k = 0u; k < uint(BRICKS_PER_THREAD); k++) {
        uint i = tid * uint(BRICKS_PER_THREAD) + k;
        voxel_brickAllocID[i] = 0u;
    }
    memoryBarrierBuffer();
    barrier();

    // ---- Phase 1: Remap occupancy old → new coordinate space, clear occupancy ----
    // Occupancy at old Morton index i was relative to the previous camera brick.
    // Translate to the current camera brick and write 1 to voxel_brickAllocID[newMorton].
    // The mapping is a bijection so no two old indices map to the same new index.
    for (uint k = 0u; k < uint(BRICKS_PER_THREAD); k++) {
        uint i = tid * uint(BRICKS_PER_THREAD) + k;
        if (voxel_brickOccupancy[i] == 1u) {
            ivec3 oldRel = ivec3(morton3D_30bDecode(i));
            ivec3 newRel = oldRel - brickDelta;
            if (all(greaterThanEqual(newRel, ivec3(0))) &&
            all(lessThan(newRel, ivec3(VOXEL_GRID_SIZE)))) {
                uint newMorton = morton3D_30bEncode(uvec3(newRel));
                voxel_brickAllocID[newMorton] = 1u;
            }
        }
        // Clear SSBO occupancy so the shadow pass writes fresh marks this frame.
        voxel_brickOccupancy[i] = 0u;
    }
    memoryBarrierBuffer();
    barrier();

    // Camera's sub-brick position in blocks (0..~15.999 per axis).
    vec3 cameraInBrick = vec3(cameraPositionInt & ivec3(VOXEL_BRICK_SIZE - 1))
    + cameraPositionFract;

    // ---- Phase 2: Count remapped occupied bricks per distance bucket ----
    for (uint k = 0u; k < uint(BRICKS_PER_THREAD); k++) {
        uint i = tid * uint(BRICKS_PER_THREAD) + k;
        if (voxel_brickAllocID[i] == 1u) {
            ivec3 brickRelCoord = ivec3(morton3D_30bDecode(i));
            uint  dist = brickDistBucket(brickRelCoord, cameraInBrick);
            atomicAdd(shared_bucketCount[dist], 1u);
        }
    }
    barrier();

    // ---- Phase 3: Prefix sum (thread 0) ----
    if (tid == 0u) {
        uint running = 0u;
        for (uint b = 0u; b < uint(NUM_DIST_BUCKETS); b++) {
            uint cnt = shared_bucketCount[b];
            shared_bucketCount[b] = running;
            running += cnt;
        }
    }
    barrier();

    // ---- Phase 4: Assign alloc IDs closest-first ----
    for (uint k = 0u; k < uint(BRICKS_PER_THREAD); k++) {
        uint i = tid * uint(BRICKS_PER_THREAD) + k;
        if (voxel_brickAllocID[i] == 1u) {
            ivec3 brickRelCoord = ivec3(morton3D_30bDecode(i));
            uint  dist = brickDistBucket(brickRelCoord, cameraInBrick);

            uint allocID = atomicAdd(shared_bucketCount[dist], 1u);
            if (allocID < uint(VOXEL_POOL_SIZE)) {
                voxel_brickAllocID[i] = allocID;
                atomicAdd(shared_allocatedCount, 1u);
            } else {
                voxel_brickAllocID[i] = VOXEL_UNALLOCATED;
            }
        } else {
            voxel_brickAllocID[i] = VOXEL_UNALLOCATED;
        }
    }
    barrier();

    // Store the final allocated count to global atomic counter
    if (tid == 0u) {
        voxel_brickAllocCounter = shared_allocatedCount;
    }
}

