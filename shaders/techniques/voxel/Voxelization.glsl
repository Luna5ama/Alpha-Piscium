#ifndef INCLUDE_techniques_Voxelization_glsl
#define INCLUDE_techniques_Voxelization_glsl a
#include "/util/Morton.glsl"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const int VOXEL_BRICK_SIZE   = 16;     // blocks per brick side (fixed)
// VOXEL_GRID_SIZE and VOXEL_POOL_SIZE come from Options.glsl (via Base.glsl -> Morton.glsl)
#define VOXEL_GRID_SIZE   SETTING_VOXEL_GRID_SIZE  // bricks per grid side (16/32/64)
#define VOXEL_POOL_SIZE   SETTING_VOXEL_POOL_SIZE  // max simultaneously allocated bricks
#define VOXEL_GRID_BRICKS (VOXEL_GRID_SIZE * VOXEL_GRID_SIZE * VOXEL_GRID_SIZE)
const uint VOXEL_UNALLOCATED = 0xFFFFFFFFu;
#define NUM_DIST_BUCKETS 1024  // Chebyshev distance buckets (units of 4 blocks), covers all grid sizes

// ---------------------------------------------------------------------------
// SSBO modifiers (define before including to override)
// ---------------------------------------------------------------------------
#ifndef VOXEL_BRICK_DATA_MODIFIER
#define VOXEL_BRICK_DATA_MODIFIER restrict readonly buffer
#endif
#ifndef VOXEL_MATERIAL_DATA_MODIFIER
#define VOXEL_MATERIAL_DATA_MODIFIER restrict readonly buffer
#endif
#ifndef VOXEL_TREE_DATA_MODIFIER
#define VOXEL_TREE_DATA_MODIFIER restrict readonly buffer
#endif

// ---------------------------------------------------------------------------
// SSBO 3 – Brick Metadata
//   [0]                          : allocation count
//   [1   .. VOXEL_GRID_BRICKS]   : occupancy flags (0=empty, 1=occupied this frame)
//   [VOXEL_GRID_BRICKS+1 .. 2*VOXEL_GRID_BRICKS] : allocation IDs (<POOL=valid, 0xFFFFFFFF=unallocated)
//   [2*VOXEL_GRID_BRICKS+1 .. 2*VOXEL_GRID_BRICKS+NUM_DIST_BUCKETS] : per-bucket counts (allocator inter-pass)
// ---------------------------------------------------------------------------
layout(std430, binding = 3) VOXEL_BRICK_DATA_MODIFIER VoxelBrickData {
    uint voxel_brickAllocCounter;
    uint voxel_brickOccupancy[VOXEL_GRID_BRICKS];
    uint voxel_brickAllocID[VOXEL_GRID_BRICKS];
    uint voxel_bucketCounts[NUM_DIST_BUCKETS];
};

// ---------------------------------------------------------------------------
// SSBO 4 – Voxel Material Data
//   Indexed by (brickAllocID * 4096 + blockMorton)
//   Value: 16-bit material ID cast to uint; 0 = empty
// ---------------------------------------------------------------------------
layout(std430, binding = 4) VOXEL_MATERIAL_DATA_MODIFIER VoxelMaterialData {
    uint voxel_materials[];   // VOXEL_POOL_SIZE * 4096 entries
};

// ---------------------------------------------------------------------------
// SSBO 8 – 64-Tree Data
//   Per brick: 1 root uint64_t + 64 leaf uint64_t (65 uint64_t total)
//   Root bit j = 1  if sub-region j (4^3) contains any non-empty block
//   Leaf j  bit k = 1  if block k within sub-region j is non-empty
//   Indexed by voxel_treeRootIndex / voxel_treeLeafIndex helpers below
// ---------------------------------------------------------------------------
layout(std430, binding = 8) VOXEL_TREE_DATA_MODIFIER VoxelTreeData {
    uvec2 voxel_tree[];       // VOXEL_POOL_SIZE * 65 uvec2 entries
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Morton code for a brick at grid coordinate brickCoord (0..VOXEL_GRID_SIZE-1 per axis)
uint voxel_brickMorton(ivec3 brickCoord) {
    #if VOXEL_GRID_SIZE == 16
    return morton3D_12bEncode(uvec3(brickCoord));
    #else
    // Grid=32: coords 0..31 (5-bit), Grid=64: coords 0..63 (6-bit)
    // morton3D_30bEncode is correct for values < 256 (bug in step 2 only affects bits 8-9)
    return morton3D_30bEncode(uvec3(brickCoord));
    #endif
}

// Morton code (0..4095) for a block at local position blockInBrick (0..15/axis)
uint voxel_blockMorton(ivec3 blockInBrick) {
    return morton3D_12bEncode(uvec3(blockInBrick));
}

// Flat index into voxel_materials[]
uint voxel_materialIndex(uint brickAllocID, uint blockMorton) {
    return brickAllocID * 4096u + blockMorton;
}

// Index of the root node for a brick in voxel_tree[]
uint voxel_treeRootIndex(uint brickAllocID) {
    return brickAllocID * 65u;
}

// Index of leaf node for sub-region subRegion (0..63) of a brick
uint voxel_treeLeafIndex(uint brickAllocID, uint subRegion) {
    return brickAllocID * 65u + 1u + subRegion;
}

// Compute the Chebyshev distance bucket (in units of 4 blocks) from the camera to
// the centre of the brick at relative grid coordinate brickRelCoord.
// cameraInBrick: camera position in blocks within its own brick (0..~15.999 per axis).
uint brickDistBucket(ivec3 brickRelCoord, vec3 cameraInBrick) {
    const ivec3 gridCenter = ivec3(VOXEL_GRID_SIZE / 2);
    vec3 brickCenter = vec3((brickRelCoord - gridCenter) * VOXEL_BRICK_SIZE) + vec3(float(VOXEL_BRICK_SIZE) * 0.5);
    vec3 delta = abs(brickCenter - cameraInBrick);
    uint dist = uint(max(max(delta.x, delta.y), delta.z) / 4.0); // floor via truncation
    return min(dist, uint(NUM_DIST_BUCKETS - 1));
}

#endif // INCLUDE_techniques_Voxelization_glsl

