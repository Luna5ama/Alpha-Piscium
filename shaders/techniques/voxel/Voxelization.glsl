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

// Image qualifier for the voxel tree 3D image (define before including to override).
// Define VOXEL_TREE_WRITE_ONLY before including to suppress voxel_treeLoad (for
// write-only passes where imageLoad on a writeonly image would be a compile error).
#ifndef VOXEL_TREE_IMG_QUALIFIER
#define VOXEL_TREE_IMG_QUALIFIER restrict readonly
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
// Dense 64-Tree 3D Image Layout
// ---------------------------------------------------------------------------
// Stored in a 3D RG32UI custom image (uimg_voxelTree / usam_voxelTree).
// Each node is one texel (uvec2 = 64-bit occupancy mask).
// All levels share the same XY footprint (widest = L1), stacked along Z.
//
// Node coordinate at level L:
//   nodeCoord = blockPos >> (2 * L)        (per-axis right-shift)
// Texel address:
//   ivec3(nodeCoord.xy, nodeCoord.z + VOXEL_TREE_L<L>_Z)
//
// Child index within a node (0-63, linear XYZ order):
//   childLocal = (blockPos >> (2*(L-1))) & ivec3(3)
//   childIdx   = childLocal.z * 16 + childLocal.y * 4 + childLocal.x

#if VOXEL_GRID_SIZE == 16
    // 256 blocks/side = 4^4, 4 tree levels (L1..L4)
    #define VOXEL_TREE_TOP_LEVEL   4
    #define VOXEL_TREE_XY          64     // L1 = 64 nodes/axis
    #define VOXEL_TREE_L1_Z        0      // Z=[0,64)   64^3 = 262144 nodes
    #define VOXEL_TREE_L2_Z        64     // Z=[64,80)  16^3 = 4096 nodes
    #define VOXEL_TREE_L3_Z        80     // Z=[80,84)  4^3  = 64 nodes
    #define VOXEL_TREE_L4_Z        84     // Z=[84,85)  1^3  = 1 root
    #define VOXEL_TREE_Z_TOTAL     85
#elif VOXEL_GRID_SIZE == 32
    // 512 blocks/side = 2×4^4, 5 tree levels (L1..L5)
    #define VOXEL_TREE_TOP_LEVEL   5
    #define VOXEL_TREE_XY          128
    #define VOXEL_TREE_L1_Z        0      // Z=[0,128)   128^3 nodes
    #define VOXEL_TREE_L2_Z        128    // Z=[128,160) 32^3  nodes
    #define VOXEL_TREE_L3_Z        160    // Z=[160,168) 8^3   nodes
    #define VOXEL_TREE_L4_Z        168    // Z=[168,170) 2^3   nodes
    #define VOXEL_TREE_L5_Z        170    // Z=[170,171) 1^3   root
    #define VOXEL_TREE_Z_TOTAL     171
#elif VOXEL_GRID_SIZE == 64
    // 1024 blocks/side = 4^5, 5 tree levels (L1..L5)
    #define VOXEL_TREE_TOP_LEVEL   5
    #define VOXEL_TREE_XY          256
    #define VOXEL_TREE_L1_Z        0      // Z=[0,256)   256^3 nodes
    #define VOXEL_TREE_L2_Z        256    // Z=[256,320) 64^3  nodes
    #define VOXEL_TREE_L3_Z        320    // Z=[320,336) 16^3  nodes
    #define VOXEL_TREE_L4_Z        336    // Z=[336,340) 4^3   nodes
    #define VOXEL_TREE_L5_Z        340    // Z=[340,341) 1^3   root
    #define VOXEL_TREE_Z_TOTAL     341
#endif

// 3D voxel tree image: one RG32UI texel per node (uvec2 = 64-bit occupancy mask).
layout(rg32ui) VOXEL_TREE_IMG_QUALIFIER uniform uimage3D uimg_voxelTree;

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

// Base Z offset of tree level L in the 3D image.
// Used by the builder/propagator (outside the hot trace loop).
// In the tracer, use _voxel_levelZOffsets[] from shared memory instead.
int voxel_treeLevelZOffset(int level) {
    #if VOXEL_TREE_TOP_LEVEL == 5
    if (level == 5) return VOXEL_TREE_L5_Z;
    #endif
    if (level == 4) return VOXEL_TREE_L4_Z;
    if (level == 3) return VOXEL_TREE_L3_Z;
    if (level == 2) return VOXEL_TREE_L2_Z;
    return VOXEL_TREE_L1_Z;
}

// Load a tree node (64-bit occupancy mask) from the 3D image.
// Switchable at compile time:
//   default               → texelFetch via usam_voxelTree (texture cache path)
//   VOXEL_TREE_USE_IMAGE_LOAD → imageLoad via uimg_voxelTree (image path)
// Not available in write-only passes (define VOXEL_TREE_WRITE_ONLY to suppress).
#ifndef VOXEL_TREE_WRITE_ONLY
#ifdef VOXEL_TREE_USE_IMAGE_LOAD
uvec2 voxel_treeLoad(ivec3 coord) {
    return imageLoad(uimg_voxelTree, coord).rg;
}
#else
uvec2 voxel_treeLoad(ivec3 coord) {
    return texelFetch(usam_voxelTree, coord, 0).rg;
}
#endif
#endif

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

