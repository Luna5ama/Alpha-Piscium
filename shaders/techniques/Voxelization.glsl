#ifndef INCLUDE_techniques_Voxelization_glsl
#define INCLUDE_techniques_Voxelization_glsl a
#include "/util/Morton.glsl"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const int VOXEL_BRICK_SIZE   = 16;     // blocks per brick side
const int VOXEL_GRID_SIZE    = 16;     // bricks per grid side (16^3 = 4096)
const int VOXEL_POOL_SIZE    = 512;    // max simultaneously allocated bricks
const uint VOXEL_UNALLOCATED = 0xFFFFFFFFu;

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
//   [0   .. 4095] : occupancy flags (0=empty, 1=occupied this frame)
//   [4096 .. 8191] : allocation IDs (<512=valid, 0xFFFFFFFF=unallocated)
//   [8192]        : allocation counter (next free slot, legacy)
//   [8193..8195]  : previous camera brick position (for cross-frame remapping)
// ---------------------------------------------------------------------------
layout(std430, binding = 3) VOXEL_BRICK_DATA_MODIFIER VoxelBrickData {
    uint voxel_brickAllocCounter;
    uint voxel_brickOccupancy[4096];
    uint voxel_brickAllocID[4096];
};

// ---------------------------------------------------------------------------
// SSBO 4 – Voxel Material Data
//   Indexed by (brickAllocID * 4096 + blockMorton)
//   Value: 16-bit material ID cast to uint; 0 = empty
// ---------------------------------------------------------------------------
layout(std430, binding = 4) VOXEL_MATERIAL_DATA_MODIFIER VoxelMaterialData {
    uint voxel_materials[];   // 512 * 4096 = 2,097,152 entries
};

// ---------------------------------------------------------------------------
// SSBO 8 – 64-Tree Data
//   Per brick: 1 root uint64_t + 64 leaf uint64_t (65 uint64_t total)
//   Root bit j = 1  if sub-region j (4^3) contains any non-empty block
//   Leaf j  bit k = 1  if block k within sub-region j is non-empty
//   Indexed by voxel_treeRootIndex / voxel_treeLeafIndex helpers below
// ---------------------------------------------------------------------------
layout(std430, binding = 8) VOXEL_TREE_DATA_MODIFIER VoxelTreeData {
    uint64_t voxel_tree[];       // 512 * 65 = 33,280 uint64_t entries
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Morton code (0..4095) for a brick at grid coordinate brickCoord (0..15/axis)
uint voxel_brickMorton(ivec3 brickCoord) {
    return morton3D_30bEncode(uvec3(brickCoord));
}

// Morton code (0..4095) for a block at local position blockInBrick (0..15/axis)
uint voxel_blockMorton(ivec3 blockInBrick) {
    return morton3D_30bEncode(uvec3(blockInBrick));
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

#endif // INCLUDE_techniques_Voxelization_glsl

