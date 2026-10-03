#ifndef INCLUDE_techniques_Voxelization_glsl
#define INCLUDE_techniques_Voxelization_glsl a
#include "/util/Morton.glsl"
#include "/util/MaterialIDConst.glsl"

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
//   Value: material ID << 1 | full-cube flag; 0 = empty
//
//   VOXEL_MATERIAL_VEC4: define before including to get a uvec4 view instead.
//   Within a sub-region, blockMorton = subRegion * 64 + blockInSr, so all 64
//   entries are contiguous — safe to read 4-at-a-time as uvec4.
//   Base index (allocID * 4096 + subRegion * 64) is always divisible by 4.
// ---------------------------------------------------------------------------
#ifdef VOXEL_MATERIAL_VEC4
layout(std430, binding = 4) VOXEL_MATERIAL_DATA_MODIFIER VoxelMaterialData {
    uvec4 voxel_materials_v4[];   // material ID << 1 | full-cube flag
};
#else
layout(std430, binding = 4) VOXEL_MATERIAL_DATA_MODIFIER VoxelMaterialData {
    uint voxel_materials[];   // material ID << 1 | full-cube flag
};
#endif

uint voxel_decodeMaterialID(uint materialData) {
    return materialData >> 1u;
}

// ---------------------------------------------------------------------------
// Dense 64-Tree Layout
// ---------------------------------------------------------------------------
// Full-volume 64-tree with 4-5 levels depending on grid size.  Each level is
// a dense array of uvec2 (64-bit masks).  Level 1 (leaf) stores one bit per
// individual block in a 4^3 sub-region.  Higher levels aggregate children.
//
// Indexing (from a full Morton code of blockPos):
//   Child index at level L = (fullMorton >> (6*(L-1))) & 63
//   Node  index at level L = VOXEL_TREE_OFFSET_L<L> + (fullMorton >> (6*L))

#if VOXEL_GRID_SIZE == 16
    // 256 blocks/side = 4^4, 4 tree levels (L1..L4)
    #define VOXEL_TREE_TOP_LEVEL   4
    #define VOXEL_TREE_OFFSET_L4   0        // 1 root node
    #define VOXEL_TREE_OFFSET_L3   1        // 64 nodes
    #define VOXEL_TREE_OFFSET_L2   65       // 4096 nodes  (brick level)
    #define VOXEL_TREE_OFFSET_L1   4161     // 262144 nodes (sub-region / leaf)
    #define VOXEL_TREE_TOTAL       266305
#elif VOXEL_GRID_SIZE == 32
    // 512 blocks/side = 2*4^4, 5 tree levels (L1..L5, root uses 8/64 bits)
    #define VOXEL_TREE_TOP_LEVEL   5
    #define VOXEL_TREE_OFFSET_L5   0        // 1 root node  (8 valid children)
    #define VOXEL_TREE_OFFSET_L4   1        // 8 nodes
    #define VOXEL_TREE_OFFSET_L3   9        // 512 nodes
    #define VOXEL_TREE_OFFSET_L2   521      // 32768 nodes  (brick level)
    #define VOXEL_TREE_OFFSET_L1   33289    // 2097152 nodes (sub-region / leaf)
    #define VOXEL_TREE_TOTAL       2130441
#elif VOXEL_GRID_SIZE == 64
    // 1024 blocks/side = 4^5, 5 tree levels (L1..L5)
    #define VOXEL_TREE_TOP_LEVEL   5
    #define VOXEL_TREE_OFFSET_L5   0        // 1 root node
    #define VOXEL_TREE_OFFSET_L4   1        // 64 nodes
    #define VOXEL_TREE_OFFSET_L3   65       // 4096 nodes
    #define VOXEL_TREE_OFFSET_L2   4161     // 262144 nodes (brick level)
    #define VOXEL_TREE_OFFSET_L1   266305   // 16777216 nodes (sub-region / leaf)
    #define VOXEL_TREE_TOTAL       17043521
#endif

// ---------------------------------------------------------------------------
// SSBO 8 – Dense 64-Tree Data
//   VOXEL_TREE_TOTAL uvec2 entries laid out level-by-level (top-down).
//   Grid=16: ~2 MB,  Grid=32: ~16 MB,  Grid=64: ~130 MB.
// ---------------------------------------------------------------------------
//layout(std430, binding = 8) VOXEL_TREE_DATA_MODIFIER VoxelTreeData {
//    uvec2 voxel_tree[];       // VOXEL_TREE_TOTAL uvec2 entries
//};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Morton code for a brick at grid coordinate brickCoord (0..VOXEL_GRID_SIZE-1 per axis)
uint voxel_brickMorton(ivec3 brickCoord) {
    #if VOXEL_GRID_SIZE == 16
    return morton3D_12bEncode(uvec3(brickCoord));
    #else
    // Grid=32: coords 0..31 (5-bit), Grid=64: coords 0..63 (6-bit)
    return morton3D_24bEncode(uvec3(brickCoord));
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

// Base offset of tree level L in voxel_tree[].
// Used by tracer and tree builder to locate nodes at any level.
uint voxel_treeLevelOffset(int level) {
    #if VOXEL_TREE_TOP_LEVEL == 5
    if (level == 5) return uint(VOXEL_TREE_OFFSET_L5);
    #endif
    if (level == 4) return uint(VOXEL_TREE_OFFSET_L4);
    if (level == 3) return uint(VOXEL_TREE_OFFSET_L3);
    if (level == 2) return uint(VOXEL_TREE_OFFSET_L2);
    return uint(VOXEL_TREE_OFFSET_L1);
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

#ifndef VOXEL_MATERIAL_VEC4
bool voxel_opaqueAtBlock(ivec3 worldBlockPos) {
    ivec3 cameraBrick = cameraPositionInt >> 4;
    ivec3 gridOrigin = (cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4;
    ivec3 gridBlockPos = worldBlockPos - gridOrigin;
    if (any(lessThan(gridBlockPos, ivec3(0))) || any(greaterThanEqual(gridBlockPos, ivec3(VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE)))) {
        return false;
    }

    ivec3 brickCoord = gridBlockPos >> 4;
    uint brickMorton = voxel_brickMorton(brickCoord);
    uint allocID = voxel_brickAllocID[brickMorton];
    if (allocID == VOXEL_UNALLOCATED) {
        return false;
    }

    ivec3 blockInBrick = gridBlockPos & 15;
    uint blockMorton = voxel_blockMorton(blockInBrick);
    uint materialData = voxel_materials[voxel_materialIndex(allocID, blockMorton)];
    return materialData >= 4u;
}
uint voxel_getMaterialID(ivec3 worldBlockPos) {
    ivec3 cameraBrick = cameraPositionInt >> 4;
    ivec3 gridOrigin = (cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4;
    ivec3 gridBlockPos = worldBlockPos - gridOrigin;
    if (any(lessThan(gridBlockPos, ivec3(0))) || any(greaterThanEqual(gridBlockPos, ivec3(VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE)))) {
        return 0;
    }

    ivec3 brickCoord = gridBlockPos >> 4;
    uint brickMorton = voxel_brickMorton(brickCoord);
    uint allocID = voxel_brickAllocID[brickMorton];
    if (allocID == VOXEL_UNALLOCATED) {
        return 0;
    }

    ivec3 blockInBrick = gridBlockPos & 15;
    uint blockMorton = voxel_blockMorton(blockInBrick);
    return voxel_decodeMaterialID(voxel_materials[voxel_materialIndex(allocID, blockMorton)]);
}
#endif

vec3 voxel_faceNormal(uint faceId) {
    uint axis = faceId >> 1u;
    float signValue = 1.0 - 2.0 * float(faceId & 1u);

    vec3 axisMask = vec3(
        float(axis == 0u),
        float(axis == 1u),
        float(axis == 2u)
    );

    return axisMask * signValue;
}

vec3 voxel_faceTangent(uint faceId) {
    uint axis = faceId >> 1u;
    float signValue = 1.0 - 2.0 * float(faceId & 1u);

    vec3 axisMask = vec3(
        float(axis == 0u),
        float(axis == 1u),
        float(axis == 2u)
    );

    // Tangent points along the "top" edge of the face, so rotate the normal's axis by +1 mod 3
    uint tangentAxis = (axis + 1u) % 3u;
    vec3 tangentMask = vec3(
        float(tangentAxis == 0u),
        float(tangentAxis == 1u),
        float(tangentAxis == 2u)
    );

    return tangentMask * signValue;
}

#endif // INCLUDE_techniques_Voxelization_glsl
