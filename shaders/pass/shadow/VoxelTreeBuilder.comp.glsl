// 64-Tree Builder – runs after the shadow pass (shadowcomp2).
//
// Builds the bottom 2 levels of the dense 64-tree for each allocated brick:
//   Level 1 (leaf, 64 uvec2 per brick): bit k = 1 if block k in sub-region is solid.
//   Level 2 (brick root, 1 uvec2 per brick): bit j = 1 if sub-region j has any solid block.
//
// Tree nodes are written to the 3D image (uimg_voxelTree) using direct 3D
// coordinates.  Sub-regions and blocks are indexed in linear XYZ order
// (z*16 + y*4 + x within the 4^3 space), which matches the tracer's
// childIdx = cz*16 + cy*4 + cx convention.
//
// Material data is still read via allocID (sparse pool, morton-indexed).
//
// Dispatch: one workgroup per brick in the VOXEL_GRID_SIZE^3 grid.
// Threads per workgroup: 64 (one per 4^3 sub-region within the brick).
// Tree image must have been cleared to 0 before this pass (done in begin5_b).

#define VOXEL_BRICK_DATA_MODIFIER restrict readonly buffer
#define VOXEL_MATERIAL_DATA_MODIFIER restrict readonly buffer
#define VOXEL_TREE_IMG_QUALIFIER restrict writeonly
#define VOXEL_TREE_WRITE_ONLY
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 64) in;
// One workgroup per brick in the VOXEL_GRID_SIZE^3 grid
#if VOXEL_GRID_SIZE == 64
const ivec3 workGroups = ivec3(262144, 1, 1);
#elif VOXEL_GRID_SIZE == 32
const ivec3 workGroups = ivec3(32768, 1, 1);
#else
const ivec3 workGroups = ivec3(4096, 1, 1);
#endif

shared uint rootMaskLo;
shared uint rootMaskHi;

void main() {
    uint brickMorton = uint(gl_WorkGroupID.x);   // 0..VOXEL_GRID_BRICKS-1
    uint subRegion   = uint(gl_LocalInvocationID.x); // 0..63 (linear)

    if (gl_LocalInvocationIndex == 0u) {
        rootMaskLo = 0u;
        rootMaskHi = 0u;
    }
    barrier();

    uint allocID = voxel_brickAllocID[brickMorton];
    if (allocID == VOXEL_UNALLOCATED) return;

    // Decode brickMorton → 3D brick coordinate within the grid
    #if VOXEL_GRID_SIZE == 16
    ivec3 brickCoord3D = ivec3(morton3D_12bDecode(brickMorton));
    #else
    ivec3 brickCoord3D = ivec3(morton3D_30bDecode(brickMorton));
    #endif

    // Decode thread index → 3D sub-region coordinate within the 4^3 sub-region grid
    // (linear XYZ: srCoord.z*16 + srCoord.y*4 + srCoord.x == subRegion)
    ivec3 srCoord3D = ivec3(int(subRegion & 3u), int((subRegion >> 2u) & 3u), int(subRegion >> 4u));

    // Build the 64-bit leaf mask for this sub-region.
    // Blocks within the sub-region are iterated in linear XYZ order; the bit
    // index equals the linear index (bz*16 + by*4 + bx == blockInSr).
    uint leafLow  = 0u;
    uint leafHigh = 0u;
    bool subRegionNonEmpty = false;

    for (uint blockInSr = 0u; blockInSr < 64u; blockInSr++) {
        // 3D coord of this block within the sub-region (linear decode)
        ivec3 bInSrCoord = ivec3(int(blockInSr & 3u), int((blockInSr >> 2u) & 3u), int(blockInSr >> 4u));
        // Full block coord in brick (0..15 per axis)
        ivec3 blockCoord = srCoord3D * 4 + bInSrCoord;
        // Material SSBO is still morton-indexed (unchanged)
        uint blockMorton = morton3D_12bEncode(uvec3(blockCoord));

        uint material = voxel_materials[allocID * 4096u + blockMorton];
        if (material != 0u) {
            subRegionNonEmpty = true;
            // bit index = blockInSr (linear order matches tracer childIdx)
            if (blockInSr < 32u) leafLow  |= (1u << blockInSr);
            else                 leafHigh |= (1u << (blockInSr - 32u));
        }
    }

    // Write Level 1 leaf node.
    // 3D position: brickCoord3D * 4 + srCoord3D, Z shifted by L1_Z offset.
    ivec3 l1Pos = brickCoord3D * 4 + srCoord3D;
    imageStore(uimg_voxelTree,
               ivec3(l1Pos.xy, l1Pos.z + VOXEL_TREE_L1_Z),
               uvec4(leafLow, leafHigh, 0u, 0u));

    // Accumulate sub-region occupancy into the shared brick root mask.
    // Bit index = subRegion (linear, matches tracer's childIdx convention).
    if (subRegionNonEmpty) {
        if (subRegion < 32u) atomicOr(rootMaskLo, 1u << subRegion);
        else                 atomicOr(rootMaskHi, 1u << (subRegion - 32u));
    }

    barrier();

    // Single writer: write Level 2 (brick root) node.
    // 3D position: brickCoord3D, Z shifted by L2_Z offset.
    if (gl_LocalInvocationIndex == 0u) {
        imageStore(uimg_voxelTree,
                   ivec3(brickCoord3D.xy, brickCoord3D.z + VOXEL_TREE_L2_Z),
                   uvec4(rootMaskLo, rootMaskHi, 0u, 0u));
    }
}
