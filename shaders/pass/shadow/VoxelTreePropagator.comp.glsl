// 64-Tree Propagator – runs after VoxelTreeBuilder (shadowcomp3).
//
// Propagates the dense 64-tree upward from Level 2 (brick roots, built by
// VoxelTreeBuilder) through Level 3, 4, and optionally 5, up to the single
// root node.  Each parent bit (linear index cz*16+cy*4+cx) is set if ANY of
// its 4^3 = 64 children have a non-zero mask.
//
// Reads children with imageLoad, writes parents with imageStore.
// The upper levels are tiny so a single workgroup with barriers is sufficient.

#define VOXEL_TREE_IMG_QUALIFIER restrict
#define VOXEL_TREE_WRITE_ONLY
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 1024) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    uint tid = gl_LocalInvocationID.x;

    // ---- Build Level 3 from Level 2 ----
    // L3 node count and nodes-per-axis per grid size:
    //   Grid=16: 64 nodes  (4/axis),  Grid=32: 512 nodes (8/axis),  Grid=64: 4096 nodes (16/axis)
    #if VOXEL_GRID_SIZE == 16
    #define L3_COUNT     64
    #define L3_AXIS_BITS 2    // log2(4)
    #define L3_AXIS_MASK 3
    #elif VOXEL_GRID_SIZE == 32
    #define L3_COUNT     512
    #define L3_AXIS_BITS 3    // log2(8)
    #define L3_AXIS_MASK 7
    #elif VOXEL_GRID_SIZE == 64
    #define L3_COUNT     4096
    #define L3_AXIS_BITS 4    // log2(16)
    #define L3_AXIS_MASK 15
    #endif

    for (uint n = tid; n < uint(L3_COUNT); n += 1024u) {
        // Decode flat index → 3D parent coord at L3
        ivec3 p = ivec3(int(n & uint(L3_AXIS_MASK)),
                        int((n >> L3_AXIS_BITS) & uint(L3_AXIS_MASK)),
                        int(n >> (2 * L3_AXIS_BITS)));
        uint lo = 0u, hi = 0u;
        for (int cz = 0; cz < 4; cz++)
        for (int cy = 0; cy < 4; cy++)
        for (int cx = 0; cx < 4; cx++) {
            ivec3 childCoord = p * 4 + ivec3(cx, cy, cz);
            uvec2 child = imageLoad(uimg_voxelTree,
                                    ivec3(childCoord.xy, childCoord.z + VOXEL_TREE_L2_Z)).rg;
            if ((child.x | child.y) != 0u) {
                uint bit = uint(cz * 16 + cy * 4 + cx);
                if (bit < 32u) lo |= (1u << bit);
                else           hi |= (1u << (bit - 32u));
            }
        }
        imageStore(uimg_voxelTree,
                   ivec3(p.xy, p.z + VOXEL_TREE_L3_Z),
                   uvec4(lo, hi, 0u, 0u));
    }

    memoryBarrierImage();
    barrier();

    // ---- Build Level 4 from Level 3 ----
    // L4 node count per grid size:
    //   Grid=16: 1 node (1/axis),  Grid=32: 8 nodes (2/axis),  Grid=64: 64 nodes (4/axis)
    #if VOXEL_GRID_SIZE == 16
    #define L4_COUNT     1
    #define L4_AXIS_BITS 0
    #define L4_AXIS_MASK 0
    #elif VOXEL_GRID_SIZE == 32
    #define L4_COUNT     8
    #define L4_AXIS_BITS 1    // log2(2)
    #define L4_AXIS_MASK 1
    #elif VOXEL_GRID_SIZE == 64
    #define L4_COUNT     64
    #define L4_AXIS_BITS 2    // log2(4)
    #define L4_AXIS_MASK 3
    #endif

    if (tid < uint(L4_COUNT)) {
        uint n = tid;
        #if VOXEL_GRID_SIZE == 16
        ivec3 p = ivec3(0);
        #else
        ivec3 p = ivec3(int(n & uint(L4_AXIS_MASK)),
                        int((n >> L4_AXIS_BITS) & uint(L4_AXIS_MASK)),
                        int(n >> (2 * L4_AXIS_BITS)));
        #endif
        uint lo = 0u, hi = 0u;
        for (int cz = 0; cz < 4; cz++)
        for (int cy = 0; cy < 4; cy++)
        for (int cx = 0; cx < 4; cx++) {
            ivec3 childCoord = p * 4 + ivec3(cx, cy, cz);
            // Guard for Grid=32 where L3 is only 8 per axis (child coord must be < 8)
            #if VOXEL_GRID_SIZE == 32
            if (any(greaterThanEqual(childCoord, ivec3(8)))) continue;
            #endif
            uvec2 child = imageLoad(uimg_voxelTree,
                                    ivec3(childCoord.xy, childCoord.z + VOXEL_TREE_L3_Z)).rg;
            if ((child.x | child.y) != 0u) {
                uint bit = uint(cz * 16 + cy * 4 + cx);
                if (bit < 32u) lo |= (1u << bit);
                else           hi |= (1u << (bit - 32u));
            }
        }
        imageStore(uimg_voxelTree,
                   ivec3(p.xy, p.z + VOXEL_TREE_L4_Z),
                   uvec4(lo, hi, 0u, 0u));
    }

    #if VOXEL_TREE_TOP_LEVEL == 5
    // ---- Build Level 5 (root) from Level 4 ----
    // Single root node at (0,0,0).
    // Grid=32: L4 only has 2 per axis (child coords 0-1); guard against 2-3.
    // Grid=64: L4 has 4 per axis; all cx,cy,cz ∈ 0-3 are valid.
    memoryBarrierImage();
    barrier();

    if (tid == 0u) {
        uint lo = 0u, hi = 0u;
        for (int cz = 0; cz < 4; cz++)
        for (int cy = 0; cy < 4; cy++)
        for (int cx = 0; cx < 4; cx++) {
            ivec3 childCoord = ivec3(cx, cy, cz);
            // Grid=32: L4 has only 2×2×2 = 8 valid nodes
            #if VOXEL_GRID_SIZE == 32
            if (any(greaterThanEqual(childCoord, ivec3(2)))) continue;
            #endif
            uvec2 child = imageLoad(uimg_voxelTree,
                                    ivec3(childCoord.xy, childCoord.z + VOXEL_TREE_L4_Z)).rg;
            if ((child.x | child.y) != 0u) {
                uint bit = uint(cz * 16 + cy * 4 + cx);
                if (bit < 32u) lo |= (1u << bit);
                else           hi |= (1u << (bit - 32u));
            }
        }
        imageStore(uimg_voxelTree,
                   ivec3(0, 0, VOXEL_TREE_L5_Z),
                   uvec4(lo, hi, 0u, 0u));
    }
    #endif
}

