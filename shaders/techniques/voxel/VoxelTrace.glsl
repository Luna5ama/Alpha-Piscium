// VoxelTrace.glsl
// Hierarchical DDA ray-tracer for the dense 64-tree voxel representation.
//
// Tree structure (see Voxelization.glsl):
//   4-5 levels depending on VOXEL_GRID_SIZE, from a single root node down
//   to individual blocks.  Each node is a 64-bit mask (uvec2) whose bits
//   indicate which of the 4^3 = 64 children are non-empty.
//
//   Level 1 (leaf) : bit = individual block in a 4^3 sub-region
//   Level 2        : bit = 4^3 sub-region within a 16^3 brick
//   Level 3+       : bit = aggregate of children at the level below
//   Level TOP      : single root covering the full grid
//
// Storage: 3D RG32UI custom image (uimg_voxelTree / usam_voxelTree).
//   Node address: ivec3(blockPos >> (2*L), blockPos.z >> (2*L) + LEVEL_Z_OFFSET)
//   Child index:  cz*16 + cy*4 + cx  where (cx,cy,cz) = (blockPos >> (2*(L-1))) & 3
//
// Algorithm:
//   Hierarchical descent / ascent through the tree.  The ray starts at the
//   top level and descends into non-empty children.  When a child is empty
//   the DDA skips to the exit of that child's cell, then ascends to the
//   correct parent level.  At the leaf level (L1), a set bit means the
//   individual block is solid → HIT.
//
// Must be included AFTER /Base.glsl (provides cameraPositionInt/Fract).

#ifndef INCLUDE_techniques_VoxelTrace_glsl
#define INCLUDE_techniques_VoxelTrace_glsl

#include "/techniques/voxel/Voxelization.glsl"

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------
struct VoxelHit {
    bool  hit;
    uint  materialID;   // material written by the shadow pass; 0 = miss
    vec3  hitPos;       // world-space entry point of the hit block
    vec3  normal;       // outward face normal of the hit surface

    #if VOXEL_TRACE_DEBUG_COUNTERS
    ivec4 debugCounters;
    #endif
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Per-level Z offsets in shared memory (loaded once by voxel_initShared).
// Index 0 unused; indices 1..VOXEL_TREE_TOP_LEVEL hold VOXEL_TREE_L*_Z.
shared int _voxel_levelZOffsets[6];

bool _voxel_testBit64(uvec2 mask, uint idx) {
    uint part = (idx < 32u) ? mask.x : mask.y;
    return ((part >> (idx & 31u)) & 1u) != 0u;
}

void voxel_initShared() {
    if (gl_LocalInvocationIndex == 0u) {
        _voxel_levelZOffsets[0] = 0;
        _voxel_levelZOffsets[1] = VOXEL_TREE_L1_Z;
        _voxel_levelZOffsets[2] = VOXEL_TREE_L2_Z;
        _voxel_levelZOffsets[3] = VOXEL_TREE_L3_Z;
        _voxel_levelZOffsets[4] = VOXEL_TREE_L4_Z;
        #if VOXEL_TREE_TOP_LEVEL == 5
        _voxel_levelZOffsets[5] = VOXEL_TREE_L5_Z;
        #else
        _voxel_levelZOffsets[5] = 0;
        #endif
    }

    barrier();
}

// ---------------------------------------------------------------------------
// Primary function
// ---------------------------------------------------------------------------
VoxelHit voxel_traceRay(vec3 worldRayOrigin, vec3 worldRayDir, int maxSteps) {
    VoxelHit result;
    result.hit            = false;
    result.materialID     = 0u;
    result.hitPos         = vec3(0.0);
    result.normal         = vec3(0.0, 1.0, 0.0);
    #if VOXEL_TRACE_DEBUG_COUNTERS
    result.debugCounters  = ivec4(0);
    #endif

    const int   GRID_BLOCKS = VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE;
    const float EPS         = 1e-4;

    // ---- Coordinate frame: grid-local block space [0, GRID_BLOCKS) ----
    ivec3 cameraBrick = cameraPositionInt >> 4;
    vec3  gridOriginF = vec3((cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4);
    vec3  posGrid     = worldRayOrigin - gridOriginF;

    worldRayDir = mix(worldRayDir, vec3(1e-7), lessThan(abs(worldRayDir), vec3(1e-7)));

    vec3  invDir      = 1.0 / worldRayDir;
    vec3  tDelta      = abs(invDir);
    ivec3 stepDirI    = ivec3(sign(worldRayDir));

    // ---- Clip ray to grid AABB ----
    vec3  tOrig  = -posGrid * invDir;
    vec3  t1g    = fma(vec3(float(GRID_BLOCKS)), invDir, tOrig);
    vec3  tMinG  = min(tOrig, t1g);
    vec3  tMaxG  = max(tOrig, t1g);
    float tEnter = max(max(tMinG.x, tMinG.y), tMinG.z);
    float tExitG = min(min(tMaxG.x, tMaxG.y), tMaxG.z);

    if (tEnter > tExitG || tExitG <= 0.0) return result;

    // ---- Precompute DDA biases ----
    vec3  tMaxBias     = fma(vec3(greaterThan(stepDirI, ivec3(0))), invDir, tOrig);

    // Optim: Precompute mask for hierarchy skipping (dir > 0 ? -1 : 0)
    ivec3 boundOffsetMask = ivec3(greaterThan(stepDirI, ivec3(0))) * ivec3(-1);

    // ---- DDA initialisation ----
    float tCurrent = max(tEnter, 0.0) + EPS;
    vec3  startPos = fma(worldRayDir, vec3(tCurrent), posGrid);
    startPos       = clamp(startPos, vec3(EPS), vec3(float(GRID_BLOCKS) - EPS));

    ivec3 blockPos = ivec3(floor(startPos));
    vec3  tMax     = fma(vec3(blockPos), invDir, tMaxBias);

    float lastT    = tCurrent;
    int   lastAxis = -1;

    vec3  posGridBiased = fma(vec3(stepDirI), vec3(1e-3), posGrid);

    if (tEnter > 0.0) {
        lastAxis = (tMinG.x >= tMinG.y && tMinG.x >= tMinG.z) ? 0 : (tMinG.y >= tMinG.z ? 1 : 2);
    }

    int level = 1;

    // ---- Main hierarchical traversal loop ----
    for (int i = 0; i < maxSteps; i++) {
        // Bounds check
        if (uint(blockPos.x | blockPos.y | blockPos.z) >= uint(GRID_BLOCKS)) break;

        #if VOXEL_TRACE_DEBUG_COUNTERS
        result.debugCounters.x++;
        #endif

        // ---- Load node mask at current level ----
        // nodeCoord = blockPos >> (2*level) gives the 3D index of the node.
        // Z component is offset by the per-level Z constant (from shared memory).
        ivec3 nodeCoord   = blockPos >> (2 * level);
        uvec2 mask        = voxel_treeLoad(ivec3(nodeCoord.xy, nodeCoord.z + _voxel_levelZOffsets[level]));
        // Child index: linear XYZ order (cz*16 + cy*4 + cx)
        ivec3 childLocal  = (blockPos >> (2 * (level - 1))) & ivec3(3);
        uint  childIdx    = uint(childLocal.z * 16 + childLocal.y * 4 + childLocal.x);

        if (_voxel_testBit64(mask, childIdx)) {
            // ---- Non-empty child ----
            if (level == 1) {
                // Leaf level: individual block is solid → HIT
                // Compute brick/block morton on-demand for material lookup
                ivec3 brickCoord  = blockPos >> 4;
                uint  brickMorton = voxel_brickMorton(brickCoord);
                uint  allocID     = voxel_brickAllocID[brickMorton];
                ivec3 blockInBrick = blockPos & ivec3(15);
                uint  blockMorton  = voxel_blockMorton(blockInBrick);
                uint  material     = voxel_materials[voxel_materialIndex(allocID, blockMorton)];

                result.hit        = true;
                result.hitPos     = fma(worldRayDir, vec3(lastT), worldRayOrigin);
                result.normal     = vec3(0.0);
                if (lastAxis != -1) result.normal[lastAxis] = float(-stepDirI[lastAxis]);
                result.materialID = material;
                return result;
            }
            // Descend into child
            level--;
            #if VOXEL_TRACE_DEBUG_COUNTERS
            result.debugCounters.y++;
            #endif
        } else {
            // ---- Empty child — skip to exit of child cell ----
            ivec3 oldBlockPos = blockPos;

            if (level == 1) {
                // Level 1: child is a single block → standard 1-block DDA step
                #if VOXEL_TRACE_DEBUG_COUNTERS
                result.debugCounters.w++;
                #endif
                if (tMax.x < tMax.y && tMax.x < tMax.z) {
                    lastT = tMax.x;
                    lastAxis = 0;
                    blockPos.x += stepDirI.x;
                    tMax.x += tDelta.x;
                } else if (tMax.y < tMax.z) {
                    lastT = tMax.y;
                    lastAxis = 1;
                    blockPos.y += stepDirI.y;
                    tMax.y += tDelta.y;
                } else {
                    lastT = tMax.z;
                    lastAxis = 2;
                    blockPos.z += stepDirI.z;
                    tMax.z += tDelta.z;
                }
            } else {
                // Level 2+: skip the child cell
                #if VOXEL_TRACE_DEBUG_COUNTERS
                result.debugCounters.z++;
                #endif

                int cellShift  = 2 * (level - 1);
                int sizeMinus1 = (1 << cellShift) - 1;
                int sizeMask   = ~sizeMinus1;

                // Determine target integer coordinate (exit boundary of current cell)
                ivec3 target = (blockPos & ivec3(sizeMask)) + (ivec3(sizeMinus1) & boundOffsetMask);

                vec3 tExit = fma(vec3(target), invDir, tMaxBias);

                if (tExit.x <= tExit.y && tExit.x <= tExit.z) {
                    lastT = tExit.x;
                    lastAxis = 0;
                } else if (tExit.y <= tExit.z) {
                    lastT = tExit.y;
                    lastAxis = 1;
                } else {
                    lastT = tExit.z;
                    lastAxis = 2;
                }

                blockPos = ivec3(floor(fma(worldRayDir, vec3(lastT), posGridBiased)));
                tMax     = fma(vec3(blockPos), invDir, tMaxBias);
            }

            // ---- Ascend: find lowest common ancestor level ----
            // The highest differing bit across all axes (bit b) tells us the
            // level: each level covers 4^L blocks, so bit b → level (b>>1)+1.
            // Skip recompute for small steps that stay within the same L1 node
            // (combined < 4 means only bits 0-1 differ → same 4-block region).
            ivec3 posDiff = oldBlockPos ^ blockPos;
            uint combined = uint(posDiff.x | posDiff.y | posDiff.z);
            if (combined >= 4u) {
                level = clamp(int(findMSB(combined) >> 1u) + 1, level, VOXEL_TREE_TOP_LEVEL);
            }
        }
    }

    return result;
}

#endif // INCLUDE_techniques_VoxelTrace_glsl

