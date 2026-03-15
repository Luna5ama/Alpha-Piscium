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
// Algorithm:
//   Hierarchical descent / ascent through the tree.  The ray starts at the
//   top level and descends into non-empty children.  When a child is empty
//   the DDA skips to the exit of that child's cell, then ascends to the
//   correct parent level.  At the leaf level (L1), a set bit means the
//   individual block is solid → HIT.
//
// Must be included AFTER /Base.glsl (provides cameraPositionInt/Fract).
// The VOXEL_*_DATA_MODIFIER defines must be set before including this file

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

#if VOXEL_GRID_SIZE == 16
uint _voxel_spreadBits(uint x) {
    x = (x * 257u) & 0x00F00Fu;
    x = (x *  17u) & 0x0C30C3u;
    x = (x *   5u) & 0x249249u;
    return x;
}
#else
uint _voxel_spreadBits(uint x) {
    x &= 0x000003FFu;
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x <<  8u)) & 0x0300F00Fu;
    x = (x | (x <<  4u)) & 0x030C30C3u;
    x = (x | (x <<  2u)) & 0x09249249u;
    return x;
}
#endif

shared uint _voxel_levelOffsets[6];
shared uint _voxel_spreadLUT[VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE];

bool _voxel_testBit64(uvec2 mask, uint idx) {
    uint part = (idx < 32u) ? mask.x : mask.y;
    return ((part >> (idx & 31u)) & 1u) != 0u;
}

uvec3 _voxel_spreadPos(ivec3 blockPos) {
    return uvec3(
        _voxel_spreadLUT[uint(blockPos.x)],
        _voxel_spreadLUT[uint(blockPos.y)],
        _voxel_spreadLUT[uint(blockPos.z)]
    );
}

uint _voxel_packSpreadPos(uvec3 spreadPos) {
    return spreadPos.x + (spreadPos.y << 1u) + (spreadPos.z << 2u);
}

void voxel_initShared() {
    if (gl_LocalInvocationIndex == 0u) {
        _voxel_levelOffsets[0] = 0u;
        _voxel_levelOffsets[1] = uint(VOXEL_TREE_OFFSET_L1);
        _voxel_levelOffsets[2] = uint(VOXEL_TREE_OFFSET_L2);
        _voxel_levelOffsets[3] = uint(VOXEL_TREE_OFFSET_L3);
        _voxel_levelOffsets[4] = uint(VOXEL_TREE_OFFSET_L4);
        #if VOXEL_TREE_TOP_LEVEL == 5
        _voxel_levelOffsets[5] = uint(VOXEL_TREE_OFFSET_L5);
        #else
        _voxel_levelOffsets[5] = 0u;
        #endif
    }

    uint localSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
    uint lutSize   = uint(VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE);
    for (uint i = gl_LocalInvocationIndex; i < lutSize; i += localSize) {
        _voxel_spreadLUT[i] = _voxel_spreadBits(i);
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
    ivec3 stepDirI    = ivec3(sign(worldRayDir));
    vec3  exitSelectPos = vec3(max(stepDirI, 0));

    // ---- Clip ray to grid AABB ----
    vec3  tOrig  = -posGrid * invDir;          // dies after tMaxBias init
    vec3  t1g    = fma(vec3(float(GRID_BLOCKS)), invDir, tOrig);
    vec3  tMinG  = min(tOrig, t1g);
    vec3  tMaxG  = max(tOrig, t1g);
    float tEnter = max(max(tMinG.x, tMinG.y), tMinG.z);
    float tExitG = min(min(tMaxG.x, tMaxG.y), tMaxG.z);

    if (tEnter > tExitG || tExitG <= 0.0) return result;

    // ---- Precompute DDA biases ----
    vec3  tMaxBias     = fma(exitSelectPos, invDir, tOrig);

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

    // ---- Spread-position for incremental Morton at level 1 ----
    uvec3 spreadPos = _voxel_spreadPos(blockPos);
    uint fullMorton = _voxel_packSpreadPos(spreadPos);

    int level = 1;

    // ---- Main hierarchical traversal loop ----
    for (int i = 0; i < maxSteps; i++) {
        // Bounds check
        if (uint(blockPos.x | blockPos.y | blockPos.z) >= uint(GRID_BLOCKS)) break;

        #if VOXEL_TRACE_DEBUG_COUNTERS
        result.debugCounters.x++;
        #endif

        // Load node mask at current level
        uint childShift   = 6u * uint(level - 1);
        uint mortonPrefix = fullMorton >> childShift;
        uint nodeIdx      = _voxel_levelOffsets[level] + (mortonPrefix >> 6u);
        uvec2 mask        = voxel_tree[nodeIdx];
        uint childIdx     = mortonPrefix & 63u;

        if (_voxel_testBit64(mask, childIdx)) {
            // ---- Non-empty child ----
            if (level == 1) {
                // Leaf level: individual block is solid → HIT
                uint allocID  = voxel_brickAllocID[fullMorton >> 12u];
                uint material = voxel_materials[(allocID << 12u) + (fullMorton & 0xFFFu)];

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
            uint oldFullMorton = fullMorton;

            if (level == 1) {
                // Level 1: child is a single block → standard 1-block DDA step
                #if VOXEL_TRACE_DEBUG_COUNTERS
                result.debugCounters.w++;
                #endif
                if (tMax.x < tMax.y && tMax.x < tMax.z) {
                    lastT = tMax.x;
                    lastAxis = 0;
                    blockPos.x += stepDirI.x;
                    tMax.x = fma(float(blockPos.x), invDir.x, tMaxBias.x);
                    spreadPos.x = _voxel_spreadLUT[uint(blockPos.x)];
                } else if (tMax.y < tMax.z) {
                    lastT = tMax.y;
                    lastAxis = 1;
                    blockPos.y += stepDirI.y;
                    tMax.y = fma(float(blockPos.y), invDir.y, tMaxBias.y);
                    spreadPos.y = _voxel_spreadLUT[uint(blockPos.y)];
                } else {
                    lastT = tMax.z;
                    lastAxis = 2;
                    blockPos.z += stepDirI.z;
                    tMax.z = fma(float(blockPos.z), invDir.z, tMaxBias.z);
                    spreadPos.z = _voxel_spreadLUT[uint(blockPos.z)];
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
                // If dir > 0, target = origin + size - 1
                // If dir < 0, target = origin
                // (using precomputed boundOffsetMask to branchlessly add size-1 or 0)
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

                // Full Morton recompute after large jump
                spreadPos = _voxel_spreadPos(blockPos);
            }
            fullMorton = _voxel_packSpreadPos(spreadPos);

            // ---- Ascend: O(1) level recomputation via findMSB ----
            // The highest differing bit between old and new Morton codes
            // tells us which tree level boundary was crossed.
            uint mortonDiff = oldFullMorton ^ fullMorton;
            // Optim: skip level re-calc for small steps within same L2 node (diff < 64)
            if (mortonDiff >= 64u) {
                level = clamp(((findMSB(mortonDiff) * 43) >> 8) + 1, level, VOXEL_TREE_TOP_LEVEL);
            }
        }
    }

    return result;
}

#endif // INCLUDE_techniques_VoxelTrace_glsl

