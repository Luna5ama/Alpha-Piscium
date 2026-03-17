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
// Entry points:
//   voxelray_setup(origin, dir, callbackData) → VoxelRay
//     Clips ray to grid, computes initial state.  Requires voxel_initShared().
//     Returns ray.level == 0 if ray misses the grid entirely.
//
//   voxel_traceRay(inout VoxelRay ray, int maxSteps) → VoxelHit
//     Traces up to maxSteps iterations.  On exhaustion the ray state is
//     written back for resumption (ray.level > 0).  On hit or grid exit
//     ray.level is set to 0.
//
// Must be included AFTER /Base.glsl (provides cameraPositionInt/Fract).
// The VOXEL_*_DATA_MODIFIER defines must be set before including this file.

#ifndef INCLUDE_techniques_VoxelTrace_glsl
#define INCLUDE_techniques_VoxelTrace_glsl

#include "/techniques/voxel/VoxelRayState.glsl"

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
    x = (x * 17u) & 0x0C30C3u;
    x = (x * 5u) & 0x249249u;
    return x;
}
#else
uint _voxel_spreadBits(uint x) {
    x &= 0x000003FFu;
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x << 8u)) & 0x0300F00Fu;
    x = (x | (x << 4u)) & 0x030C30C3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}
#endif

shared uint _voxel_levelOffsets[6];
shared ivec2 _voxel_levelSizeMask[6];
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
    // return spreadPos.x | (spreadPos.y << 1u) | (spreadPos.z << 2u)
    // Integer add/sub is 2x faster on Nvidia GPUs
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
    uint lutSize = uint(VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE);
    for (uint i = gl_LocalInvocationIndex; i < lutSize; i += localSize) {
        _voxel_spreadLUT[i] = _voxel_spreadBits(i);
    }

    if (gl_LocalInvocationIndex < 6u) {
        int cellShift = (int(gl_LocalInvocationIndex) - 1) << 1;
        int sizeMask = -(1 << cellShift);
        // Store absolute cell size (1 << cellShift) in the Y component
        _voxel_levelSizeMask[gl_LocalInvocationIndex] = ivec2(sizeMask, 1 << cellShift);
    }

    barrier();
}

// ---------------------------------------------------------------------------
// voxelray_setup
// Clips the ray to the voxel grid AABB, advances to the entry point, and
// initialises all VoxelRay fields ready for voxel_traceRay.
// Requires voxel_initShared() to have been called first (uses _voxel_spreadLUT).
// Returns a ray with level == 0 if the ray misses the grid entirely.
// ---------------------------------------------------------------------------
VoxelRay voxelray_setup(vec3 worldRayOrigin, vec3 worldRayDir, uint callbackData) {
    VoxelRay ray = voxelray_init();
    ray.worldRayOrigin = worldRayOrigin;
    ray.callbackData   = callbackData;

    const int   GRID_BLOCKS = VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE;
    const float EPS = 1e-4;

    // Sanitize direction — avoid zero components
    worldRayDir = mix(worldRayDir, vec3(1e-7), lessThan(abs(worldRayDir), vec3(1e-7)));
    ray.worldRayDir = worldRayDir;

    ivec3 cameraBrick = cameraPositionInt >> 4;
    vec3  gridOriginF = vec3((cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4);
    vec3  posGrid     = worldRayOrigin - gridOriginF;

    vec3  invDir   = 1.0 / worldRayDir;
    vec3  tOrig    = -posGrid * invDir;
    vec3  t1g      = fma(vec3(float(GRID_BLOCKS)), invDir, tOrig);
    vec3  tMinG    = min(tOrig, t1g);
    vec3  tMaxG    = max(tOrig, t1g);
    float tEnter   = max(max(tMinG.x, tMinG.y), tMinG.z);
    float tExitG   = min(min(tMaxG.x, tMaxG.y), tMaxG.z);

    if (tEnter > tExitG || tExitG <= 0.0) {
        // Miss — level stays 0
        return ray;
    }

    float tCurrent = max(tEnter, 0.0) + EPS;
    vec3  startPos = fma(worldRayDir, vec3(tCurrent), posGrid);
    startPos = clamp(startPos, vec3(EPS), vec3(float(GRID_BLOCKS) - EPS));
    ivec3 blockPos = ivec3(floor(startPos));

    ray.lastT = tCurrent;

    // Entry axis (only meaningful when ray started outside the grid)
    ray.lastAxis = -1;
    if (tEnter > 0.0) {
//        ray.lastAxis = (tMinG.x >= tMinG.y && tMinG.x >= tMinG.z) ? 0
//                     : (tMinG.y >= tMinG.z                        ? 1 : 2);
    }

    // Encode initial block position as fullMorton via shared spread LUT
    uvec3 spreadPos = _voxel_spreadPos(blockPos);
    ray.fullMorton  = _voxel_packSpreadPos(spreadPos);
    ray.level       = 1;

    return ray;
}

// ---------------------------------------------------------------------------
// Primary trace function (stateful)
// ---------------------------------------------------------------------------
VoxelHit voxel_traceRay(inout VoxelRay ray, int maxSteps) {
    #if VOXEL_TRACE_DEBUG_COUNTERS
    ivec4 debugCounters = ivec4(0);
    #endif

    // Early-out: ray missed grid or is already complete
    if (ray.level != 0) {
        const int GRID_BLOCKS = VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE;

        vec3 worldRayOrigin = ray.worldRayOrigin;
        vec3 worldRayDir = ray.worldRayDir;

        // ---- Coordinate frame: grid-local block space [0, GRID_BLOCKS) ----
        ivec3 cameraBrick = cameraPositionInt >> 4;
        vec3  gridOriginF = vec3((cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4);
        vec3  posGrid = worldRayOrigin - gridOriginF;

        vec3  invDir = 1.0 / worldRayDir;

        // ---- Precompute DDA biases ----
        ivec3 boundOffsetMask = ~(floatBitsToInt(worldRayDir) >> 31);;
        vec3  tOrig           = -posGrid * invDir;
        vec3  posGridBiased   = fma(sign(worldRayDir), vec3(1e-3), posGrid);

        // ---- Seed DDA state from ray ----
        float lastT = ray.lastT;
        ivec3 lastMask = ivec3(0.0);
        if (ray.lastAxis == 0) lastMask.x = 1;
        else if (ray.lastAxis == 1) lastMask.y = 1;
        else if (ray.lastAxis == 2) lastMask.z = 1;
        int   level = ray.level;
        uint  fullMorton = ray.fullMorton;

        // Derive blockPos from the authoritative fullMorton (avoids clamp/bias ambiguity)
        ivec3 blockPos = ivec3(morton3D_30bDecode(fullMorton));

        // ---- Main hierarchical traversal loop ----
        for (int i = 0; i < maxSteps; i++) {
            // Bounds check — also serves as grid-exit detection
            if (uint(blockPos.x | blockPos.y | blockPos.z) >= uint(GRID_BLOCKS)) {
                ray.level = 0;
                break;
            }

            #if VOXEL_TRACE_DEBUG_COUNTERS
            debugCounters.x++;
            #endif

            // Load node mask at current level
            uint childShift = 6u * uint(level - 1);
            uint mortonPrefix = fullMorton >> childShift;
            uint nodeIdx = _voxel_levelOffsets[level] + (mortonPrefix >> 6u);
            uvec2 mask = voxel_tree[nodeIdx];
            uint  childIdx = mortonPrefix & 63u;

            // Branchless bit check
            uint maskPart = mask[childIdx >> 5u];
            bool isHit = bool((maskPart >> (childIdx & 31u)) & 1u);

            if (isHit) {
                // ---- Non-empty child ----
                if (level == 1) {
                    // Leaf level: individual block is solid → HIT
                    uint allocID = voxel_brickAllocID[fullMorton >> 12u];
                    uint material = voxel_materials[(allocID << 12u) + (fullMorton & 0xFFFu)];

                    VoxelHit result;

                    result.hit = true;
                    result.hitPos = fma(worldRayDir, vec3(lastT), worldRayOrigin);
                    result.materialID = material;

                    result.normal = vec3(0.0);
                    ivec3 iMask = lastMask;
                    iMask.y &= ~iMask.x;             // Resolve ties (corner hits) to X
                    iMask.z &= ~(iMask.x | iMask.y); // Resolve ties to Y over Z
                    result.normal = -sign(worldRayDir) * vec3(iMask);

                    ray.level = 0;

                    #if VOXEL_TRACE_DEBUG_COUNTERS
                    result.debugCounters = debugCounters;
                    #endif
                    return result;
                }
                // Descend into child
                level--;
                #if VOXEL_TRACE_DEBUG_COUNTERS
                debugCounters.y++;
                #endif
            } else {
                // ---- Empty child — skip to exit of child cell ----
                #if VOXEL_TRACE_DEBUG_COUNTERS
                if (level == 1) debugCounters.w++;
                else debugCounters.z++;
                #endif

                ivec2 sizeMask = _voxel_levelSizeMask[level];
                ivec3 target = (blockPos & sizeMask.x) + (sizeMask.y & boundOffsetMask);

                vec3 tExit = fma(vec3(target), invDir, tOrig);
                lastT = min(min(tExit.x, tExit.y), tExit.z);

                // Reuse lastT to identify exit axis (saves 3 MIN vs step+min)
                lastMask = ivec3(lessThanEqual(tExit, vec3(lastT)));

                blockPos = ivec3(floor(fma(worldRayDir, vec3(lastT), posGridBiased)));
                uvec3 spreadPos = _voxel_spreadPos(blockPos);
                uint  oldFullMorton = fullMorton;
                fullMorton = _voxel_packSpreadPos(spreadPos);

                // Ascend: O(1) level recomputation via findMSB
                uint mortonDiff = oldFullMorton ^ fullMorton;
                int newLevel = ((findMSB(mortonDiff) * 43) >> 8) + 1;
                level = min(newLevel, VOXEL_TREE_TOP_LEVEL);
            }
        }

        // Write back state for resumption if still active (not done)
        if (ray.level != 0) {
            ray.lastT = lastT;
            ray.lastAxis = (lastMask.x > 0.5) ? 0 : ((lastMask.y > 0.5) ? 1 : 2);
            ray.level = level;
            ray.fullMorton = fullMorton;
        }
    }

    VoxelHit result;
    result.hit        = false;
    result.materialID = 0u;
    result.hitPos     = vec3(0.0);
    result.normal     = vec3(0.0, 1.0, 0.0);
    #if VOXEL_TRACE_DEBUG_COUNTERS
    result.debugCounters = debugCounters
    #endif
    return result;
}

#endif // INCLUDE_techniques_VoxelTrace_glsl



