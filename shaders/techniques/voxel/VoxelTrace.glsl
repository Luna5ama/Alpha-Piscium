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
//   correct parent level. At the leaf level (L1), the material selects the
//   full-cube fast path or generated block-model intersection.
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
#define INCLUDE_techniques_VoxelTrace_glsl a

#include "/techniques/voxel/VoxelRayState.glsl"
#include "/util/HardcodedPBR.glsl"
#include "/techniques/voxel/BlockModels.glsl"

layout(std430, binding = 8) restrict readonly buffer VoxelTreeData {
    uint voxel_treeScalar[];
};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------
struct VoxelHit {
    bool hit;
    uint materialID;   // material written by the shadow pass; 0 = miss
    vec3 hitPos;       // world-space entry point of the hit block
    vec3 normal;       // outward face normal of the hit surface
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
    x = (x * 0x00010001u) & 0xFF0000FFu;
    x = (x * 0x00000101u) & 0x0F00F00Fu;
    x = (x * 0x00000011u) & 0xC30C30C3u;
    x = (x * 0x00000005u) & 0x49249249u;
    return x;
}
#endif

shared uint _voxel_levelOffsets[6];
shared ivec2 _voxel_levelSizeMask[6];
shared uint _voxel_spreadLUT[VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE];

uint _voxel_packBlockPos(ivec3 blockPos) {
    // Integer add/sub is 2x faster on Nvidia GPUs
    return _voxel_spreadLUT[uint(blockPos.x)] +
        (_voxel_spreadLUT[uint(blockPos.y)] << 1u) +
        (_voxel_spreadLUT[uint(blockPos.z)] << 2u);
}

void voxel_initShared() {
    if (gl_LocalInvocationIndex == 0u) {
        _voxel_levelOffsets[0] = 0u << 1u;
        _voxel_levelOffsets[1] = uint(VOXEL_TREE_OFFSET_L1) << 1u;
        _voxel_levelOffsets[2] = uint(VOXEL_TREE_OFFSET_L2) << 1u;
        _voxel_levelOffsets[3] = uint(VOXEL_TREE_OFFSET_L3) << 1u;
        _voxel_levelOffsets[4] = uint(VOXEL_TREE_OFFSET_L4) << 1u;
        #if VOXEL_TREE_TOP_LEVEL == 5
        _voxel_levelOffsets[5] = uint(VOXEL_TREE_OFFSET_L5) << 1u;
        #else
        _voxel_levelOffsets[5] = 0u << 1u;
        #endif
    }

    uint localSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
    uint lutSize = uint(VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE);
    for (uint i = gl_LocalInvocationIndex; i < lutSize; i += localSize) {
        _voxel_spreadLUT[i] = _voxel_spreadBits(i);
    }

    if (gl_LocalInvocationIndex == 0u) {
        _voxel_levelSizeMask[0] = ivec2(0);
    } else if (gl_LocalInvocationIndex < 6u) {
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
    ray.callbackData = callbackData;

    const int GRID_BLOCKS = VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE;
    const float EPS = 1e-4;

    // Sanitize direction — avoid zero components
    worldRayDir = mix(worldRayDir, vec3(1e-7), lessThan(abs(worldRayDir), vec3(1e-7)));
    ray.worldRayDir = worldRayDir;

    ivec3 cameraBrick = cameraPositionInt >> 4;
    vec3 gridOriginF = vec3((cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4);
    vec3 posGrid = worldRayOrigin - gridOriginF;

    vec3 invDir = 1.0 / worldRayDir;
    vec3 tOrig = -posGrid * invDir;
    vec3 t1g = fma(vec3(float(GRID_BLOCKS)), invDir, tOrig);
    vec3 tMinG = min(tOrig, t1g);
    vec3 tMaxG = max(tOrig, t1g);
    float tEnter = max(max(tMinG.x, tMinG.y), tMinG.z);
    float tExitG = min(min(tMaxG.x, tMaxG.y), tMaxG.z);

    if (tEnter > tExitG || tExitG <= 0.0) {
        // Miss — level stays 0
        return ray;
    }

    float tCurrent = max(tEnter, 0.0) + EPS;
    vec3 startPos = fma(worldRayDir, vec3(tCurrent), posGrid);
    startPos = clamp(startPos, vec3(EPS), vec3(float(GRID_BLOCKS) - EPS));
    ivec3 blockPos = ivec3(floor(startPos));

    ray.lastT = tCurrent;

    // Entry axis (only meaningful when ray started outside the grid)
    ray.lastAxis = -1;
    if (tEnter > 0.0) {
        ray.lastAxis = (tMinG.x >= tMinG.y && tMinG.x >= tMinG.z) ? 0 : (tMinG.y >= tMinG.z ? 1 : 2);
    }

    // Encode initial block position as fullMorton via shared spread LUT
    ray.fullMorton = _voxel_packBlockPos(blockPos);
    ray.level = 1;

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
        vec3 gridOriginF = vec3((cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4);
        vec3 posGrid = worldRayOrigin - gridOriginF;

        vec3 invDir = 1.0 / worldRayDir;

        // ---- Precompute DDA stepping ----
        ivec3 boundOffsetMask = ~(floatBitsToInt(worldRayDir) >> 31);
        vec3 tOrig = -posGrid * invDir;
        ivec3 stepDir = ivec3(sign(worldRayDir));
        ivec3 stepBack = min(stepDir, ivec3(0));

        // ---- Seed DDA state from ray ----
        float lastT = ray.lastT;
        int lastAxis = ray.lastAxis;
        int level = ray.level;
        uint fullMorton = ray.fullMorton;

        // Derive blockPos from the authoritative fullMorton (avoids clamp/bias ambiguity)
        ivec3 blockPos = ivec3(morton3D_30bDecode(fullMorton));

        // ---- Main hierarchical traversal loop ----
        for (int i = 0; i < maxSteps; i++) {
            // Bounds check — also serves as grid-exit detection
            if (uint(blockPos.x | blockPos.y | blockPos.z) >= uint(GRID_BLOCKS)) {
                break;
            }

            #if VOXEL_TRACE_DEBUG_COUNTERS
            debugCounters.x++;
            #endif

            // Load node mask at current level
            uint childShift = 6u * uint(level - 1);
            uint mortonPrefix = fullMorton >> childShift;
            uint nodeIdx = (mortonPrefix >> 6u);
            uint childIdx = mortonPrefix & 63u;

            // Branchless bit check
            uint maskPart = voxel_treeScalar[_voxel_levelOffsets[level] + (nodeIdx << 1u) + (childIdx >> 5u)];
            bool isHit = bool((maskPart >> (childIdx & 31u)) & 1u);

            if (isHit && level == 1) {
                uint allocID = voxel_brickAllocID[fullMorton >> 12u];
                uint material = voxel_materials[(allocID << 12u) + (fullMorton & 0xFFFu)];
                HardcodedPBR hardcoded = hardcodedpbr_decode(material);

                if (hardcoded.isFullCube) {
                    VoxelHit result;
                    result.hit = true;
                    result.hitPos = fma(worldRayDir, vec3(lastT), worldRayOrigin);
                    result.materialID = material;

                    vec3 normalDir = -vec3(stepDir);
                    result.normal = normalDir * vec3(equal(ivec3(lastAxis), ivec3(0, 1, 2)));
                    ray.level = 0;

                    #if VOXEL_TRACE_DEBUG_COUNTERS
                    result.debugCounters = debugCounters;
                    #endif
                    return result;
                }

                if (hardcoded.blockModelID != 0u && material != MATERIAL_ID_WATER) {
                    vec3 blockLocalRayOrigin = worldRayOrigin - gridOriginF - vec3(blockPos);
                    float modelT;
                    vec3 modelNormal;
                    if (voxel_intersectBlockModel(
                            hardcoded.blockModelID, blockLocalRayOrigin, worldRayDir, modelT, modelNormal)) {
                        VoxelHit result;
                        result.hit = true;
                        result.hitPos = fma(worldRayDir, vec3(modelT), worldRayOrigin);
                        result.materialID = material;
                        result.normal = modelNormal;
                        ray.level = 0;
                        #if VOXEL_TRACE_DEBUG_COUNTERS
                        result.debugCounters = debugCounters;
                        #endif
                        return result;
                    }
                }

                isHit = false;
            }

            if (isHit) {
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
                ivec3 cellMin = blockPos & sizeMask.x;
                ivec3 target = cellMin + (sizeMask.y & boundOffsetMask);

                vec3 tExit = fma(vec3(target), invDir, tOrig);
                lastT = min(min(tExit.x, tExit.y), tExit.z);

                // Reuse lastT to identify exit axis (saves 3 MIN vs step+min)
                bvec3 nonExitMask = greaterThan(tExit, vec3(lastT));

                ivec3 cellMax = cellMin + sizeMask.y - 1;
                ivec3 exitBlockPos = target + stepBack;
                blockPos = exitBlockPos;
                if (nonExitMask.z) {
                    blockPos.z = clamp(int(floor(fma(worldRayDir.z, lastT, posGrid.z))), cellMin.z, cellMax.z);
                } else {
                    lastAxis = 2;
                }
                if (nonExitMask.y) {
                    blockPos.y = clamp(int(floor(fma(worldRayDir.y, lastT, posGrid.y))), cellMin.y, cellMax.y);
                } else {
                    lastAxis = 1;
                }
                if (nonExitMask.x) {
                    blockPos.x = clamp(int(floor(fma(worldRayDir.x, lastT, posGrid.x))), cellMin.x, cellMax.x);
                } else {
                    lastAxis = 0;
                }

                if (uint(blockPos.x | blockPos.y | blockPos.z) >= uint(GRID_BLOCKS)) {
                    level = 0;
                    break;
                }

                uint oldFullMorton = fullMorton;
                fullMorton = _voxel_packBlockPos(blockPos);

                // Ascend: O(1) level recomputation via findMSB
                uint mortonDiff = oldFullMorton ^ fullMorton;
                int newLevel = ((findMSB(mortonDiff) * 43) >> 8) + 1;
                level = min(newLevel, VOXEL_TREE_TOP_LEVEL);
            }
        }

        // Write back state for resumption if still active (not done)
        ray.level = level;
        if (level != 0) {
            ray.lastT = lastT;
            ray.lastAxis = (lastAxis >= 0 && lastAxis <= 2) ? lastAxis : 2;
            ray.fullMorton = fullMorton;
        }
    }

    VoxelHit result;
    result.hit = false;
    result.materialID = 0u;
    result.hitPos = vec3(0.0);
    result.normal = vec3(0.0, 1.0, 0.0);
    #if VOXEL_TRACE_DEBUG_COUNTERS
    result.debugCounters = debugCounters;
    #endif
    return result;
}

#endif // INCLUDE_techniques_VoxelTrace_glsl
