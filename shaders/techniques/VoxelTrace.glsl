// VoxelTrace.glsl
// Hierarchical DDA ray-tracer for the sparse 64-tree voxel representation.
//
// Tree structure (see Voxelization.glsl):
//   Top level  : 16×16×16 flat brick grid  (each cell = 16 blocks/axis)
//   Level 1    : root uint64_t per brick      (64-bit mask of 4^3 sub-regions)
//   Level 2    : 64 leaf uint64_t per brick   (64-bit mask of 4^3 blocks per sub-region)
//
// Algorithm (adapted from "A guide to fast voxel ray tracing using sparse 64-trees"):
//   Block-level DDA with two skip accelerators:
//     • Unallocated brick  → jump to exit of the 16³ brick AABB
//     • Empty sub-region   → jump to exit of the 4³ sub-region AABB
//
// Must be included AFTER /Base.glsl (provides cameraPositionInt/Fract).
// The VOXEL_*_DATA_MODIFIER defines must be set before including this file

#ifndef INCLUDE_techniques_VoxelTrace_glsl
#define INCLUDE_techniques_VoxelTrace_glsl

#include "/techniques/Voxelization.glsl"

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------
struct VoxelHit {
    bool  hit;
    uint  materialID;   // material written by the shadow pass; 0 = miss
    vec3  hitPos;       // world-space entry point of the hit block
    vec3  normal;       // outward face normal of the hit surface

    #if VOXEL_TRACE_DEBUG_COUNTERS
    // Debug counters (x=before-unalloc-skip, y=before-root-skip,
    //                 z=before-leaf-skip,   w=after-leaf-miss/DDA-step)
    ivec4 debugCounters;
    #endif
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// 24-bit 3D Morton encode for coords in [0,255] (no initial mask needed).
// Produces a packed code where:
//   bits [0:11]  = 12-bit Morton of (coords & 15)      = blockMorton
//   bits [12:23] = 12-bit Morton of (coords >> 4)       = brickMorton
//   bits [0:5]   = 6-bit Morton of (coords & 3)         = blockSrMorton
//   bits [6:11]  = 6-bit Morton of ((coords >> 2) & 3)  = srMorton
//
// Uses multiply instead of shift+OR for bit-spreading: since the bit groups
// never overlap at each step, (x | (x << N)) == x * (1 + 2^N). Integer
// multiply routes to the FMAHeavy pipe, relieving the ALU pipe bottleneck.
uint _voxel_morton24b(uvec3 x) {
    x = (x * 257u) & 0x00F00Fu;   // x*(1+2^8) == x|(x<<8) when x<256
    x = (x *  17u) & 0x0C30C3u;   // x*(1+2^4) == x|(x<<4), nibbles isolated
    x = (x *   5u) & 0x249249u;   // x*(1+2^2) == x|(x<<2), pairs isolated
    return x.x + x.y * 2u + x.z * 4u;
}

// Test bit [0..63] in a uint64_t mask.
bool _voxel_testBit64(uint64_t mask, uint idx) {
    return ((mask >> idx) & uint64_t(1)) != uint64_t(0);
}

// Advance the DDA past a cell using precomputed exit-plane intersections.
// tExit and exitBlockBias are computed at the call site from per-ray biases.
// tMax is recomputed via FMA for all three axes, reusing the floor'd float
// values for the two non-exit axes to avoid redundant int→float conversion.
// posGridBiased = posGrid + stepDir * 1e-3: pre-biased to nudge non-exit axes
// in the ray direction before floor(), preventing cell-boundary rounding from
// corrupting tMax.
void _voxel_skipCell(
vec3 tExit,
vec3 worldRayDir, vec3 posGridBiased, vec3 invDir, vec3 tMaxBias, vec3 negStepDir,
inout ivec3 blockPos, inout vec3 tMax,
inout float lastT, inout vec3 lastNorm
) {
    if (tExit.x <= tExit.y && tExit.x <= tExit.z) {
        lastT    = tExit.x;
        lastNorm = vec3(negStepDir.x, 0.0, 0.0);
    } else if (tExit.y <= tExit.z) {
        lastT    = tExit.y;
        lastNorm = vec3(0.0, negStepDir.y, 0.0);
    } else {
        lastT    = tExit.z;
        lastNorm = vec3(0.0, 0.0, negStepDir.z);
    }

    vec3 exitPos  = fma(worldRayDir, vec3(lastT), posGridBiased);
    vec3 floorPos = floor(exitPos);
    blockPos      = ivec3(floorPos);
    // Reuse floor'd float directly — avoids 3 I2F (SFU) from vec3(blockPos).
    // floor(x) is integer-valued, so float(int(floor(x))) == floor(x) exactly
    // for block positions in [0, 255].
    tMax = fma(floorPos, invDir, tMaxBias);
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

    const int   GRID_BLOCKS = VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE; // 256
    const float EPS         = 1e-4;

    // Coordinate frame: grid-local block space [0, 256)^3
    ivec3 cameraBrick = cameraPositionInt >> 4;
    vec3  gridOriginF = vec3((cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4);
    vec3  posGrid     = worldRayOrigin - gridOriginF;

    // Clamp near-zero direction components to avoid Inf/NaN in DDA.
    // When dir ≈ 0 on an axis, 1/dir → Inf, and 0*Inf → NaN poisons tMax,
    // causing the DDA to pick that axis every step with stepDirI=0 → stuck.
    worldRayDir = mix(worldRayDir, vec3(1e-7), lessThan(abs(worldRayDir), vec3(1e-7)));

    vec3  invDir      = 1.0 / worldRayDir;
    ivec3 stepDirI    = ivec3(sign(worldRayDir));
    vec3  stepDir     = vec3(stepDirI);
    vec3  negStepDir  = -stepDir;

    // exitSelectPos: 1.0 for positive ray direction, 0.0 otherwise
    ivec3 stepDirPos    = max(stepDirI, 0);
    vec3  exitSelectPos = vec3(stepDirPos);

    // Clip ray to grid AABB (reuse t0g for bias precomputation below)
    vec3  t0g    = -posGrid * invDir;
    vec3  t1g    = fma(vec3(float(GRID_BLOCKS)), invDir, t0g);
    vec3  tMinG  = min(t0g, t1g);
    vec3  tMaxG  = max(t0g, t1g);
    float tEnter = max(max(tMinG.x, tMinG.y), tMinG.z);
    float tExitG = min(min(tMaxG.x, tMaxG.y), tMaxG.z);

    if (tEnter > tExitG || tExitG <= 0.0) return result;

    // Precompute DDA biases (all derived from t0g = -posGrid * invDir)
    // tMaxBias:    tMax[i] = blockPos[i] * invDir[i] + tMaxBias[i]
    // exitTBiasN:  tExit[i] = cellMin[i] * invDir[i] + exitTBiasN[i]  (N = cellSize)
    vec3  tMaxBias    = fma(exitSelectPos, invDir, t0g);
    vec3  exitTBias16 = fma(exitSelectPos, 16.0 * invDir, t0g);
    vec3  exitTBias4  = fma(exitSelectPos, 4.0  * invDir, t0g);

    // DDA initialisation
    float tCurrent = max(tEnter, 0.0) + EPS;
    vec3  startPos = fma(worldRayDir, vec3(tCurrent), posGrid);
    startPos       = clamp(startPos, vec3(EPS), vec3(float(GRID_BLOCKS) - EPS));

    ivec3 blockPos = ivec3(floor(startPos));
    vec3  tMax     = fma(vec3(blockPos), invDir, tMaxBias);
    vec3  tDelta   = abs(invDir);

    float lastT    = tCurrent;
    vec3  lastNorm = vec3(0.0);

    // Pre-bias posGrid in the ray direction for skip-cell floor() stability.
    // At cell boundaries, float precision can round floor() to the PREVIOUS
    // cell, corrupting tMax. Nudging by stepDir*1e-3 prevents this.
    // Computed once here rather than per skip call.
    vec3  posGridBiased = fma(stepDir, vec3(1e-3), posGrid);

    if (tEnter > 0.0) {
        bvec3 entered = equal(vec3(tEnter), tMinG);
        lastNorm      = negStepDir * vec3(entered);
    }

    // Main DDA loop
    for (int i = 0; i < maxSteps; i++) {
        // Unsigned bounds check via OR-reduction: any negative component wraps
        // to a large uint, any component >= 256 has bits above bit 7.
        if (uint(blockPos.x | blockPos.y | blockPos.z) > 255u) break;

        // ---- Single 24-bit Morton encode of blockPos ----
        // Produces a combined code encoding both brick and block positions:
        //   [12:23] = brickMorton,  [0:11] = blockMorton
        //   [6:11]  = srMorton,     [0:5]  = blockSrMorton
        uint fullMorton     = _voxel_morton24b(uvec3(blockPos));
        uint brickMorton    = fullMorton >> 12u;
        uint allocID        = voxel_brickAllocID[brickMorton];

        // ---- Level 0 : brick allocation check ----
        #if VOXEL_TRACE_DEBUG_COUNTERS
        result.debugCounters.x++;
        #endif
        if (allocID == VOXEL_UNALLOCATED) {
            ivec3 cellMin = blockPos & ivec3(-16);
            vec3  tExit   = fma(vec3(cellMin), invDir, exitTBias16);
            _voxel_skipCell(tExit, worldRayDir, posGridBiased, invDir, tMaxBias, negStepDir,
                blockPos, tMax, lastT, lastNorm);
            continue;
        }

        // Deferred Morton extracts: only computed when brick is allocated.
        // Saves 3 ALU ops (AND, BFE, AND) on the hot brick-skip path.
        uint srMorton       = bitfieldExtract(fullMorton, 6, 6);

        // ---- Level 1 : root mask check ----
        uint64_t rootMask = voxel_tree[voxel_treeRootIndex(allocID)];

        // y: reached root check (brick was allocated)
        #if VOXEL_TRACE_DEBUG_COUNTERS
        result.debugCounters.y++;
        #endif
        if (!_voxel_testBit64(rootMask, srMorton)) {
            ivec3 cellMin = blockPos & ivec3(-4);
            vec3  tExit   = fma(vec3(cellMin), invDir, exitTBias4);
            _voxel_skipCell(tExit, worldRayDir, posGridBiased, invDir, tMaxBias, negStepDir,
                blockPos, tMax, lastT, lastNorm);
            continue;
        }

        // ---- Level 2 : leaf block bitmask check ----
        uint64_t leafMask   = voxel_tree[voxel_treeLeafIndex(allocID, srMorton)];
        uint blockSrMorton  = fullMorton & 63u;
        // z: reached leaf check (sub-region was non-empty)
        #if VOXEL_TRACE_DEBUG_COUNTERS
        result.debugCounters.z++;
        #endif
        if (_voxel_testBit64(leafMask, blockSrMorton)) {
            uint blockMorton = fullMorton & 0xFFFu;
            uint matIdx = voxel_materialIndex(allocID, blockMorton);
            uint material     = voxel_materials[matIdx];
            result.hit        = true;
            result.hitPos     = fma(worldRayDir, vec3(lastT), worldRayOrigin);
            result.normal     = lastNorm;
            result.materialID = material;
            return result;
        }

        // ---- Standard one-block DDA step ----
        // w: leaf was empty, advancing one block
        #if VOXEL_TRACE_DEBUG_COUNTERS
        result.debugCounters.w++;
        #endif
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            lastT = tMax.x; lastNorm = vec3(negStepDir.x, 0.0, 0.0);
            blockPos.x += stepDirI.x; tMax.x += tDelta.x;
        } else if (tMax.y < tMax.z) {
            lastT = tMax.y; lastNorm = vec3(0.0, negStepDir.y, 0.0);
            blockPos.y += stepDirI.y; tMax.y += tDelta.y;
        } else {
            lastT = tMax.z; lastNorm = vec3(0.0, 0.0, negStepDir.z);
            blockPos.z += stepDirI.z; tMax.z += tDelta.z;
        }
    }

    return result;
}

#endif // INCLUDE_techniques_VoxelTrace_glsl

