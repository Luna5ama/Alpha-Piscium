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
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Test bit [0..63] in a uint64_t mask.
bool _voxel_testBit64(uint64_t mask, uint idx) {
    return ((mask >> idx) & uint64_t(1)) != uint64_t(0);
}

// Advance the DDA past a cell using precomputed exit-plane intersections.
// tExit and exitBlockBias are computed at the call site from per-ray biases.
// tMax is uniformly recomputed from blockPos via FMA for all three axes.
void _voxel_skipCell(
    ivec3 cellMin, vec3 tExit, ivec3 exitBlockBias,
    vec3 worldRayDir, vec3 posGrid, vec3 invDir, vec3 tMaxBias, vec3 negStepDir,
    inout ivec3 blockPos, inout vec3 tMax,
    inout float lastT, inout vec3 lastNorm
) {
    vec3 exitPos;
    if (tExit.x <= tExit.y && tExit.x <= tExit.z) {
        lastT      = tExit.x;
        lastNorm   = vec3(negStepDir.x, 0.0, 0.0);
        blockPos.x = cellMin.x + exitBlockBias.x;
        exitPos    = fma(worldRayDir, vec3(lastT), posGrid);
        blockPos.y = int(floor(exitPos.y));
        blockPos.z = int(floor(exitPos.z));
    } else if (tExit.y <= tExit.z) {
        lastT      = tExit.y;
        lastNorm   = vec3(0.0, negStepDir.y, 0.0);
        blockPos.y = cellMin.y + exitBlockBias.y;
        exitPos    = fma(worldRayDir, vec3(lastT), posGrid);
        blockPos.x = int(floor(exitPos.x));
        blockPos.z = int(floor(exitPos.z));
    } else {
        lastT      = tExit.z;
        lastNorm   = vec3(0.0, 0.0, negStepDir.z);
        blockPos.z = cellMin.z + exitBlockBias.z;
        exitPos    = fma(worldRayDir, vec3(lastT), posGrid);
        blockPos.x = int(floor(exitPos.x));
        blockPos.y = int(floor(exitPos.y));
    }
    // Uniform tMax recomputation: tMax = blockPos * invDir + tMaxBias
    tMax = fma(vec3(blockPos), invDir, tMaxBias);
}

// ---------------------------------------------------------------------------
// Primary function
// ---------------------------------------------------------------------------
VoxelHit voxel_traceRay(vec3 worldRayOrigin, vec3 worldRayDir, int maxSteps) {
    VoxelHit result;
    result.hit        = false;
    result.materialID = 0u;
    result.hitPos     = vec3(0.0);
    result.normal     = vec3(0.0, 1.0, 0.0);

    const int   GRID_BLOCKS = VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE; // 256
    const float EPS         = 1e-4;

    // Coordinate frame: grid-local block space [0, 256)^3
    ivec3 cameraBrick = cameraPositionInt >> 4;
    vec3  gridOriginF = vec3((cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4);
    vec3  posGrid     = worldRayOrigin - gridOriginF;
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

    // exitBlockBiasN[i]: new blockPos on exit axis = cellMin + exitBlockBias
    //   dir > 0 → cellSize (first block of next cell)
    //   dir < 0 → -1       (last block of previous cell)
    ivec3 exitBlockBias16 = stepDirPos * 17 - 1;
    ivec3 exitBlockBias4  = stepDirPos * 5  - 1;

    // DDA initialisation
    float tCurrent = max(tEnter, 0.0) + EPS;
    vec3  startPos = fma(worldRayDir, vec3(tCurrent), posGrid);
    startPos       = clamp(startPos, vec3(EPS), vec3(float(GRID_BLOCKS) - EPS));

    ivec3 blockPos = ivec3(floor(startPos));
    vec3  tMax     = fma(vec3(blockPos), invDir, tMaxBias);
    vec3  tDelta   = abs(invDir);

    float lastT    = tCurrent;
    vec3  lastNorm = vec3(0.0);

    if (tEnter > 0.0) {
        bvec3 entered = equal(vec3(tEnter), tMinG);
        lastNorm      = negStepDir * vec3(entered);
    }

    // Main DDA loop
    for (int i = 0; i < maxSteps; i++) {
        // Unsigned bounds check: negative wraps to large uint, caught in one test
        if (any(greaterThanEqual(uvec3(blockPos), uvec3(GRID_BLOCKS)))) break;

        // ---- Level 0 : brick allocation check ----
        ivec3 brickRel    = blockPos >> 4;
        uint  brickMorton = voxel_brickMorton(brickRel);
        uint  allocID     = voxel_brickAllocID[brickMorton];

        if (allocID == VOXEL_UNALLOCATED) {
            ivec3 cellMin = blockPos & ivec3(-16);
            vec3  tExit   = fma(vec3(cellMin), invDir, exitTBias16);
            _voxel_skipCell(cellMin, tExit, exitBlockBias16,
                            worldRayDir, posGrid, invDir, tMaxBias, negStepDir,
                            blockPos, tMax, lastT, lastNorm);
            continue;
        }

        // ---- Level 1 : sub-region bitmask check ----
        ivec3    blockInBrick = blockPos & ivec3(15);
        ivec3    srCoord      = blockInBrick >> 2;
        uint     srMorton     = morton3D_6bEncode(uvec3(srCoord));
        uint64_t rootMask     = voxel_tree[voxel_treeRootIndex(allocID)];

        if (!_voxel_testBit64(rootMask, srMorton)) {
            ivec3 cellMin = blockPos & ivec3(-4);
            vec3  tExit   = fma(vec3(cellMin), invDir, exitTBias4);
            _voxel_skipCell(cellMin, tExit, exitBlockBias4,
                            worldRayDir, posGrid, invDir, tMaxBias, negStepDir,
                            blockPos, tMax, lastT, lastNorm);
            continue;
        }

        // ---- Level 2 : leaf block bitmask check ----
        ivec3    blockInSr     = blockInBrick & ivec3(3);
        uint     blockSrMorton = morton3D_6bEncode(uvec3(blockInSr));
        uint64_t leafMask      = voxel_tree[voxel_treeLeafIndex(allocID, srMorton)];

        if (_voxel_testBit64(leafMask, blockSrMorton)) {
            uint blockMorton  = voxel_blockMorton(blockInBrick);
            uint material     = voxel_materials[voxel_materialIndex(allocID, blockMorton)];
            result.hit        = true;
            result.materialID = material;
            result.hitPos     = fma(worldRayDir, vec3(lastT), worldRayOrigin);
            result.normal     = lastNorm;
            return result;
        }

        // ---- Standard one-block DDA step ----
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

