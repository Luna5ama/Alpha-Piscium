// VoxelTrace.glsl
// Hierarchical DDA ray-tracer for the sparse 64-tree voxel representation.
//
// Tree structure (see Voxelization.glsl):
//   Top level  : 16×16×16 flat brick grid  (each cell = 16 blocks/axis)
//   Level 1    : root uvec2 per brick       (64-bit mask of 4³ sub-regions)
//   Level 2    : 64 leaf uvec2  per brick   (64-bit mask of 4³ blocks per sub-region)
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

// Test bit [0..63] packed as lo (x) + hi (y) uvec2.
bool _voxel_testBit64(uvec2 mask, uint idx) {
    return idx < 32u
        ? ((mask.x >>  idx)        & 1u) != 0u
        : ((mask.y >> (idx - 32u)) & 1u) != 0u;
}

// Compute the number of DDA steps from blockPos to the exit face of a cell.
ivec3 _voxel_stepsToExit(ivec3 blockPos, ivec3 cellMin, int cellSize, ivec3 stepDirI) {
    ivec3 stepsPos = cellMin + cellSize - blockPos;   // dir > 0
    ivec3 stepsNeg = blockPos - cellMin + ivec3(1);   // dir < 0
    ivec3 steps    = mix(stepsNeg, stepsPos, greaterThan(stepDirI, ivec3(0)));
    return mix(ivec3(0x7FFFF), steps, notEqual(stepDirI, ivec3(0)));
}

// Advance the DDA past a cell. Computes the exit axis exactly, updates
// blockPos (integer-exact for exit axis, floor(continuous) for others),
// tMax, lastT, and lastNorm.
void _voxel_skipCell(
    ivec3 cellMin, int cellSize, ivec3 stepDirI, vec3 stepDir,
    vec3 posGrid, vec3 worldRayDir, vec3 invDir, vec3 tDelta,
    inout ivec3 blockPos, inout vec3 tMax,
    inout float lastT, inout vec3 lastNorm
) {
    ivec3 ste   = _voxel_stepsToExit(blockPos, cellMin, cellSize, stepDirI);
    vec3  tExit = tMax + vec3(ste - 1) * tDelta;

    // Determine exit axis and advance.
    // Exit-axis: blockPos from exact integer arithmetic, tMax accumulated.
    // Non-exit axes: blockPos from floor(continuous position), tMax recomputed.
    // Using invDir (= 1/dir) avoids division-by-zero for axis-aligned rays;
    // for zero-dir axes, invDir = ±inf so tMax stays ±inf (never selected).
    vec3 exitPos;
    if (tExit.x <= tExit.y && tExit.x <= tExit.z) {
        lastT    = tExit.x;
        lastNorm = vec3(-stepDir.x, 0.0, 0.0);
        blockPos.x += ste.x * stepDirI.x;
        tMax.x     += float(ste.x) * tDelta.x;
        exitPos = posGrid + worldRayDir * lastT;
        blockPos.y = int(floor(exitPos.y));
        blockPos.z = int(floor(exitPos.z));
        tMax.y = (float(blockPos.y) + max(stepDir.y, 0.0) - posGrid.y) * invDir.y;
        tMax.z = (float(blockPos.z) + max(stepDir.z, 0.0) - posGrid.z) * invDir.z;
    } else if (tExit.y <= tExit.z) {
        lastT    = tExit.y;
        lastNorm = vec3(0.0, -stepDir.y, 0.0);
        blockPos.y += ste.y * stepDirI.y;
        tMax.y     += float(ste.y) * tDelta.y;
        exitPos = posGrid + worldRayDir * lastT;
        blockPos.x = int(floor(exitPos.x));
        blockPos.z = int(floor(exitPos.z));
        tMax.x = (float(blockPos.x) + max(stepDir.x, 0.0) - posGrid.x) * invDir.x;
        tMax.z = (float(blockPos.z) + max(stepDir.z, 0.0) - posGrid.z) * invDir.z;
    } else {
        lastT    = tExit.z;
        lastNorm = vec3(0.0, 0.0, -stepDir.z);
        blockPos.z += ste.z * stepDirI.z;
        tMax.z     += float(ste.z) * tDelta.z;
        exitPos = posGrid + worldRayDir * lastT;
        blockPos.x = int(floor(exitPos.x));
        blockPos.y = int(floor(exitPos.y));
        tMax.x = (float(blockPos.x) + max(stepDir.x, 0.0) - posGrid.x) * invDir.x;
        tMax.y = (float(blockPos.y) + max(stepDir.y, 0.0) - posGrid.y) * invDir.y;
    }
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
    vec3  stepDir     = sign(worldRayDir);
    ivec3 stepDirI    = ivec3(stepDir);

    // Clip ray to grid AABB
    vec3  t0g    = -posGrid * invDir;
    vec3  t1g    = (float(GRID_BLOCKS) - posGrid) * invDir;
    vec3  tMinG  = min(t0g, t1g);
    vec3  tMaxG  = max(t0g, t1g);
    float tEnter = max(max(tMinG.x, tMinG.y), tMinG.z);
    float tExitG = min(min(tMaxG.x, tMaxG.y), tMaxG.z);

    if (tEnter > tExitG || tExitG <= 0.0) return result;

    float tCurrent = max(tEnter, 0.0) + EPS;
    vec3  startPos = posGrid + worldRayDir * tCurrent;
    startPos       = clamp(startPos, vec3(EPS), vec3(float(GRID_BLOCKS) - EPS));

    // DDA initialisation
    ivec3 blockPos = ivec3(floor(startPos));
    vec3  tMax     = (vec3(blockPos) + max(stepDir, vec3(0.0)) - posGrid) * invDir;
    vec3  tDelta   = abs(invDir);

    float lastT    = tCurrent;
    vec3  lastNorm = vec3(0.0);

    if (tEnter > 0.0) {
        bvec3 entered = equal(vec3(tEnter), tMinG);
        lastNorm      = -stepDir * vec3(entered);
    }

    // Main DDA loop
    for (int i = 0; i < maxSteps; i++) {
        if (any(lessThan(blockPos, ivec3(0))) ||
            any(greaterThanEqual(blockPos, ivec3(GRID_BLOCKS)))) break;

        // ---- Level 0 : brick allocation check ----
        ivec3 brickRel    = blockPos >> 4;
        uint  brickMorton = voxel_brickMorton(brickRel);
        uint  allocID     = voxel_brickAllocID[brickMorton];

        if (allocID == VOXEL_UNALLOCATED) {
            ivec3 brickMin = brickRel * 16;
            _voxel_skipCell(brickMin, 16, stepDirI, stepDir,
                            posGrid, worldRayDir, invDir, tDelta,
                            blockPos, tMax, lastT, lastNorm);
            continue;
        }

        // ---- Level 1 : sub-region bitmask check ----
        ivec3 blockInBrick = blockPos & ivec3(15);
        ivec3 srCoord      = blockInBrick >> 2;
        uint  srMorton     = morton3D_30bEncode(uvec3(srCoord));
        uvec2 rootMask     = voxel_tree[voxel_treeRootIndex(allocID)];

        if (!_voxel_testBit64(rootMask, srMorton)) {
            ivec3 srMin = brickRel * 16 + srCoord * 4;
            _voxel_skipCell(srMin, 4, stepDirI, stepDir,
                            posGrid, worldRayDir, invDir, tDelta,
                            blockPos, tMax, lastT, lastNorm);
            continue;
        }

        // ---- Level 2 : leaf block bitmask check ----
        ivec3 blockInSr     = blockInBrick & ivec3(3);
        uint  blockSrMorton = morton3D_30bEncode(uvec3(blockInSr));
        uvec2 leafMask      = voxel_tree[voxel_treeLeafIndex(allocID, srMorton)];

        if (_voxel_testBit64(leafMask, blockSrMorton)) {
            uint blockMorton  = voxel_blockMorton(blockInBrick);
            uint material     = voxel_materials[voxel_materialIndex(allocID, blockMorton)];
            result.hit        = true;
            result.materialID = material;
            result.hitPos     = worldRayOrigin + worldRayDir * lastT;
            result.normal     = lastNorm;
            return result;
        }

        // ---- Standard one-block DDA step ----
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            lastT = tMax.x; lastNorm = vec3(-stepDir.x, 0.0, 0.0);
            blockPos.x += stepDirI.x; tMax.x += tDelta.x;
        } else if (tMax.y < tMax.z) {
            lastT = tMax.y; lastNorm = vec3(0.0, -stepDir.y, 0.0);
            blockPos.y += stepDirI.y; tMax.y += tDelta.y;
        } else {
            lastT = tMax.z; lastNorm = vec3(0.0, 0.0, -stepDir.z);
            blockPos.z += stepDirI.z; tMax.z += tDelta.z;
        }
    }

    return result;
}

#endif // INCLUDE_techniques_VoxelTrace_glsl

