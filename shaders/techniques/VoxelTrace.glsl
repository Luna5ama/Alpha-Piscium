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
//   Both skips are computed from the live DDA state (tMax, tDelta), avoiding
//   full AABB re-intersections.
//
// Must be included AFTER /Base.glsl (provides cameraPositionInt/Fract).
// The VOXEL_*_DATA_MODIFIER defines must be set before including this file
// (defaults to restrict readonly buffer if Voxelization.glsl is not already
// included).

#ifndef INCLUDE_techniques_VoxelTrace_glsl
#define INCLUDE_techniques_VoxelTrace_glsl

#include "/techniques/Voxelization.glsl"

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------
struct VoxelHit {
    bool  hit;
    uint  materialID;   // material written by the shadow pass; 0 = miss
    vec3  hitPos;       // world-space entry point of the hit block (1 unit = 1 block)
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

// Given the current DDA state (blockPos, tMax, tDelta, stepDir) and a cell
// AABB min in GRID block-space, compute the number of DDA steps until the
// ray exits the cell in each axis.  Zero-direction axes return a large value.
ivec3 _voxel_stepsToExit(ivec3 blockPos, ivec3 cellMin, int cellSize, ivec3 stepDirI) {
    ivec3 stepsPos = cellMin + cellSize - blockPos;   // dir > 0
    ivec3 stepsNeg = blockPos - cellMin + ivec3(1);   // dir < 0
    ivec3 steps    = mix(stepsNeg, stepsPos, greaterThan(stepDirI, ivec3(0)));
    // Zero-direction axes: ray never exits through that face.
    return mix(ivec3(0x7FFFF), steps, notEqual(stepDirI, ivec3(0)));
}

// Compute time to exit a cell and the outward face normal that corresponds to
// the exit face, given stepsToExit in each axis and the current DDA tMax /
// tDelta arrays.
float _voxel_exitTime(ivec3 stepsToExit, vec3 tMax, vec3 tDelta) {
    vec3 tExit = tMax + vec3(stepsToExit - 1) * tDelta;
    return min(tExit.x, min(tExit.y, tExit.z));
}

// Smallest-axis selector that avoids NaN/tie issues.
// Returns the normal of the ENTRY face (pointing toward the ray origin).
vec3 _voxel_exitNormal(ivec3 stepsToExit, vec3 tMax, vec3 tDelta, vec3 stepDir) {
    vec3 tExit = tMax + vec3(stepsToExit - 1) * tDelta;
    bvec3 isMin = bvec3(
        tExit.x <= tExit.y && tExit.x <= tExit.z,
        tExit.y <  tExit.x && tExit.y <= tExit.z,
        tExit.z <  tExit.x && tExit.z <  tExit.y
    );
    return -stepDir * vec3(isMin);
}

// ---------------------------------------------------------------------------
// Primary function
// ---------------------------------------------------------------------------
//
// worldRayOrigin : ray origin in world block coordinates.
//                  For primary rays: vec3(cameraPositionInt) + cameraPositionFract
// worldRayDir    : normalised ray direction  (block units == world units)
// maxSteps       : iteration cap (256 is comfortable for primary rays)
//
VoxelHit voxel_traceRay(vec3 worldRayOrigin, vec3 worldRayDir, int maxSteps) {
    VoxelHit result;
    result.hit        = false;
    result.materialID = 0u;
    result.hitPos     = vec3(0.0);
    result.normal     = vec3(0.0, 1.0, 0.0);

    const int   GRID_BLOCKS = VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE; // 256
    const float EPS         = 1e-4;

    // ------------------------------------------------------------------
    // Coordinate frame: posGrid is the FIXED ray origin in grid-local
    // block space.  All tMax values are absolute times from posGrid.
    // ------------------------------------------------------------------
    ivec3 cameraBrick = cameraPositionInt >> 4;
    vec3  gridOriginF = vec3((cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4);

    vec3  posGrid = worldRayOrigin - gridOriginF;   // fixed origin in [0,256)^3
    vec3  invDir  = 1.0 / worldRayDir;
    vec3  stepDir = sign(worldRayDir);
    ivec3 stepDirI = ivec3(stepDir);

    // ------------------------------------------------------------------
    // Clip ray to grid AABB and find entry / exit times.
    // ------------------------------------------------------------------
    vec3  t0g    = -posGrid * invDir;
    vec3  t1g    = (float(GRID_BLOCKS) - posGrid) * invDir;
    vec3  tMinG  = min(t0g, t1g);
    vec3  tMaxG  = max(t0g, t1g);
    float tEnter = max(max(tMinG.x, tMinG.y), tMinG.z);
    float tExitG = min(min(tMaxG.x, tMaxG.y), tMaxG.z);

    if (tEnter > tExitG || tExitG <= 0.0) return result;

    // Nudge into the grid.
    float tCurrent = max(tEnter, 0.0) + EPS;
    vec3  startPos = posGrid + worldRayDir * tCurrent;
    startPos       = clamp(startPos, vec3(EPS), vec3(float(GRID_BLOCKS) - EPS));

    // ------------------------------------------------------------------
    // DDA initialisation.
    // tMax[a] = absolute time (from posGrid) to next block boundary in axis a.
    // tDelta  = time per one-block step in each axis.
    // ------------------------------------------------------------------
    ivec3 blockPos = ivec3(startPos);
    vec3  tMax     = (vec3(blockPos) + max(stepDir, vec3(0.0)) - posGrid) * invDir;
    vec3  tDelta   = abs(invDir);   // time per 1-block step

    float lastT    = tCurrent;
    vec3  lastNorm = vec3(0.0);

    // Approximate entry normal from grid face (meaningful only when
    // the ray starts outside the grid).
    if (tEnter > 0.0) {
        bvec3 entered = equal(vec3(tEnter), tMinG);
        lastNorm      = -stepDir * vec3(entered);
    }

    // ------------------------------------------------------------------
    // Main DDA loop
    // ------------------------------------------------------------------
    for (int i = 0; i < maxSteps; i++) {

        if (any(lessThan   (blockPos, ivec3(0))) ||
            any(greaterThanEqual(blockPos, ivec3(GRID_BLOCKS)))) break;

        // ---- Level 0 : brick allocation check ----
        ivec3 brickRel    = blockPos >> 4;           // 0..15 per axis
        uint  brickMorton = voxel_brickMorton(brickRel);
        uint  allocID     = voxel_brickAllocID[brickMorton];

        if (allocID == VOXEL_UNALLOCATED) {
            // Skip the entire 16^3 brick in one jump.
            ivec3 brickMin    = brickRel * 16;
            ivec3 ste         = _voxel_stepsToExit(blockPos, brickMin, 16, stepDirI);
            vec3  tExitV      = tMax + vec3(ste - 1) * tDelta;
            float tSkip       = min(tExitV.x, min(tExitV.y, tExitV.z));
            bvec3 isMin = bvec3(
                tExitV.x <= tExitV.y && tExitV.x <= tExitV.z,
                tExitV.y <  tExitV.x && tExitV.y <= tExitV.z,
                tExitV.z <  tExitV.x && tExitV.z <  tExitV.y
            );
            lastNorm = -stepDir * vec3(isMin);
            lastT    = tSkip;

            vec3 newPos = posGrid + worldRayDir * (tSkip + EPS);
            if (any(lessThan(newPos, vec3(0.0))) ||
                any(greaterThanEqual(newPos, vec3(float(GRID_BLOCKS))))) break;

            blockPos = ivec3(newPos);
            tMax     = (vec3(blockPos) + max(stepDir, vec3(0.0)) - posGrid) * invDir;
            continue;
        }

        // ---- Level 1 : sub-region bitmask check ----
        ivec3 blockInBrick = blockPos & ivec3(15);   // 0..15 per axis
        ivec3 srCoord      = blockInBrick >> 2;      // 0..3  per axis
        uint  srMorton     = morton3D_encode(uvec3(srCoord));

        uvec2 rootMask = voxel_tree[voxel_treeRootIndex(allocID)];

        if (!_voxel_testBit64(rootMask, srMorton)) {
            // Skip the 4^3 sub-region.
            ivec3 srMin  = brickRel * 16 + srCoord * 4;
            ivec3 ste    = _voxel_stepsToExit(blockPos, srMin, 4, stepDirI);
            vec3  tExitV = tMax + vec3(ste - 1) * tDelta;
            float tSkip  = min(tExitV.x, min(tExitV.y, tExitV.z));
            bvec3 isMin = bvec3(
                tExitV.x <= tExitV.y && tExitV.x <= tExitV.z,
                tExitV.y <  tExitV.x && tExitV.y <= tExitV.z,
                tExitV.z <  tExitV.x && tExitV.z <  tExitV.y
            );
            lastNorm = -stepDir * vec3(isMin);
            lastT    = tSkip;

            vec3 newPos = posGrid + worldRayDir * (tSkip + EPS);
            if (any(lessThan(newPos, vec3(0.0))) ||
                any(greaterThanEqual(newPos, vec3(float(GRID_BLOCKS))))) break;

            blockPos = ivec3(newPos);
            tMax     = (vec3(blockPos) + max(stepDir, vec3(0.0)) - posGrid) * invDir;
            continue;
        }

        // ---- Level 2 : leaf block bitmask check ----
        ivec3 blockInSr    = blockInBrick & ivec3(3);   // 0..3 per axis
        uint  blockSrMorton = morton3D_encode(uvec3(blockInSr));

        uvec2 leafMask = voxel_tree[voxel_treeLeafIndex(allocID, srMorton)];

        if (_voxel_testBit64(leafMask, blockSrMorton)) {
            // *** HIT ***
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

    return result;   // miss
}

#endif // INCLUDE_techniques_VoxelTrace_glsl

