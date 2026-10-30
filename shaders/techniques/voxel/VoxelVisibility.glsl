#ifndef INCLUDE_techniques_voxel_VoxelVisibility_glsl
#define INCLUDE_techniques_voxel_VoxelVisibility_glsl a

#include "/techniques/voxel/VoxelTraversal.glsl"
#include "/techniques/voxel/BlockModels.glsl"

bool _voxel_blockModelOccludes(
    uint modelData, vec3 rayOrigin, vec3 rayDir, vec3 inverseRayDir, float maxT
) {
    uint rotation = modelData & 0x1FFu;
    uint quadCount = (modelData >> 25u) & 0x3Fu;
    ivec2 texelCoord = ivec2(int(bitfieldExtract(modelData, 9, 14) << 1u), int(bitfieldExtract(modelData, 23, 2)));
    uvec3 transform = uvec3(rotation, rotation >> 3u, rotation >> 6u);
    ivec3 axis = ivec3(transform & 3u);
    vec3 transformSign = mix(vec3(1.0), vec3(-1.0), notEqual(transform & 4u, uvec3(0u)));
    rayOrigin -= vec3(0.5);
    rayOrigin = vec3(rayOrigin[axis.x], rayOrigin[axis.y], rayOrigin[axis.z]) * transformSign + vec3(0.5);
    rayDir = vec3(rayDir[axis.x], rayDir[axis.y], rayDir[axis.z]) * transformSign;

    if ((modelData & 0x80000000u) != 0u) {
        inverseRayDir = vec3(inverseRayDir[axis.x], inverseRayDir[axis.y], inverseRayDir[axis.z]) * transformSign;
        while (quadCount != 0u) {
            float hitT = maxT;
            int hitAxis = -1;
            _voxel_intersectBlockModelAxisAlignedQuad(rayOrigin, rayDir, inverseRayDir, texelCoord, hitT, hitAxis);
            if (hitAxis >= 0) return true;
            texelCoord.x += 2;
            --quadCount;
        }
    } else {
        while (quadCount != 0u) {
            float hitT = maxT;
            vec3 hitNormal;
            if (_voxel_intersectBlockModelQuad(rayOrigin, rayDir, texelCoord, hitT, hitNormal)) return true;
            texelCoord.x += 2;
            --quadCount;
        }
    }
    return false;
}

// A finite segment is visible only after reaching its end; exhausted or out-of-grid traces remain unverified.
bool voxel_isVisible(vec3 worldRayOrigin, vec3 worldRayDir, float maxT, int maxSteps) {
    if (maxT <= 0.0) return true;

    const int GRID_BLOCKS = VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE;
    const float EPS = 1e-4;
    worldRayDir = mix(worldRayDir, vec3(1e-7), lessThan(abs(worldRayDir), vec3(1e-7)));
    vec3 posGrid = worldRayOrigin - _voxel_gridOriginF;
    vec3 inverseRayDir = 1.0 / worldRayDir;
    vec3 tOrig = -posGrid * inverseRayDir;
    vec3 t1g = fma(vec3(float(GRID_BLOCKS)), inverseRayDir, tOrig);
    vec3 tMinG = min(tOrig, t1g);
    vec3 tMaxG = max(tOrig, t1g);
    float tEnter = max(max(tMinG.x, tMinG.y), tMinG.z);
    float tExitG = min(min(tMaxG.x, tMaxG.y), tMaxG.z);
    if (tEnter > 0.0 || tExitG < maxT) return false;

    vec3 startPos = clamp(fma(worldRayDir, vec3(EPS), posGrid), vec3(EPS), vec3(float(GRID_BLOCKS) - EPS));
    ivec3 blockPos = ivec3(floor(startPos));
    uint fullMorton = _voxel_packBlockPos(blockPos);
    int level = 1;
    ivec3 directionSign = floatBitsToInt(worldRayDir) >> 31;
    ivec3 boundOffsetMask = ~directionSign;
    uint rayFaceMask = uint(42 + directionSign.x + (directionSign.y << 2) + (directionSign.z << 4));

    for (int i = 0; i < maxSteps; i++) {
        uint mortonPrefix = fullMorton >> (6u * uint(level - 1));
        uint maskPart = voxel_treeScalar[_voxel_levelOffsets[level] + (mortonPrefix >> 5u)];
        bool occupied = (maskPart & (1u << (mortonPrefix & 31u))) != 0u;
        if (occupied && level != 1) {
            level--;
            continue;
        }
        if (occupied) {
            uint allocID = voxel_brickAllocID[fullMorton >> 12u];
            uint materialData = voxel_materials[(allocID << 12u) + (fullMorton & 0xFFFu)];
            if ((materialData & 1u) != 0u) return false;

            uint material = voxel_decodeMaterialID(materialData);
            uint modelData = texelFetch(usam_pbrLUT2, ivec2(int(rayFaceMask), int(material)), 0).x;
            if (modelData != 0u && material != MATERIAL_ID_WATER &&
                _voxel_blockModelOccludes(modelData, posGrid - vec3(blockPos), worldRayDir, inverseRayDir, maxT)) {
                return false;
            }
        }

        ivec2 sizeMask = _voxel_levelSizeMask[level];
        // Eight consecutive Morton children cover one aligned 2x2x2 region.
        if (((maskPart >> (mortonPrefix & 24u)) & 0xffu) == 0u) sizeMask *= 2;
        ivec3 cellMin = blockPos & sizeMask.x;
        ivec3 target = cellMin + (sizeMask.y & boundOffsetMask);
        vec3 tExit = fma(vec3(target), inverseRayDir, tOrig);
        float nextT = min(min(tExit.x, tExit.y), tExit.z);
        if (nextT >= maxT) return true;

        ivec3 cellMax = cellMin + sizeMask.y - 1;
        ivec3 tracedBlockPos = clamp(ivec3(floor(fma(worldRayDir, vec3(nextT), posGrid))), cellMin, cellMax);
        blockPos = mix(target + directionSign, tracedBlockPos, greaterThan(tExit, vec3(nextT)));
        if (uint(blockPos.x | blockPos.y | blockPos.z) >= uint(GRID_BLOCKS)) return false;

        uint oldFullMorton = fullMorton;
        fullMorton = _voxel_packBlockPos(blockPos);
        level = ((findMSB(oldFullMorton ^ fullMorton) * 43) >> 8) + 1;
    }
    return false;
}

#endif
