#ifndef INCLUDE_techniques_voxel_VoxelAlphaTest_glsl
#define INCLUDE_techniques_voxel_VoxelAlphaTest_glsl a

#include "/util/BitPacking.glsl"
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"
#include "/techniques/voxel/BlockModels.glsl"

float voxel_alphaTestThreshold(uint material, uint face) {
    uint index = voxel_faceTexcoordIndex(material, face) + VOXEL_FACE_TEXCOORD_COUNT;
    return uintBitsToFloat(voxel_faceTexcoords[index].y);
}

bool voxel_alphaTestFace(uint material, vec3 hitPos, uint face) {
    uint index = voxel_faceTexcoordIndex(material, face);
    float threshold = voxel_alphaTestThreshold(material, face);
    if (threshold == 0.0) return true;
    vec4 tc = unpackUnorm4x16(voxel_faceTexcoords[index]);
    vec2 atlasUV = mix(tc.xw, tc.zy, voxel_faceLocalUV(face, hitPos));
    return textureLod(usam_blockAtlasColor, atlasUV, 0.0).a >= threshold;
}

bool voxel_alphaTest(uint material, vec3 hitPos, vec3 normal) {
    return voxel_alphaTestFace(material, hitPos, voxel_faceIndexFromNormal(normal));
}

bool voxel_intersectAlphaTestedCube(
    uint material, vec3 rayOrigin, vec3 rayDir, vec3 inverseRayDir,
    float maxT, out float hitT, out vec3 hitNormal
) {
    vec3 t0 = -rayOrigin * inverseRayDir;
    vec3 t1 = fma(vec3(1.0), inverseRayDir, t0);
    vec3 tNear = min(t0, t1);
    vec3 tFar = max(t0, t1);
    float entryT = max(max(tNear.x, tNear.y), tNear.z);
    float exitT = min(min(tFar.x, tFar.y), tFar.z);
    int entryAxis = tNear.x >= tNear.y && tNear.x >= tNear.z ? 0 : (tNear.y >= tNear.z ? 1 : 2);
    vec3 entryNormal = -sign(rayDir) * vec3(equal(ivec3(entryAxis), ivec3(0, 1, 2)));
    // A segment starting inside an opaque block is already occluded.
    if (entryT < 0.0 && voxel_alphaTestThreshold(material, voxel_faceIndexFromNormal(entryNormal)) == 0.0) {
        hitT = 0.0;
        hitNormal = entryNormal;
        return true;
    }
    if (entryT >= 0.0 && entryT <= maxT &&
        voxel_alphaTest(material, fma(rayDir, vec3(entryT), rayOrigin), entryNormal)) {
        hitT = entryT;
        hitNormal = entryNormal;
        return true;
    }
    int exitAxis = tFar.x <= tFar.y && tFar.x <= tFar.z ? 0 : (tFar.y <= tFar.z ? 1 : 2);
    vec3 exitNormal = sign(rayDir) * vec3(equal(ivec3(exitAxis), ivec3(0, 1, 2)));
    if (exitT >= 0.0 && exitT <= maxT &&
        voxel_alphaTest(material, fma(rayDir, vec3(exitT), rayOrigin), exitNormal)) {
        hitT = exitT;
        hitNormal = exitNormal;
        return true;
    }
    return false;
}

void voxel_blockModelInterval(vec3 rayOrigin, vec3 inverseRayDir, out float minT, inout float maxT) {
    // Use the same local plane subtraction as the quad intersection; grid-space DDA distances round differently.
    vec3 t0 = -rayOrigin * inverseRayDir;
    vec3 t1 = (vec3(1.0) - rayOrigin) * inverseRayDir;
    vec3 nearT = min(t0, t1);
    vec3 farT = max(t0, t1);
    minT = max(0.0, max(max(nearT.x, nearT.y), nearT.z));
    maxT = min(maxT, min(min(farT.x, farT.y), farT.z));
}

bool voxel_missesBlockModelBounds(
    uint material, vec3 blockRayDir, vec3 rayOrigin, vec3 inverseRayDir, float minT, float maxT
) {
    uvec3 negative = uvec3(lessThan(blockRayDir, vec3(0.0)));
    uint octant = negative.x | (negative.y << 1u) | (negative.z << 2u);
    uint bounds = voxel_modelBounds[material * 8u + octant];
    // Cover rounding in local-space projection for rays that originate far from this block.
    vec3 magnitude = abs(rayOrigin);
    float slack = 1e-5 + 2e-6 * max(max(magnitude.x, magnitude.y), magnitude.z);
    vec3 lo = vec3((uvec3(bounds) >> uvec3(0, 5, 10)) & 31u) * (1.0 / 31.0) - slack;
    vec3 hi = vec3((uvec3(bounds) >> uvec3(15, 20, 25)) & 31u) * (1.0 / 31.0) + slack;
    vec3 nearPlane = (lo - rayOrigin) * inverseRayDir;
    vec3 farPlane = (hi - rayOrigin) * inverseRayDir;
    vec3 nearT = min(nearPlane, farPlane);
    vec3 farT = max(nearPlane, farPlane);
    float entryT = max(max(nearT.x, nearT.y), nearT.z);
    float exitT = min(min(farT.x, farT.y), farT.z);
    return max(minT, entryT) > min(maxT, exitT);
}

#ifdef RESTIR_INITIAL_CANDIDATE_WRITE
bool voxel_intersectAlphaTestedBlockModel(
    uint modelData, uint material, vec3 rayOrigin, vec3 rayDir,
    vec3 inverseRayDir, float maxT, bool anyHit, out float hitT, out vec3 hitNormal
) {
    vec3 blockRayOrigin = rayOrigin;
    vec3 blockRayDir = rayDir;
    uint rotation = modelData & 0x1FFu;
    uint quadCount = (modelData >> 25u) & 0x3Fu;
    ivec2 firstQuad = ivec2(int(bitfieldExtract(modelData, 9, 14) << 1u), int(bitfieldExtract(modelData, 23, 2)));
    rayOrigin -= vec3(0.5);
    _voxel_rotateBlockModelRay(rotation, rayOrigin, rayDir);
    rayOrigin += vec3(0.5);
    bool axisAligned = (modelData & 0x80000000u) != 0u;
    vec3 unusedOrigin = vec3(0.0);
    _voxel_rotateBlockModelRay(rotation, unusedOrigin, inverseRayDir);
    float minT;
    voxel_blockModelInterval(rayOrigin, inverseRayDir, minT, maxT);
    if (voxel_missesBlockModelBounds(material, blockRayDir, rayOrigin, inverseRayDir, minT, maxT)) {
        return false;
    }
    float previousT = minT;
    int previousQuad = int(quadCount);
    bool rejected = false;
    while (true) {
        float closestT = maxT;
        int closestQuad = -1;
        int closestAxis = -1;
        vec3 closestNormal;
        ivec2 texelCoord = firstQuad;
        for (uint quad = 0u; quad < quadCount; ++quad) {
            float candidateT = closestT;
            int axis = -1;
            vec3 normal;
            bool candidate;
            if (axisAligned) {
                _voxel_intersectBlockModelAxisAlignedQuad(rayOrigin, rayDir, inverseRayDir, texelCoord, candidateT, axis);
                candidate = axis >= 0;
            } else {
                candidate = _voxel_intersectBlockModelQuad(rayOrigin, rayDir, texelCoord, candidateT, normal);
            }
            if (candidate && candidateT >= minT &&
                (!rejected || candidateT > previousT || (candidateT == previousT && int(quad) < previousQuad))) {
                closestT = candidateT;
                closestQuad = int(quad);
                closestAxis = axis;
                if (!axisAligned) closestNormal = normal;
            }
            texelCoord.x += 2;
        }
        if (closestQuad < 0) return false;
        vec3 normal;
        if (axisAligned) {
            normal = -sign(rayDir) * vec3(equal(ivec3(closestAxis), ivec3(0, 1, 2)));
        } else {
            normal = normalize(closestNormal);
            if (dot(normal, rayDir) > 0.0) normal = -normal;
        }
        normal = _voxel_unrotateBlockModelVector(rotation, normal);
        if (voxel_alphaTest(material, fma(blockRayDir, vec3(closestT), blockRayOrigin), normal)) {
            hitT = closestT;
            hitNormal = normal;
            return true;
        }
        rejected = true;
        previousT = closestT;
        previousQuad = closestQuad;
    }
}
#else
bool voxel_intersectAlphaTestedBlockModel(
    uint modelData, uint material, vec3 rayOrigin, vec3 rayDir,
    vec3 inverseRayDir, float maxT, bool anyHit, out float hitT, out vec3 hitNormal
) {
    vec3 blockRayOrigin = rayOrigin;
    vec3 blockRayDir = rayDir;
    uint rotation = modelData & 0x1FFu;
    uint quadCount = (modelData >> 25u) & 0x3Fu;
    ivec2 texelCoord = ivec2(int(bitfieldExtract(modelData, 9, 14) << 1u), int(bitfieldExtract(modelData, 23, 2)));
    rayOrigin -= vec3(0.5);
    _voxel_rotateBlockModelRay(rotation, rayOrigin, rayDir);
    rayOrigin += vec3(0.5);
    vec3 unusedOrigin = vec3(0.0);
    _voxel_rotateBlockModelRay(rotation, unusedOrigin, inverseRayDir);
    float minT;
    voxel_blockModelInterval(rayOrigin, inverseRayDir, minT, maxT);
    if (voxel_missesBlockModelBounds(material, blockRayDir, rayOrigin, inverseRayDir, minT, maxT)) return false;
    hitT = maxT;
    bool hit = false;
    if ((modelData & 0x80000000u) != 0u) {
        while (quadCount != 0u) {
            float candidateT = hitT;
            int axis = -1;
            _voxel_intersectBlockModelAxisAlignedQuad(rayOrigin, rayDir, inverseRayDir, texelCoord, candidateT, axis);
            if (axis >= 0 && candidateT >= minT) {
                uint worldAxis = (rotation >> (uint(axis) * 3u)) & 3u;
                uint face = worldAxis * 2u + uint(blockRayDir[worldAxis] > 0.0);
                if (voxel_alphaTestFace(material, fma(blockRayDir, vec3(candidateT), blockRayOrigin), face)) {
                    hitT = candidateT;
                    hitNormal = -sign(blockRayDir) * vec3(equal(uvec3(worldAxis), uvec3(0, 1, 2)));
                    hit = true;
                    if (anyHit) return true;
                }
            }
            texelCoord.x += 2;
            --quadCount;
        }
    } else {
        while (quadCount != 0u) {
            float candidateT = hitT;
            vec3 normal;
            if (_voxel_intersectBlockModelQuad(rayOrigin, rayDir, texelCoord, candidateT, normal) && candidateT >= minT) {
                normal = normalize(normal);
                if (dot(normal, rayDir) > 0.0) normal = -normal;
                normal = _voxel_unrotateBlockModelVector(rotation, normal);
                if (voxel_alphaTest(material, fma(blockRayDir, vec3(candidateT), blockRayOrigin), normal)) {
                    hitT = candidateT;
                    hitNormal = normal;
                    hit = true;
                    if (anyHit) return true;
                }
            }
            texelCoord.x += 2;
            --quadCount;
        }
    }
    return hit;
}
#endif

#endif
