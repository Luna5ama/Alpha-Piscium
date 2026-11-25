#ifndef INCLUDE_techniques_gi_ObservedBlockVisibility_glsl
#define INCLUDE_techniques_gi_ObservedBlockVisibility_glsl a

#include "/techniques/voxel/VoxelAlphaTest.glsl"
#include "/util/HardcodedPBR.glsl"

// Cross the observed 32-block neighbourhood in original ray units. Only a
// witnessed material and its shipped block geometry can add an occluder.
bool gi_observedBlockOccludes(vec3 p, vec3 dir, float maxT, ivec3 endpoint) {
    vec3 inverseDir = 1.0 / mix(dir, vec3(1e-15), equal(dir, vec3(0.0)));
    vec3 lo = -p * inverseDir;
    vec3 hi = (32.0 - p) * inverseDir;
    vec3 enter = min(lo, hi);
    vec3 leave = max(lo, hi);
    float startT = max(0.0, max(enter.x, max(enter.y, enter.z)));
    float endT = min(maxT, min(leave.x, min(leave.y, leave.z)));
    if (endT <= startT) return false;
    ivec3 cell = clamp(ivec3(floor(fma(dir, vec3(startT + 0.0001), p))), ivec3(0), ivec3(31));
    ivec3 origin = ivec3(floor(p));
    ivec3 delta = ivec3(sign(dir));
    ivec3 side = ivec3(greaterThan(dir, vec3(0.0)));
    uint faceMask = uint(42 - int(dir.x < 0.0) - 4 * int(dir.y < 0.0) - 16 * int(dir.z < 0.0));
    for (int i = 0; i < 96; ++i) {
        if (all(equal(cell, endpoint))) break;
        uvec2 sampleData = global_giObservedCells[cell.x + 32 * (cell.y + 32 * cell.z)];
        if (sampleData.y == global_giModelEpoch && any(notEqual(cell, origin))) {
            uint material = sampleData.x;
            HardcodedPBR flags = hardcodedpbr_decode(material);
            vec3 rayOrigin = p - vec3(cell);
            float t;
            vec3 n;
            if (flags.isFullCube) {
                if (voxel_intersectAlphaTestedCube(material, rayOrigin, dir, inverseDir, endT, t, n)) return true;
            } else if (flags.isKnown && material != MATERIAL_ID_WATER && material < uint(textureSize(usam_pbrLUT2, 0).y)) {
                uint model = texelFetch(usam_pbrLUT2, ivec2(faceMask, material), 0).r;
                if (model != 0u && voxel_intersectAlphaTestedBlockModel(model, material, rayOrigin, dir, inverseDir, endT, true, t, n)) return true;
            }
        }
        vec3 next = (vec3(cell + side) - p) * inverseDir;
        next = mix(next, vec3(1e30), equal(dir, vec3(0.0)));
        float edge = min(next.x, min(next.y, next.z));
        if (edge >= endT) break;
        cell += delta * ivec3(lessThanEqual(next, vec3(edge)));
        if (uint(cell.x | cell.y | cell.z) >= 32u) break;
    }
    return false;
}

#endif
