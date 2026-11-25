#ifndef INCLUDE_techniques_gi_ObservedBlockVisibility_glsl
#define INCLUDE_techniques_gi_ObservedBlockVisibility_glsl a

#include "/techniques/voxel/VoxelAlphaTest.glsl"
#include "/util/HardcodedPBR.glsl"

bool gi_intersectObservedModel(uint material, vec3 origin, vec3 dir, vec3 invDir, uint faceMask,
                               float endT, bool anyHit, out float t, out vec3 n) {
    HardcodedPBR flags = hardcodedpbr_decode(material);
    if (flags.isFullCube) return voxel_intersectAlphaTestedCube(material, origin, dir, invDir, endT, t, n);
    if (!flags.isKnown || material == MATERIAL_ID_WATER || material >= uint(textureSize(usam_pbrLUT2, 0).y)) return false;
    uint model = texelFetch(usam_pbrLUT2, ivec2(faceMask, material), 0).r;
    return model != 0u && voxel_intersectAlphaTestedBlockModel(model, material, origin, dir, invDir, endT, anyHit, t, n);
}

uint gi_modelFaceMask(vec3 dir) {
    return uint(42 - int(dir.x < 0.0) - 4 * int(dir.y < 0.0) - 16 * int(dir.z < 0.0));
}

bool gi_sourceContact(vec3 p, vec3 dir, vec2 span, vec3 sampled, uint material, vec3 n,
                      inout ivec3 endpoint, out float t, out vec3 normal) {
    // A different part of an extended face is eligible only when that cell
    // independently supplied the very same material during this observation.
    bool same = false;
    if (uint(endpoint.x | endpoint.y | endpoint.z) < 32u) {
        uvec2 cell = global_giObservedCells[endpoint.x + 32 * (endpoint.y + 32 * endpoint.z)];
        same = all(equal(cell, uvec2(material, global_giModelEpoch)));
    }
    if (!same) {
        endpoint = ivec3(floor(sampled - n * 0.0625));
        if (uint(endpoint.x | endpoint.y | endpoint.z) < 32u) {
            uvec2 owner = global_giObservedCells[endpoint.x + 32 * (endpoint.y + 32 * endpoint.z)];
            if (owner.y == global_giModelEpoch && owner.x != material) return false;
        }
    }
    vec3 invDir = 1.0 / mix(dir, vec3(1e-15), equal(dir, vec3(0.0)));
    // The cube helper normally runs inside a DDA cell. Here the candidate
    // came from a pixel, so establish that same precondition explicitly.
    float enterT, exitT = span.y;
    voxel_blockModelInterval(p - vec3(endpoint), invDir, enterT, exitT);
    if (exitT < max(enterT, span.x)) return false;
    return gi_intersectObservedModel(material, p - vec3(endpoint), dir, invDir,
                                    gi_modelFaceMask(dir), span.y, false, t, normal)
        && t > 0.0 && t >= span.x;
}

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
    uint faceMask = gi_modelFaceMask(dir);
    for (int i = 0; i < 96; ++i) {
        if (all(equal(cell, endpoint))) break;
        uvec2 sampleData = global_giObservedCells[cell.x + 32 * (cell.y + 32 * cell.z)];
        if (sampleData.y == global_giModelEpoch && any(notEqual(cell, origin))) {
            uint material = sampleData.x;
            float t;
            vec3 n;
            if (gi_intersectObservedModel(material, p - vec3(cell), dir, inverseDir, faceMask, endT, true, t, n)) return true;
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
