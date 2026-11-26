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

#if defined(RESTIR_INITIAL_CANDIDATE_WRITE) && SETTING_PBR_MATERIAL == 0
// The visible area can shrink to one dark texel just before a lamp leaves
// the screen. Give each observed surface region equal area in the integral.
uint gi_observedEmission(ivec3 cell) {
    int index = cell.x + 32 * (cell.y + 32 * cell.z);
    uint material = global_giObservedCells[index].x;
    vec3 radiance = vec3(0);
    int count = 0;
    for (int tile = 0; tile < 64; ++tile) {
        uvec2 sampleData = global_giLightSamples[index * 64 + tile];
        if (sampleData.y == global_giModelEpoch && global_giLightSampleMaterial[index * 64 + tile] == material) {
            radiance += colors_FP16LuvToWorkingColor(sampleData.x);
            ++count;
        }
    }
    return colors_workingColorToFP16Luv(radiance / float(max(count, 1)));
}

// Look for actual model contacts with native built-in emitters observed at any
// zoom level. The group stamps skip empty 4x4x4 regions in the local cache.
bool gi_observedSource(vec3 p, vec3 dir, vec2 span, out ivec3 source, out float t, out vec3 n) {
    vec3 invDir = 1.0 / mix(dir, vec3(1e-15), equal(dir, vec3(0.0)));
    vec3 lo = -p * invDir;
    vec3 hi = (32.0 - p) * invDir;
    vec3 entry = min(lo, hi);
    vec3 leave = max(lo, hi);
    float startT = max(span.x, max(entry.x, max(entry.y, entry.z)));
    float endT = min(span.y, min(leave.x, min(leave.y, leave.z)));
    if (endT <= startT) return false;
    ivec3 cell = clamp(ivec3(floor(fma(dir, vec3(startT + 0.0001), p))), ivec3(0), ivec3(31));
    ivec3 side = ivec3(greaterThan(dir, vec3(0.0)));
    ivec3 origin = ivec3(floor(p));
    for (int i = 0; i < 96; ++i) {
        ivec3 group = cell >> 2;
        bool scan = global_giLightGroupEpoch[group.x + 8 * (group.y + 8 * group.z)] == global_giModelEpoch;
        if (scan && any(notEqual(cell, origin))) {
            int index = cell.x + 32 * (cell.y + 32 * cell.z);
            uvec2 record = global_giObservedCells[index];
            if (record.y == global_giModelEpoch && hardcodedpbr_decode(record.x).emissive > 0.0
                && gi_intersectObservedModel(record.x, p - vec3(cell), dir, invDir, gi_modelFaceMask(dir), endT, false, t, n)
                && t > 0.0 && t >= span.x) {
                source = cell;
                return true;
            }
        }
        int size = scan ? 1 : 4;
        ivec3 lower = cell & -size;
        vec3 next = (vec3(lower + size * side) - p) * invDir;
        next = mix(next, vec3(1e30), equal(dir, vec3(0.0)));
        float edge = min(next.x, min(next.y, next.z));
        if (edge >= endT) break;
        ivec3 crossing = ivec3(floor(fma(dir, vec3(edge), p)));
        cell = mix(clamp(crossing, lower, lower + size - 1),
                   lower + side * (size + 1) - 1, lessThanEqual(next, vec3(edge)));
        if (uint(cell.x | cell.y | cell.z) >= 32u) break;
    }
    return false;
}
#endif

#endif
