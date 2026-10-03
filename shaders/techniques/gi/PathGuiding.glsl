#ifndef INCLUDE_techniques_gi_PathGuiding_glsl
#define INCLUDE_techniques_gi_PathGuiding_glsl a

#include "/util/Coords.glsl"
#include "/util/Math.glsl"

// Von Mises-Fisher lobe that guides diffuse initial GI candidates and RC fresh rays. It is folded about the
// sampling-frame tangent plane: a direction below the plane is mirrored above it, so the density on the upper
// hemisphere is g(mu, w) + g(mu', w) with mu' = mu mirrored.
struct PathGuideLobe {
    vec3 axisView;
    float kappa;
    // Fraction of the diffuse technique budget given to the lobe; 0 disables it.
    float alpha;
};

// Two lobes per RC face, each fitted to one online cluster of the training directions.
struct PathGuide {
    PathGuideLobe lobe0;
    PathGuideLobe lobe1;
};

// Training record flags written by GIReSTIRPairedSpatialShade (x holds the oct32 world direction).
#define PG_RECORD_VALID 1u
// The final sample was taken from a spatial neighbor.
#define PG_RECORD_NEIGHBOR 2u
// The final sample left the geometry (sky).
#define PG_RECORD_SKY 4u
// The sample was rejected because it was attributed to the specular lobe.
#define PG_RECORD_SPECULAR 8u

PathGuideLobe pathGuideLobe_none() {
    PathGuideLobe lobe;
    lobe.axisView = vec3(0.0, 0.0, 1.0);
    lobe.kappa = 0.0;
    lobe.alpha = 0.0;
    return lobe;
}

PathGuide pathGuide_none() {
    PathGuide guide;
    guide.lobe0 = pathGuideLobe_none();
    guide.lobe1 = pathGuideLobe_none();
    return guide;
}

float pathGuide_vmfPdf(vec3 axis, float kappa, vec3 dir) {
    return kappa / (PI_2 * (1.0 - exp(-2.0 * kappa))) * exp(kappa * (dot(axis, dir) - 1.0));
}

float pathGuide_foldedPdf(vec3 axisTangent, float kappa, vec3 dirTangent) {
    vec3 mirroredAxis = vec3(axisTangent.xy, -axisTangent.z);
    return pathGuide_vmfPdf(axisTangent, kappa, dirTangent) + pathGuide_vmfPdf(mirroredAxis, kappa, dirTangent);
}

vec3 pathGuide_sampleFolded(vec3 axisTangent, float kappa, vec2 xi) {
    float w = 1.0 + log(xi.x + (1.0 - xi.x) * exp(-2.0 * kappa)) / kappa;
    float v = sqrt(max(1.0 - w * w, 0.0));
    float phi = PI_2 * xi.y;
    vec3 up = abs(axisTangent.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, axisTangent));
    vec3 B = cross(axisTangent, T);
    vec3 dir = (T * cos(phi) + B * sin(phi)) * v + axisTangent * w;
    dir.z = abs(dir.z);
    return dir;
}

// Folded pdf of a lobe for a direction in the sampling frame.
float pathGuide_lobePdf(PathGuideLobe lobe, mat3 samplingTbnInv, vec3 dirTangent) {
    return pathGuide_foldedPdf(normalize(samplingTbnInv * lobe.axisView), lobe.kappa, dirTangent);
}

// Number of the 256 technique bins given to each lobe, taken from the diffuse bins (256 - specularBins).
uvec2 pathGuide_bins(PathGuide guide, float specularProbability) {
    float diffuseBins = 256.0 - specularProbability * 256.0;
    return uvec2(floor(vec2(guide.lobe0.alpha, guide.lobe1.alpha) * diffuseBins));
}

#if SETTING_DEBUG_PATH_GUIDE >= 1 && SETTING_DEBUG_PATH_GUIDE <= 4
// Fixed world-up lobes for validating the sampler and pdf without training.
PathGuide pathGuide_debugLobe() {
    PathGuide guide = pathGuide_none();
    guide.lobe0.axisView = normalize(mat3(gbufferModelView) * vec3(0.0, 1.0, 0.0));
    #if SETTING_DEBUG_PATH_GUIDE == 1
    guide.lobe0.kappa = 1.0;
    guide.lobe0.alpha = 0.5;
    #elif SETTING_DEBUG_PATH_GUIDE == 2
    guide.lobe0.kappa = 8.0;
    guide.lobe0.alpha = 0.5;
    #elif SETTING_DEBUG_PATH_GUIDE == 3
    guide.lobe0.kappa = 32.0;
    guide.lobe0.alpha = 0.5;
    #else
    guide.lobe0.kappa = 32.0;
    guide.lobe0.alpha = 0.25;
    #endif
    return guide;
}
#endif

#if defined(SETTING_GI_PATH_GUIDING) && defined(SETTING_RC_ENABLE)
#define PATH_GUIDING_ENABLED a
#include "/techniques/gi/RadianceCache.glsl"

#ifndef PG_DATA_MODIFIER
#define PG_DATA_MODIFIER restrict readonly buffer
#endif

// Per RC face slot and RC side, one record per lobe of: xyz = sum of world-space training directions (int32 as uint),
// w = sample count, both in fixed point with PG_FIXED_POINT_SCALE.
layout(std430, binding = 6) PG_DATA_MODIFIER PathGuideData {
    uvec4 pg_stats[];
};

const uint PG_STATS_PER_SLOT = 2u;
// A training direction further than this cosine from the first lobe seeds the empty second lobe.
const float PG_SPLIT_COSINE = 0.5;

const float PG_FIXED_POINT_SCALE = 1024.0;
// Caps the decayed sample count so that one frame of splats cannot overflow the int32 sums.
const float PG_MAX_SAMPLES = 256.0;
// Prior samples added to the concentration estimate so that few coherent samples do not give a sharp lobe.
const float PG_PRIOR_SAMPLES = 4.0;
const float PG_MIN_CONCENTRATION = 0.1;
const float PG_MIN_SAMPLES = 8.0;
const float PG_MAX_KAPPA = 32.0;
// Share of the diffuse technique bins given to a fitted lobe. Keeps q >= q_cosine / 2, so a mis-fitted lobe at most
// doubles the second moment of the diffuse estimate.
const float PG_ALPHA = 0.5;
// Per-frame decay of the statistics when they migrate to the current RC side.
const float PG_DECAY = 0.9;

uint pathGuide_statsIndex(uint side, uint slot) {
    return (side * uint(SETTING_RC_POOL_SIZE) + slot) * PG_STATS_PER_SLOT;
}

uvec4 pathGuide_decayStats(uvec4 stats) {
    vec4 decayed = vec4(vec3(ivec3(stats.xyz)), float(stats.w)) * PG_DECAY;
    decayed *= min(1.0, PG_MAX_SAMPLES * PG_FIXED_POINT_SCALE / max(decayed.w, 1.0));
    return uvec4(uvec3(ivec3(round(decayed.xyz))), uint(round(decayed.w)));
}

// Resolve by world identity because face slots are allocated again every frame.
void pathGuide_previousStats(uint entryIndex, uint worldKeyHash, uint level, uint faceId, out uvec4 stats0, out uvec4 stats1) {
    stats0 = uvec4(0u);
    stats1 = uvec4(0u);
    uvec4 entry = rc_indirection[rc_bufferEntryIndex(rc_previousSide(), entryIndex)];
    if (
        entry.x == RC_INVALID
        || entry.z != worldKeyHash
        || !rc_entryMetaValid(entry.w)
        || rc_entryMetaLevel(entry.w) != level
        || !rc_hasFace(entry.y, faceId)
    ) {
        return;
    }
    uint slot = rc_faceReservoirIndex(entry.x, entry.y, faceId);
    if (slot >= uint(SETTING_RC_POOL_SIZE)) {
        return;
    }
    uint statsIndex = pathGuide_statsIndex(rc_previousSide(), slot);
    stats0 = pathGuide_decayStats(pg_stats[statsIndex]);
    stats1 = pathGuide_decayStats(pg_stats[statsIndex + 1u]);
}

// Current-side RC slot of the face that owns a primary surface point, keyed like RadianceCacheTouch,
// at the level RC lookups use. RC_INVALID when the face has no slot this frame.
uint pathGuide_faceSlot(vec3 viewPos, vec3 geomNormalView) {
    vec3 worldPos = coords_pos_viewToWorld(viewPos - geomNormalView * 0.02, gbufferModelViewInverse) + cameraPosition;
    uint faceId = rc_faceIdFromNormal(coords_dir_viewToWorld(geomNormalView));
    uint level = rc_selectLevel(worldPos);
    ivec3 worldCellCoord = rc_worldCellCoord(worldPos, level);
    uvec4 entry = rc_indirection[rc_bufferEntryIndex(rc_currentSide(), rc_entryIndex(level, worldCellCoord))];
    if (
        entry.x == RC_INVALID
        || entry.z != rc_worldKeyHash(level, worldCellCoord)
        || !rc_entryMetaValid(entry.w)
        || rc_entryMetaLevel(entry.w) != level
        || !rc_hasFace(entry.y, faceId)
    ) {
        return RC_INVALID;
    }
    uint slot = rc_faceReservoirIndex(entry.x, entry.y, faceId);
    return slot < uint(SETTING_RC_POOL_SIZE) ? slot : RC_INVALID;
}

float pathGuide_statsSampleCount(uvec4 stats) {
    return float(stats.w) * (1.0 / PG_FIXED_POINT_SCALE);
}

// Maximum-likelihood fit (Banerjee et al. 2005 kappa approximation) of one record's directions; alpha is 0 when the
// record has too few or too spread samples.
PathGuideLobe pathGuide_fitLobe(uvec4 stats) {
    PathGuideLobe lobe = pathGuideLobe_none();
    vec3 directionSum = vec3(ivec3(stats.xyz)) * (1.0 / PG_FIXED_POINT_SCALE);
    float sampleCount = pathGuide_statsSampleCount(stats);
    float sumLength = length(directionSum);
    float r = sumLength / (sampleCount + PG_PRIOR_SAMPLES);
    if (sampleCount >= PG_MIN_SAMPLES && r >= PG_MIN_CONCENTRATION) {
        lobe.axisView = mat3(gbufferModelView) * (directionSum / sumLength);
        lobe.kappa = min(r * (3.0 - r * r) / (1.0 - r * r), PG_MAX_KAPPA);
        lobe.alpha = PG_ALPHA;
    }
    return lobe;
}

PathGuide pathGuide_fit(uvec4 stats0, uvec4 stats1) {
    PathGuide guide = pathGuide_none();
    guide.lobe0 = pathGuide_fitLobe(stats0);
    guide.lobe1 = pathGuide_fitLobe(stats1);
    if (guide.lobe0.alpha > 0.0 && guide.lobe1.alpha > 0.0) {
        // Split the guide budget by sample count.
        float count0 = pathGuide_statsSampleCount(stats0);
        float count1 = pathGuide_statsSampleCount(stats1);
        guide.lobe0.alpha = PG_ALPHA * count0 / (count0 + count1);
        guide.lobe1.alpha = PG_ALPHA - guide.lobe0.alpha;
    }
    return guide;
}

PathGuide pathGuide_load(uint slot) {
    uint statsIndex = pathGuide_statsIndex(rc_currentSide(), slot);
    return pathGuide_fit(pg_stats[statsIndex], pg_stats[statsIndex + 1u]);
}

// Record that a training direction is added to: the nearer of the two online clusters, seeding the second one with
// the first direction that is far from the first cluster.
uint pathGuide_trainingRecord(vec3 directionWorld, uvec4 stats0, uvec4 stats1) {
    if (stats0.w == 0u) {
        return 0u;
    }
    float cosine0 = dot(directionWorld, normalize(vec3(ivec3(stats0.xyz))));
    if (stats1.w == 0u) {
        return cosine0 < PG_SPLIT_COSINE ? 1u : 0u;
    }
    float cosine1 = dot(directionWorld, normalize(vec3(ivec3(stats1.xyz))));
    return cosine1 > cosine0 ? 1u : 0u;
}
#endif

#endif
