#ifndef INCLUDE_debug_LightSampling_glsl
#define INCLUDE_debug_LightSampling_glsl a
#include "/Base.glsl"
#include "/techniques/voxel/Voxelization.glsl"
#include "/util/HardcodedPBR.glsl"
#include "/util/Hash.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Colors.glsl"
#include "/util/Coords.glsl"
#define LS_TREE_ARITY SETTING_DEBUG_LIGHT_ARITY
#if LS_TREE_ARITY == 2
#define LS_TREE_BITS 1
#elif LS_TREE_ARITY == 4
#define LS_TREE_BITS 2
#else
#define LS_TREE_BITS 3
#endif
#include "LightSamplingLayout.glsl"

const uint LS_INVALID = 0xffffffffu;
layout(std430, binding = 14) buffer LightSamplingAudit {
    vec4 ls_audit[256];
    uint ls_reasons[16];
};

const uint LS_LOCAL_NODE_CAPACITY = 262144u;
const uint LS_LEAF_BIT = 0x80000000u;
struct LightSamplingNode {
    vec4 boundsMinMass;
    vec4 boundsMaxLink;
    #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 4 || SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 5
    vec4 meanVariance;
    #endif
};
layout(std430, binding = 12) buffer LightSamplingData {
    uint ls_allocatedNodes;
    uint ls_emitterCount;
    uint ls_overflowCount;
    uint ls_exhaustedCount;
    uint ls_traceCount;
    uint ls_visibleCount;
    uint ls_receiverCount;
    uint ls_reserved;
    LightSamplingNode ls_nodes[];
};

// Internal links pack the child base in bits 0..27 and count - 1 in bits 28..30.
uint ls_childBase(uint link) {
    return link & 0x0fffffffu;
}

uint ls_childCount(uint link) {
    return ((link >> 28u) & 7u) + 1u;
}

uint ls_childLink(uint base, uint count) {
    return base | ((count - 1u) << 28u);
}

bool ls_isEmitter(uint materialID) {
    return materialID > 1u && hardcodedpbr_decode(materialID).emissive > 0.0;
}

LightSamplingNode ls_emptyNode() {
    LightSamplingNode node;
    node.boundsMinMass = vec4(0.0);
    node.boundsMaxLink = vec4(0.0);
    #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 4 || SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 5
    node.meanVariance = vec4(0.0);
    #endif
    return node;
}

LightSamplingNode ls_merge(LightSamplingNode a, LightSamplingNode b, uint child) {
    #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 4 || SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 5
    // Copy empty merges exactly: reciprocal error must not multiply world-position squared.
    if (a.boundsMinMass.w == 0.0) { b.boundsMaxLink.w = uintBitsToFloat(child); return b; }
    if (b.boundsMinMass.w == 0.0) { a.boundsMaxLink.w = uintBitsToFloat(child); return a; }
    #else
    if (a.boundsMinMass.w == 0.0) a.boundsMinMass.xyz = b.boundsMinMass.xyz;
    if (b.boundsMinMass.w == 0.0) b.boundsMinMass.xyz = a.boundsMinMass.xyz;
    if (a.boundsMinMass.w == 0.0) a.boundsMaxLink.xyz = b.boundsMaxLink.xyz;
    if (b.boundsMinMass.w == 0.0) b.boundsMaxLink.xyz = a.boundsMaxLink.xyz;
    #endif
    LightSamplingNode merged = ls_emptyNode();
    merged.boundsMinMass = vec4(min(a.boundsMinMass.xyz, b.boundsMinMass.xyz), a.boundsMinMass.w + b.boundsMinMass.w);
    merged.boundsMaxLink = vec4(max(a.boundsMaxLink.xyz, b.boundsMaxLink.xyz), uintBitsToFloat(child));
    #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 4 || SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 5
    float mass = merged.boundsMinMass.w;
    if (mass > 0.0) {
        float wa = a.boundsMinMass.w / mass;
        float wb = b.boundsMinMass.w / mass;
        vec3 delta = b.meanVariance.xyz - a.meanVariance.xyz;
        vec3 mean = a.boundsMinMass.w >= b.boundsMinMass.w
            ? a.meanVariance.xyz + wb * delta : b.meanVariance.xyz - wa * delta;
        merged.meanVariance = vec4(mean,
            wa * a.meanVariance.w + wb * b.meanVariance.w + wa * wb * dot(delta, delta));
    }
    #endif
    return merged;
}

float ls_random(inout uint state) {
    state = hash_pcg_11(state);
    return (float(state >> 9u) + 0.5) * (1.0 / 8388608.0);
}

bool ls_loadReceiver(ivec2 pixel, out vec3 origin, out vec3 normal, out vec3 albedo) {
    float viewZ = texelFetch(usam_gbufferSolidViewZ, pixel, 0).r;
    if (viewZ <= -65536.0) return false;
    GBufferData data = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, pixel, 0), data);
    gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, pixel, 0), data);
    if (ls_isEmitter(data.materialID)) return false;
    vec2 screenPos = coords_texelToUV(pixel, uval_mainImageSizeRcp) - uval_taaJitterUV;
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    normal = normalize(coords_dir_viewToWorld(data.geomNormal));
    origin = coords_pos_viewToWorld(viewPos, gbufferModelViewInverse) + cameraPosition + normal * 0.003;
    albedo = colors_material_toWorkSpace(data.albedo);
    return true;
}

// Cosine-weighted SG integral divided by 2*pi (Tokuyoshi et al. 2024, corrected fit).
float ls_sgCosine(float k, float c) {
    float front, back;
    if (k < 0.05) {
        front = 0.5 + k * (-1.0 / 6.0 + k * (1.0 / 24.0 - k / 120.0));
        back = 0.5 + k * (-5.0 / 6.0 + k * (17.0 / 24.0 - k * 49.0 / 120.0));
    } else {
        float e = exp(-k);
        front = (k - 1.0 + e) / (k * k);
        back = e * (1.0 - (1.0 + k) * e) / (k * k);
    }
    float t = k * sqrt(0.5 * (k * k + 2.7360833 * k + 17.021297) /
        (k * k * k + 4.0100827 * k * k + 15.219156 * k + 76.087896));
    float u;
    if (t < 0.01) {
        u = 0.5 * (c + 1.0) + t * (c * c - 1.0) * 0.2820947918;
    } else {
        // A&S 7.1.26 erfc for |t*c| and t; endpoint normalization is exactly 2*t.
        vec2 v = vec2(abs(t * c), t);
        vec2 r = 1.0 / (1.0 + 0.3275911 * v);
        vec2 erfcV = (((((1.061405429 * r - 1.453152027) * r) + 1.421413741) * r
            - 0.284496736) * r + 0.254829592) * r * exp(-v * v);
        vec2 q = exp(-v * v) * 0.5641895835 - v * erfcV;
        u = clamp((2.0 * max(t * c, 0.0) + q.x - q.y) / (2.0 * t), 0.0, 1.0);
    }
    return max(mix(back, front, u), 0.000001 / (1.0 + k));
}

bool ls_behindReceiver(vec3 delta, vec3 halfSize, vec3 normal) {
    float extent = dot(abs(normal), halfSize);
    return dot(normal, delta) + extent < -0.000001 * (dot(abs(normal), abs(delta)) + extent + 1.0);
}

float ls_geometricWeight(LightSamplingNode node, vec3 origin, vec3 normal) {
    if (node.boundsMinMass.w == 0.0) return 0.0;
    vec3 halfSize = (node.boundsMaxLink.xyz - node.boundsMinMass.xyz) * 0.5;
    vec3 delta = node.boundsMinMass.xyz + halfSize - origin;
    float radius = length(halfSize);
    #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 4 || SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 5
    delta = node.meanVariance.xyz - origin;
    radius = sqrt(2.0 * node.meanVariance.w);
    #endif
    float distanceSquared = dot(delta, delta);
    #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 5
    float k = distanceSquared / node.meanVariance.w;
    float c = distanceSquared > 0.0 ? clamp(dot(normal, delta) * inversesqrt(distanceSquared), -1.0, 1.0) : 0.0;
    return node.boundsMinMass.w / node.meanVariance.w * ls_sgCosine(k, c);
    #else
    float height = dot(normal, delta);
    #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 1 || SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 2 || SETTING_DEBUG_LIGHT_WEIGHT_MODEL >= 7
    if (ls_behindReceiver(delta, halfSize, normal)) return 0.0;
    #endif
    #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 2
    float extent = dot(abs(normal), halfSize);
    float clippedHeight = height >= extent ? height : max(height + extent, 0.0) * max(height + extent, 0.0) / (4.0 * extent);
    float cosineBound = clamp(clippedHeight * inversesqrt(max(distanceSquared, 0.25)), 0.001, 1.0);
    #else
    float cosineBound = clamp((height + radius) * inversesqrt(max(distanceSquared, 0.25)), 0.001, 1.0);
    #endif
    return node.boundsMinMass.w * cosineBound / max(distanceSquared, max(radius * radius, 0.25));
    #endif
}

#if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 3 || SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 6 || SETTING_DEBUG_LIGHT_WEIGHT_MODEL >= 11
#define LS_NEAR_FAR

// Scores are summed over representatives before normalization over siblings.
vec3 ls_distanceScores(float amplitude, vec3 delta, vec3 halfSize) {
    if (amplitude == 0.0) return vec3(0.0);
    vec3 nearest = max(abs(delta) - halfSize, 0.0);
    vec3 farthest = abs(delta) + halfSize;
    float near2 = dot(nearest, nearest);
    return vec3(near2 > 0.0 ? amplitude / near2 : 0.0,
        amplitude / dot(farthest, farthest), near2 == 0.0 ? amplitude : 0.0);
}

vec3 ls_nativeScores(LightSamplingNode node, vec3 origin, vec3 normal) {
    vec3 halfSize = (node.boundsMaxLink.xyz - node.boundsMinMass.xyz) * 0.5;
    vec3 delta = node.boundsMinMass.xyz + halfSize - origin;
    #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL != 3
    if (ls_behindReceiver(delta, halfSize, normal)) return vec3(0.0);
    #endif
    float cosine = clamp((dot(normal, delta) + length(halfSize)) *
        inversesqrt(max(dot(delta, delta), 0.25)), 0.001, 1.0);
    return ls_distanceScores(node.boundsMinMass.w * cosine, delta, halfSize);
}
#endif

#if SETTING_DEBUG_LIGHT_WEIGHT_MODEL >= 7
bool ls_crossesBrick(LightSamplingNode node) {
    return any(notEqual(ivec3(node.boundsMinMass.xyz) >> 4, ivec3(node.boundsMaxLink.xyz - 1.0) >> 4));
}

#if SETTING_DEBUG_LIGHT_WEIGHT_MODEL != 7 && SETTING_DEBUG_LIGHT_WEIGHT_MODEL != 11
#define LS_WARD
uniform sampler2D usam_lsClusterAtlas;
const int LS_CLUSTER_SCENE_ROW = (SETTING_DEBUG_LIGHT_CLUSTER_SCENE - 1) * 23;

vec3 ls_clusterScores(vec4 centerVariance, float mass, vec3 origin, vec3 normal, vec3 lo, vec3 hi) {
    vec3 delta = centerVariance.xyz - origin;
    float distance2 = dot(delta, delta);
    float radius2 = 2.0 * centerVariance.w;
    float cosine = clamp((dot(normal, delta) + sqrt(radius2)) * inversesqrt(max(distance2, 0.25)), 0.001, 1.0);
    #ifdef LS_NEAR_FAR
    vec3 halfSize = (hi - lo) * 0.5;
    return ls_distanceScores(mass * cosine, lo + halfSize - origin, halfSize);
    #else
    return vec3(mass * cosine / max(distance2, max(radius2, 0.25)), 0.0, 0.0);
    #endif
}
#endif

vec3 ls_experimentScores(LightSamplingNode node, uint nodeIndex, vec3 origin, vec3 normal) {
    uint link = floatBitsToUint(node.boundsMaxLink.w);
    if (node.boundsMinMass.w == 0.0 || (link & LS_LEAF_BIT) != 0u || !ls_crossesBrick(node)) {
        #ifdef LS_NEAR_FAR
        return ls_nativeScores(node, origin, normal);
        #else
        return vec3(ls_geometricWeight(node, origin, normal), 0.0, 0.0);
        #endif
    }
    #ifndef LS_WARD
    vec3 scores = vec3(0.0);
    uint base = ls_childBase(link);
    for (uint i = 0u; i < ls_childCount(link); ++i) {
        #ifdef LS_NEAR_FAR
        scores += ls_nativeScores(ls_nodes[base + i], origin, normal);
        #else
        scores.x += ls_geometricWeight(ls_nodes[base + i], origin, normal);
        #endif
    }
    return scores;
    #else
    if (nodeIndex >= 7620u) return vec3(uintBitsToFloat(0x7fc00000u));
    ivec2 slot = ivec2(int(nodeIndex), LS_CLUSTER_SCENE_ROW);
    vec4 lowMass = texelFetch(usam_lsClusterAtlas, slot, 0);
    vec3 high = texelFetch(usam_lsClusterAtlas, slot + ivec2(0, 1), 0).xyz;
    if (any(notEqual(lowMass.xyz, node.boundsMinMass.xyz)) || any(notEqual(high, node.boundsMaxLink.xyz)) ||
        abs(lowMass.w - node.boundsMinMass.w) > 0.000002 * node.boundsMinMass.w) {
        return vec3(uintBitsToFloat(0x7fc00000u));
    }
    vec3 halfSize = (node.boundsMaxLink.xyz - node.boundsMinMass.xyz) * 0.5;
    if (ls_behindReceiver(node.boundsMinMass.xyz + halfSize - origin, halfSize, normal)) return vec3(0.0);
    #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 8 || SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 12
    vec4 centerVariance = texelFetch(usam_lsClusterAtlas, slot + ivec2(0, 2), 0);
    return ls_clusterScores(centerVariance, node.boundsMinMass.w, origin, normal, lowMass.xyz, high);
    #else
    #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 9 || SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 13
    const int count = 2;
    const int first = 3;
    const int massRow = 9;
    #else
    const int count = 4;
    const int first = 5;
    const int massRow = 10;
    #endif
    vec4 masses = texelFetch(usam_lsClusterAtlas, slot + ivec2(0, massRow), 0);
    vec3 scores = vec3(0.0);
    for (int j = 0; j < count; ++j) {
        if (masses[j] > 0.0) {
            vec4 centerVariance = texelFetch(usam_lsClusterAtlas, slot + ivec2(0, first + j), 0);
            #ifdef LS_NEAR_FAR
            int boundRow = 11 + 2 * (first + j - 3);
            vec3 lo = texelFetch(usam_lsClusterAtlas, slot + ivec2(0, boundRow), 0).xyz;
            vec3 hi = texelFetch(usam_lsClusterAtlas, slot + ivec2(0, boundRow + 1), 0).xyz;
            #else
            vec3 lo = vec3(0.0), hi = vec3(0.0);
            #endif
            scores += ls_clusterScores(centerVariance, masses[j], origin, normal, lo, hi);
        }
    }
    return scores;
    #endif
    #endif
}
#endif

LightSamplingNode ls_select(vec3 origin, vec3 normal, inout uint rng, uint auditSlot, out float pdf) {
    LightSamplingNode node = ls_nodes[1];
    pdf = 1.0;
    #ifdef LS_WARD
    if (LS_TREE_ARITY != 4 || VOXEL_GRID_SIZE != 16 ||
        float(ls_emitterCount) != texelFetch(usam_lsClusterAtlas, ivec2(0, LS_CLUSTER_SCENE_ROW + 1), 0).w) {
        pdf = uintBitsToFloat(0x7fc00000u);
        return ls_emptyNode();
    }
    #endif
    while ((floatBitsToUint(node.boundsMaxLink.w) & LS_LEAF_BIT) == 0u && node.boundsMinMass.w > 0.0) {
        uint link = floatBitsToUint(node.boundsMaxLink.w);
        uint child = ls_childBase(link);
        uint count = ls_childCount(link);
        vec4 weightsLow = vec4(0.0);
        #if LS_TREE_ARITY == 8
        vec4 weightsHigh = vec4(0.0);
        #endif
        #ifdef LS_NEAR_FAR
        vec4 nearLow = vec4(0.0), zeroLow = vec4(0.0);
        #if LS_TREE_ARITY == 8
        vec4 nearHigh = vec4(0.0), zeroHigh = vec4(0.0);
        #endif
        float nearTotal = 0.0, zeroTotal = 0.0;
        #endif
        float total = 0.0;
        for (uint i = 0u; i < count; ++i) {
            LightSamplingNode candidate = ls_nodes[child + i];
            #if SETTING_DEBUG_LIGHT_WEIGHT_MODEL >= 7
            vec3 scores = ls_experimentScores(candidate, child + i, origin, normal);
            if (isnan(scores.x)) {
                pdf = scores.x;
                return ls_emptyNode();
            }
            #elif defined(LS_NEAR_FAR)
            vec3 scores = ls_nativeScores(candidate, origin, normal);
            #endif
            #ifdef LS_NEAR_FAR
            float nearWeight = scores.x, weight = scores.y, zeroWeight = scores.z;
            nearTotal += nearWeight;
            zeroTotal += zeroWeight;
            #if LS_TREE_ARITY == 8
            if (i >= 4u) { nearHigh[i - 4u] = nearWeight; zeroHigh[i - 4u] = zeroWeight; }
            else
            #endif
            { nearLow[i] = nearWeight; zeroLow[i] = zeroWeight; }
            #elif SETTING_DEBUG_LIGHT_WEIGHT_MODEL >= 7
            float weight = scores.x;
            #else
            float weight = ls_geometricWeight(candidate, origin, normal);
            #endif
            #if LS_TREE_ARITY == 8
            if (i >= 4u) weightsHigh[i - 4u] = weight;
            else
            #endif
            weightsLow[i] = weight;
            total += weight;
        }
        if (total == 0.0) { pdf = 0.0; return ls_emptyNode(); }
        #ifdef LS_NEAR_FAR
        weightsLow = 0.5 * (weightsLow / total + (zeroTotal > 0.0 ? zeroLow / zeroTotal : nearLow / nearTotal));
        #if LS_TREE_ARITY == 8
        weightsHigh = 0.5 * (weightsHigh / total + (zeroTotal > 0.0 ? zeroHigh / zeroTotal : nearHigh / nearTotal));
        #endif
        total = 0.0;
        for (uint i = 0u; i < count; ++i) {
            #if LS_TREE_ARITY == 8
            total += i < 4u ? weightsLow[i] : weightsHigh[i - 4u];
            #else
            total += weightsLow[i];
            #endif
        }
        #endif
        float threshold = ls_random(rng) * total;
        uint selected = 0u;
        float selectedWeight = weightsLow.x;
        float cumulative = selectedWeight;
        while (threshold >= cumulative) {
            ++selected;
            #if LS_TREE_ARITY == 8
            selectedWeight = selected < 4u ? weightsLow[selected] : weightsHigh[selected - 4u];
            #else
            selectedWeight = weightsLow[selected];
            #endif
            cumulative += selectedWeight;
        }
        pdf *= selectedWeight / total;
        node = ls_nodes[child + selected];
    }
    #ifdef SETTING_DEBUG_LIGHT_AUDIT
    if (auditSlot != LS_INVALID) {
        uint offset = auditSlot * 4u;
        ls_audit[offset] = vec4(origin, node.boundsMaxLink.w);
        ls_audit[offset + 1u] = vec4(normal, pdf);
        ls_audit[offset + 3u] = vec4(node.boundsMinMass.xyz, 1.0);
    }
    #endif
    return node;
}
#endif
