#ifndef INCLUDE_debug_LightSampling_glsl
#define INCLUDE_debug_LightSampling_glsl a
#include "/Base.glsl"
#include "/techniques/voxel/Voxelization.glsl"
#include "/util/HardcodedPBR.glsl"
#include "/util/Hash.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Colors.glsl"
#include "/util/Coords.glsl"
#include "LightSamplingVisibility.glsl"

const uint LS_LOCAL_NODE_CAPACITY = 262144u;
const uint LS_OUTER_NODE_COUNT = uint(VOXEL_GRID_BRICKS) * 2u;
const uint LS_LEAF_BIT = 0x80000000u;
struct LightSamplingNode {
    vec4 boundsMinMass;
    vec4 boundsMaxLink;
};
layout(std430, binding = 12) buffer LightSamplingData {
    uint ls_allocatedNodes;
    uint ls_emitterCount;
    uint ls_overflowCount;
    uint ls_exhaustedCount;
    uint ls_traceCount;
    uint ls_visibleCount;
    uint ls_receiverCount;
    uint ls_guidedSelections;
    LightSamplingNode ls_nodes[];
};

bool ls_isEmitter(uint materialID) {
    return materialID > 1u && hardcodedpbr_decode(materialID).emissive > 0.0;
}

LightSamplingNode ls_emptyNode() {
    return LightSamplingNode(vec4(0.0), vec4(0.0));
}

LightSamplingNode ls_merge(LightSamplingNode a, LightSamplingNode b, uint child) {
    if (a.boundsMinMass.w == 0.0) a.boundsMinMass.xyz = b.boundsMinMass.xyz;
    if (b.boundsMinMass.w == 0.0) b.boundsMinMass.xyz = a.boundsMinMass.xyz;
    if (a.boundsMinMass.w == 0.0) a.boundsMaxLink.xyz = b.boundsMaxLink.xyz;
    if (b.boundsMinMass.w == 0.0) b.boundsMaxLink.xyz = a.boundsMaxLink.xyz;
    return LightSamplingNode(
        vec4(min(a.boundsMinMass.xyz, b.boundsMinMass.xyz), a.boundsMinMass.w + b.boundsMinMass.w),
        vec4(max(a.boundsMaxLink.xyz, b.boundsMaxLink.xyz), uintBitsToFloat(child))
    );
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

float ls_geometricWeight(LightSamplingNode node, vec3 origin, vec3 normal) {
    if (node.boundsMinMass.w == 0.0) return 0.0;
    vec3 halfSize = (node.boundsMaxLink.xyz - node.boundsMinMass.xyz) * 0.5;
    vec3 delta = node.boundsMinMass.xyz + halfSize - origin;
    float radius = length(halfSize);
    float distanceSquared = dot(delta, delta);
    float cosineBound = clamp((dot(normal, delta) + radius) * inversesqrt(max(distanceSquared, 0.25)), 0.001, 1.0);
    return node.boundsMinMass.w * cosineBound / max(distanceSquared, max(radius * radius, 0.25));
}

vec2 ls_nodeWeights(LightSamplingNode node, vec3 origin, vec3 normal) {
    float geometry = ls_geometricWeight(node, origin, normal);
    float visibility = geometry > 0.0 ? lsv_nodeFactor(node.boundsMinMass.xyz, node.boundsMaxLink.xyz, origin) : 1.0;
    return geometry * vec2(1.0, visibility);
}

vec2 ls_oneLevelWeights(LightSamplingNode node, vec3 origin, vec3 normal) {
    uint link = floatBitsToUint(node.boundsMaxLink.w);
    if (node.boundsMinMass.w == 0.0 || (link & LS_LEAF_BIT) != 0u) return ls_nodeWeights(node, origin, normal);
    return ls_nodeWeights(ls_nodes[link], origin, normal) + ls_nodeWeights(ls_nodes[link + 1u], origin, normal);
}

vec2 ls_weights(LightSamplingNode node, vec3 origin, vec3 normal) {
    #ifdef SETTING_DEBUG_LIGHT_LOOKAHEAD
    uint link = floatBitsToUint(node.boundsMaxLink.w);
    if (node.boundsMinMass.w == 0.0 || (link & LS_LEAF_BIT) != 0u) return ls_nodeWeights(node, origin, normal);
    return ls_oneLevelWeights(ls_nodes[link], origin, normal) + ls_oneLevelWeights(ls_nodes[link + 1u], origin, normal);
    #else
    return ls_nodeWeights(node, origin, normal);
    #endif
}

LightSamplingNode ls_select(vec3 origin, vec3 normal, inout uint rng, uint auditSlot, out float pdf) {
    LightSamplingNode node = ls_nodes[1];
    vec2 probabilities = vec2(1.0);
    bool guided = false;
    #if SETTING_DEBUG_LIGHT_VISIBILITY > 0
    bool chooseBase = ls_random(rng) < 0.1;
    #else
    bool chooseBase = true;
    #endif
    while ((floatBitsToUint(node.boundsMaxLink.w) & LS_LEAF_BIT) == 0u && node.boundsMinMass.w > 0.0) {
        uint child = floatBitsToUint(node.boundsMaxLink.w);
        LightSamplingNode left = ls_nodes[child];
        LightSamplingNode right = ls_nodes[child + 1u];
        vec2 leftWeight = ls_weights(left, origin, normal);
        vec2 rightWeight = ls_weights(right, origin, normal);
        vec2 probability = leftWeight / (leftWeight + rightWeight);
        if (leftWeight.y + rightWeight.y == 0.0) probability.y = probability.x;
        guided = guided || abs(probability.x - probability.y) > 1e-6;
        bool chooseLeft = ls_random(rng) < (chooseBase ? probability.x : probability.y);
        probabilities *= chooseLeft ? probability : vec2(1.0) - probability;
        node = chooseLeft ? left : right;
    }
    #if SETTING_DEBUG_LIGHT_VISIBILITY > 0
    if (guided) atomicAdd(ls_guidedSelections, 1u);
    pdf = 0.1 * probabilities.x + 0.9 * probabilities.y;
    #else
    pdf = probabilities.x;
    #endif
    if (auditSlot != LSV_INVALID) {
        uint offset = auditSlot * 4u;
        lsv_audit[offset] = vec4(origin, node.boundsMaxLink.w);
        lsv_audit[offset + 1u] = vec4(normal, pdf);
        lsv_audit[offset + 2u] = vec4(probabilities, float(chooseBase), 1.0);
        lsv_audit[offset + 3u] = vec4(node.boundsMinMass.xyz, uintBitsToFloat(lsv_brick(origin)));
    }
    return node;
}
#endif
