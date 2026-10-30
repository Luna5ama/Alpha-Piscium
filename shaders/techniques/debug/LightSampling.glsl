#ifndef INCLUDE_debug_LightSampling_glsl
#define INCLUDE_debug_LightSampling_glsl a
#include "/Base.glsl"
#include "/techniques/voxel/Voxelization.glsl"
#include "/util/HardcodedPBR.glsl"
#include "/util/Hash.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Colors.glsl"
#include "/util/Coords.glsl"
// Experiment snapshots select the Morton prefix radix without adding a user option.
#define LS_TREE_BITS 3
#define LS_TREE_ARITY (1 << LS_TREE_BITS)

const uint LS_INVALID = 0xffffffffu;
layout(std430, binding = 14) buffer LightSamplingAudit {
    vec4 ls_audit[];
};

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
    #ifdef SETTING_DEBUG_LIGHT_WEIGHT
    // Uniform projected extent approximates the hemisphere-clipped cosine.
    // This is a contribution estimate, not a conservative bound.
    float extent = dot(abs(normal), halfSize);
    float height = dot(normal, delta);
    float clippedHeight = height >= extent ? height : (height <= -extent ? 0.0 : (height + extent) * (height + extent) / (4.0 * extent));
    float spreadSquared = dot(halfSize, halfSize) / 3.0;
    float scaleSquared = distanceSquared + spreadSquared;
    float cosine = clamp(clippedHeight * inversesqrt(scaleSquared), 0.0, 1.0);
    // A small proposal floor preserves support despite coarse energy estimates.
    return node.boundsMinMass.w * max(cosine, 0.001) / scaleSquared;
    #else
    float cosineBound = clamp((dot(normal, delta) + radius) * inversesqrt(max(distanceSquared, 0.25)), 0.001, 1.0);
    return node.boundsMinMass.w * cosineBound / max(distanceSquared, max(radius * radius, 0.25));
    #endif
}

LightSamplingNode ls_select(vec3 origin, vec3 normal, inout uint rng, uint auditSlot, out float pdf) {
    LightSamplingNode node = ls_nodes[1];
    pdf = 1.0;
    while ((floatBitsToUint(node.boundsMaxLink.w) & LS_LEAF_BIT) == 0u && node.boundsMinMass.w > 0.0) {
        uint link = floatBitsToUint(node.boundsMaxLink.w);
        uint child = ls_childBase(link);
        uint count = ls_childCount(link);
        vec4 weightsLow = vec4(0.0);
        #if LS_TREE_ARITY == 8
        vec4 weightsHigh = vec4(0.0);
        #endif
        float total = 0.0;
        for (uint i = 0u; i < count; ++i) {
            float weight = ls_geometricWeight(ls_nodes[child + i], origin, normal);
            #if LS_TREE_ARITY == 8
            if (i >= 4u) weightsHigh[i - 4u] = weight;
            else
            #endif
            weightsLow[i] = weight;
            total += weight;
        }
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
