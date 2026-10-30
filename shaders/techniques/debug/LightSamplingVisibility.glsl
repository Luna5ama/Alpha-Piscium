#ifndef INCLUDE_debug_LightSamplingVisibility_glsl
#define INCLUDE_debug_LightSamplingVisibility_glsl a
#include "/Base.glsl"
#include "/techniques/voxel/Voxelization.glsl"

const uint LSV_BRICKS = 4096u;
const uint LSV_DIRECTIONS = 512u;
const uint LSV_ORIGINS = 16u;
const uint LSV_INVALID = 0xffffffffu;
const uint LSV_COUNTS_OFFSET = 65536u;
const uint LSV_SH_OFFSET = 69632u;
const uint LSV_MOMENTS_OFFSET = 143360u;
const uint LSV_DISTANCES_OFFSET = 4337664u;
layout(std430, binding = 13) buffer LightSamplingVisibilityData {
    uint lsv_data[];
};

vec2 lsv_loadPair(uint index) {
    return uintBitsToFloat(uvec2(lsv_data[index], lsv_data[index + 1u]));
}

void lsv_storePair(uint index, vec2 value) {
    uvec2 words = floatBitsToUint(value);
    lsv_data[index] = words.x;
    lsv_data[index + 1u] = words.y;
}
layout(std430, binding = 14) buffer LightSamplingVisibilityCounters {
    uint lsv_frame;
    uint lsv_ready;
    uint lsv_probeCount;
    uint lsv_exhausted;
    uint lsv_activeBricks;
    uint lsv_originCount;
    uint lsv_hashXor;
    uint lsv_hashSum;
    uint lsv_originHashXor;
    uint lsv_originHashSum;
    uint lsv_trainingRuns;
    uint lsv_cameraBrickX;
    uint lsv_cameraBrickY;
    uint lsv_cameraBrickZ;
    uint lsv_reserved3;
    uint lsv_reserved4;
    vec4 lsv_audit[];
};

uint lsv_brick(vec3 origin) {
    ivec3 relative = (ivec3(floor(origin)) >> 4) - (cameraPositionInt >> 4) + 8;
    if (any(lessThan(relative, ivec3(0))) || any(greaterThanEqual(relative, ivec3(16)))) return LSV_INVALID;
    return voxel_brickMorton(relative);
}

vec3 lsv_direction(uint bin) {
    float y = -1.0 + (float(bin >> 5u) + 0.5) * (2.0 / 16.0);
    float phi = (float(bin & 31u) + 0.5) * (6.28318530718 / 32.0);
    float radius = sqrt(1.0 - y * y);
    return vec3(radius * cos(phi), y, radius * sin(phi));
}

float lsv_basis(vec3 d, uint index) {
    if (index == 0u) return 0.28209479177387814;
    if (index == 1u) return 0.4886025119029199 * d.x;
    if (index == 2u) return 0.4886025119029199 * d.y;
    if (index == 3u) return 0.4886025119029199 * d.z;
    if (index == 4u) return 1.0925484305920792 * d.x * d.y;
    if (index == 5u) return 1.0925484305920792 * d.y * d.z;
    if (index == 6u) return 1.0925484305920792 * d.x * d.z;
    if (index == 7u) return 0.31539156525252005 * (3.0 * d.y * d.y - 1.0);
    return 0.5462742152960396 * (d.x * d.x - d.z * d.z);
}

// Exact least-squares projection for the fixed 16-by-32 direction grid.
float lsv_projectionWeight(vec3 direction, uint index) {
    float basis = lsv_basis(direction, index);
    if (index == 0u) return 0.02454417003560938 * basis + 0.00010931862326547718 * lsv_basis(direction, 7u);
    if (index == 1u || index == 3u) return 0.024495849150797604 * basis;
    if (index == 2u) return 0.024639942381096434 * basis;
    if (index == 4u || index == 5u) return 0.024307611751678924 * basis;
    if (index == 7u) return 0.025031052577621754 * basis + 0.00010931862326547718 * lsv_basis(direction, 0u);
    return 0.024543364916895613 * basis;
}

float lsv_survival(uint brick, vec3 direction, float distance) {
    float threshold = distance / (distance + 16.0);
    #if SETTING_DEBUG_LIGHT_VISIBILITY == 3
    vec2 moments = vec2(0.0);
    for (uint i = 0u; i < 9u; ++i) moments += lsv_loadPair(LSV_SH_OFFSET + (brick * 9u + i) * 2u) * lsv_basis(direction, i);
    #else
    uint row = uint(clamp((direction.y + 1.0) * 8.0, 0.0, 15.0));
    float phi = mod(atan(direction.z, direction.x), 6.28318530718);
    uint column = min(uint(phi * (32.0 / 6.28318530718)), 31u);
    uint bin = row * 32u + column;
    #if SETTING_DEBUG_LIGHT_VISIBILITY == 1
    uint offset = (brick * LSV_DIRECTIONS + bin) * LSV_ORIGINS;
    uint survived = 0u;
    for (uint i = 0u; i < LSV_ORIGINS; ++i) survived += uint(uintBitsToFloat(lsv_data[LSV_DISTANCES_OFFSET + offset + i]) >= threshold);
    return float(survived) / float(lsv_data[LSV_COUNTS_OFFSET + brick]);
    #else
    vec2 moments = lsv_loadPair(LSV_MOMENTS_OFFSET + (brick * LSV_DIRECTIONS + bin) * 2u);
    #endif
    #endif
    #if SETTING_DEBUG_LIGHT_VISIBILITY != 1
    float mean = clamp(moments.x, 0.0, 1.0);
    float second = clamp(moments.y, mean * mean, mean);
    float variance = max(second - mean * mean, 0.0);
    float delta = threshold - mean;
    return delta <= 0.0 ? 1.0 : variance / (variance + delta * delta);
    #endif
}

float lsv_nodeFactor(vec3 boundsMin, vec3 boundsMax, vec3 origin) {
    #if SETTING_DEBUG_LIGHT_VISIBILITY == 0
    return 1.0;
    #else
    if (lsv_ready == 0u) return 1.0;
    uint brick = lsv_brick(origin);
    if (brick == LSV_INVALID || lsv_data[LSV_COUNTS_OFFSET + brick] == 0u) return 1.0;
    vec3 halfSize = (boundsMax - boundsMin) * 0.5;
    vec3 delta = boundsMin + halfSize - origin;
    float distance = length(delta);
    float radius = length(halfSize);
    float ratio = radius / distance;
    if (ratio > 0.5) return 1.0;
    vec3 direction = delta / distance;
    vec3 tangent = normalize(cross(direction, abs(direction.y) > 0.95 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
    vec3 bitangent = cross(direction, tangent);
    float cosine = sqrt(1.0 - ratio * ratio);
    float nearDistance = distance - radius;
    float estimate = lsv_survival(brick, direction, nearDistance);
    estimate += lsv_survival(brick, direction * cosine + tangent * ratio, nearDistance);
    estimate += lsv_survival(brick, direction * cosine - tangent * ratio, nearDistance);
    estimate += lsv_survival(brick, direction * cosine + bitangent * ratio, nearDistance);
    estimate += lsv_survival(brick, direction * cosine - bitangent * ratio, nearDistance);
    float strength = (1.0 - ratio) * (1.0 - ratio);
    return 1.0 - strength + strength * (estimate * 0.2);
    #endif
}
#endif
