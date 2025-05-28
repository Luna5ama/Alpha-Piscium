/*
    Contains code adopted from:
        https://iolite-engine.com/blog_posts/minimal_agx_implementation
        MIT License
        Copyright (c) 2024 Missing Deadlines (Benjamin Wrensch)

        You can find full license texts in /licenses
*/

// All values used to derive this implementation are sourced from Troy’s initial AgX implementation/OCIO config file available here:
//   https://github.com/sobotka/AgX

#include "/util/Rand.glsl"

shared uint shared_lumHistogram[256];
shared float shared_topBinSum[16];
shared float shared_weightSum[16];

// Mean error^2: 3.6705141e-06
vec3 agxDefaultContrastApprox(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;

    return + 15.5     * x4 * x2
    - 40.14    * x4 * x
    + 31.96    * x4
    - 6.868    * x2 * x
    + 0.4298   * x2
    + 0.1191   * x
    - 0.00232;
}

vec3 agx(vec3 val) {
    const mat3 agx_mat = mat3(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772, 0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104
    );

    // Input transform
    val = agx_mat * val;

    // Log2 space encoding
    const float EV_RANGE = SETTING_TONE_MAPPING_DYNAMIC_RANGE;
    const float EV_RANGE_HALF = EV_RANGE * 0.5;
    const float MID_GREY = 0.18;
    val = clamp(log2(val / MID_GREY), -EV_RANGE_HALF, EV_RANGE_HALF);
    val = (val + EV_RANGE_HALF) / EV_RANGE;

    // Apply sigmoid function approximation
    val = agxDefaultContrastApprox(val);

    return val;
}

vec3 agxEotf(vec3 val) {
    const mat3 agx_mat_inv = mat3(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116
    );

    // Undo input transform
    val = agx_mat_inv * val;

    // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
    val = pow(val, vec3(2.2));

    return val;
}

vec3 agxLook(vec3 val) {
    const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
    float luma = dot(val, lw);

    // Default
    vec3 offset = vec3(0.0);
    vec3 slope = vec3(1.0);
    vec3 power = vec3(1.0);
    float sat = 1.0;

    #if SETTING_TONE_MAPPING_LOOK == 1
    // Golden
    slope = vec3(1.0, 0.9, 0.5);
    power = vec3(0.8);
    sat = 0.8;
    #elif SETTING_TONE_MAPPING_LOOK == 2
    // Punchy
    slope = vec3(1.0);
    power = vec3(1.35, 1.35, 1.35);
    sat = 1.4;
    #elif SETTING_TONE_MAPPING_LOOK == 3
    // Custom
    offset = vec3(SETTING_TONE_MAPPING_OFFSET_R, SETTING_TONE_MAPPING_OFFSET_G, SETTING_TONE_MAPPING_OFFSET_B);
    slope = vec3(SETTING_TONE_MAPPING_SLOPE_R, SETTING_TONE_MAPPING_SLOPE_G, SETTING_TONE_MAPPING_SLOPE_B);
    power = vec3(SETTING_TONE_MAPPING_POWER_R, SETTING_TONE_MAPPING_POWER_G, SETTING_TONE_MAPPING_POWER_B);
    sat = SETTING_TONE_MAPPING_SATURATION;
    #endif

    // ASC CDL
    val = pow(val * slope + offset, power);
    return luma + sat * (val - luma);
}

uint histoIndex(float x, float range, uint outputRangeExclusive) {
    const float EPSILON = 0.001;
    uint binIndex;
    if (x < EPSILON) {
        binIndex = 0;
    } else {
        float lumMapped = saturate(log2(x + 1.0) / range);
        binIndex = uint(lumMapped * float(outputRangeExclusive - 2) + 1.0);
    }
    return binIndex;
}

void toneMapping_init() {
    shared_lumHistogram[gl_LocalInvocationIndex] = 0u;
    if (gl_LocalInvocationIndex < 16) {
        shared_topBinSum[gl_LocalInvocationIndex] = 0.0;
        shared_weightSum[gl_LocalInvocationIndex] = 0.0;
    }
    barrier();
}

vec3 applyAgx(vec3 color) {
    vec3 result = color;
    color = agx(color);
    color = agxLook(color);
    color = agxEotf(color);
    color = saturate(color);
    return color;
}

void toneMapping_apply(inout vec4 outputColor) {
    {
        float weightSum = subgroupAdd(outputColor.a);
        if (subgroupElect()) {
            shared_weightSum[gl_SubgroupID] = weightSum;
        }
    }

    {
        float lumimance = colors_srgbLuma(applyAgx(outputColor.rgb * exp2(global_exposure.x)));
        uint binIndexAvg = histoIndex(lumimance, 1.0, 256);
        float noise = rand_stbnVec1(ivec2(gl_GlobalInvocationID.xy), frameCounter);
        atomicAdd(shared_lumHistogram[binIndexAvg], uint(outputColor.a + noise));
    }

    {
        float lumimance = colors_srgbLuma(applyAgx(outputColor.rgb * exp2(global_exposure.y)));
        float topBinV = float(lumimance > SETTING_EXPOSURE_TOP_BIN_LUM) * outputColor.a;
        float topBinSum = subgroupAdd(topBinV);
        if (subgroupElect()) {
            shared_topBinSum[gl_SubgroupID] = topBinSum;
        }
    }

    outputColor.rgb = applyAgx(outputColor.rgb * exp2(global_exposure.w));
    outputColor.rgb = pow(outputColor.rgb, vec3(1.0 / SETTING_TONE_MAPPING_OUTPUT_GAMMA));

    barrier();

    if (gl_SubgroupID == 0 && gl_SubgroupInvocationID < gl_NumSubgroups) {
        float topBinV = shared_topBinSum[gl_SubgroupInvocationID];
        float topBinSum = subgroupAdd(topBinV);
        float weightV = shared_weightSum[gl_SubgroupInvocationID];
        float weightSum = subgroupAdd(weightV);
        if (subgroupElect()) {
            float noise = rand_stbnVec1(ivec2(gl_GlobalInvocationID.xy), frameCounter);
            atomicAdd(global_lumHistogramTopBinSum, uint(topBinSum + noise));
            atomicAdd(global_lumHistogramWeightSum, uint(weightSum + noise));
        }
    }

    atomicAdd(global_lumHistogram[gl_LocalInvocationIndex], shared_lumHistogram[gl_LocalInvocationIndex]);
}