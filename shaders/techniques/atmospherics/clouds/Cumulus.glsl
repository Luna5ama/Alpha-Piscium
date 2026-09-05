#ifndef INCLUDE_clouds_Cumulus_glsl
#define INCLUDE_clouds_Cumulus_glsl a

#include "Common.glsl"
#include "/util/Hash.glsl"

const float CU_CELL_FREQUENCY = 0.24 * exp2(SETTING_CLOUDS_LOW_BASE_FREQ);
const float CU_BILLOW_FREQUENCY = 0.65 * exp2(SETTING_CLOUDS_LOW_BILLOWY_FREQ);
const float CU_DETAIL_FREQUENCY = 1.2 * exp2(SETTING_CLOUDS_HIGH_BILLOWY_FREQ);
const float CU_DENSITY_SCALE = 96.0 * SETTING_CLOUDS_CU_DENSITY;

float clouds_cu_lobe(vec2 offset, float height, float centerHeight, float radius, float verticalRadius) {
    return (1.0 - length(vec3(offset / radius, (height - centerHeight) / verticalRadius))) * radius;
}

float clouds_cu_union(float a, float b) {
    float overlap = max(0.075 - abs(a - b), 0.0);
    return max(a, b) + overlap * overlap / 0.3;
}

float clouds_cu_density(vec3 rayPos, float heightFraction) {
    if (heightFraction <= 0.0 || heightFraction >= 1.0) return 0.0;

    vec3 pos = vec3(rayPos.x, heightFraction * SETTING_CLOUDS_CU_THICKNESS, rayPos.z);
    #if defined(SETTING_CLOUDS_CU_WIND) && !defined(SETTING_SCREENSHOT_MODE)
    pos += uval_cuDetailWind;
    #endif

    vec2 cellPos = pos.xz * CU_CELL_FREQUENCY;
    ivec2 cell = ivec2(floor(cellPos));
    vec2 localPos = fract(cellPos);
    float shape = -1.0;

    // Overlapping thermals share a condensation base but rise to different heights.
    for (int z = -1; z <= 1; ++z) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 offset = ivec2(x, z);
            vec3 seed = hash_uintToFloat(hash_33_q3(uvec3(uvec2(cell + offset), 0x1919810u)));
            vec2 center = vec2(offset) + mix(vec2(0.18), vec2(0.82), seed.xy);
            vec2 delta = localPos - center;
            float radius = mix(0.12, 0.86, SETTING_CLOUDS_CU_COVERAGE) * mix(0.8, 1.12, seed.x);
            float top = mix(0.28, 0.94, smoothstep(0.1, 0.92, seed.z));
            if (dot(delta, delta) > pow2(radius + 0.22) || heightFraction > top + 0.12) continue;

            vec2 lean = (seed.xy - 0.5) * radius * 0.65;
            float base = clouds_cu_lobe(delta, heightFraction, top * 0.12, radius, top * 0.28);
            float body = clouds_cu_lobe(delta - lean, heightFraction, top * 0.40, radius * 0.77, top * 0.36);
            float crown = clouds_cu_lobe(delta - lean * 1.6, heightFraction, top * 0.71, radius * 0.57, top * 0.29);
            shape = clouds_cu_union(shape, clouds_cu_union(base, clouds_cu_union(body, crown)));
        }
    }
    if (shape <= 0.0) return 0.0;

    vec3 curl = texture(usam_cumulusCurl, pos * 0.35).xyz;
    vec3 billowPos = pos * CU_BILLOW_FREQUENCY + curl * 0.09;
    float billow = texture(usam_cumulusDetail1, billowPos).r;
    float entrainment = mix(0.35, 1.0, smoothstep(0.04, 0.2, heightFraction));
    shape -= pow3(1.0 - billow) * 0.32 * entrainment;
    if (shape <= 0.0) return 0.0;

    // Erode the interface; the interior stays coherent instead of becoming wispy.
    vec3 detailPos = pos * CU_DETAIL_FREQUENCY + curl * 0.13;
    float erosion = (1.0 - texture(usam_cumulusDetail2, detailPos).r) * 0.045;
    erosion += (1.0 - texture(usam_cumulusDetail2, detailPos * 3.7).r) * 0.012;
    shape -= erosion * entrainment;
    float surface = smoothstep(0.0, 0.04, shape);
    float condensation = smoothstep(0.0, 0.025, heightFraction);
    float liquidWater = mix(0.55, 1.35, smoothstep(0.0, 0.32, heightFraction));
    return surface * condensation * liquidWater * CU_DENSITY_SCALE;
}

float clouds_cu_column(vec3 origin, vec3 direction, vec2 layerRange, int steps, float jitter) {
    vec2 interval;
    if (!clouds_layerInterval(origin, direction, layerRange, interval)) return 0.0;
    float rayLength = min(interval.y, 12.0);
    float column = 0.0;
    for (int i = 0; i < steps; ++i) {
        float start = pow2(float(i) / float(steps)) * rayLength;
        float end = pow2(float(i + 1) / float(steps)) * rayLength;
        vec3 pos = origin + direction * mix(start, end, jitter);
        float height = (length(pos) - layerRange.x) / (layerRange.y - layerRange.x);
        column += clouds_cu_density(pos, height) * (end - start);
    }
    return column;
}

#endif
