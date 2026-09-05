#ifndef INCLUDE_clouds_Cumulus_glsl
#define INCLUDE_clouds_Cumulus_glsl a

#include "Common.glsl"
#include "/util/Hash.glsl"

const float CU_CELL_FREQUENCY = 0.24 * exp2(SETTING_CLOUDS_LOW_BASE_FREQ);
const float CU_BILLOW_FREQUENCY = 0.65 * exp2(SETTING_CLOUDS_LOW_BILLOWY_FREQ);
const float CU_DETAIL_FREQUENCY = 2.1 * exp2(SETTING_CLOUDS_HIGH_BILLOWY_FREQ);
const float CU_DENSITY_SCALE = 240.0 * SETTING_CLOUDS_CU_DENSITY;

float clouds_cu_lobe(vec2 offset, float height, float centerHeight, float radius, float verticalRadius) {
    return (1.0 - length(vec3(offset / radius, (height - centerHeight) / verticalRadius))) * radius;
}

float clouds_cu_union(float a, float b, float width) {
    float overlap = max(width - abs(a - b), 0.0);
    return max(a, b) + overlap * overlap / (4.0 * width);
}

float clouds_cu_thermal(vec2 offset, float height, float top, float radius) {
    float stem = clouds_cu_lobe(offset, height, top * 0.36, radius * 0.68, top * 0.60);
    float head = clouds_cu_lobe(offset, height, top * 0.70, radius * 0.70, top * 0.30);
    return clouds_cu_union(stem, head, radius * 0.12);
}

float clouds_cu_density(vec3 rayPos, float heightFraction) {
    if (heightFraction <= 0.0 || heightFraction >= 1.0) return 0.0;

    vec3 pos = vec3(rayPos.x, heightFraction * SETTING_CLOUDS_CU_THICKNESS, rayPos.z);
    #if defined(SETTING_CLOUDS_CU_WIND) && !defined(SETTING_SCREENSHOT_MODE)
    pos += uval_cuDetailWind;
    #endif

    float moisture = texture(usam_cumulusDetail1, vec3(pos.x * 0.045, 0.37, pos.z * 0.045)).r;
    float coverage = saturate(SETTING_CLOUDS_CU_COVERAGE * 2.0 + moisture * 2.5 - 1.1);
    if (coverage <= 0.0) return 0.0;
    float development = smoothstep(0.05, 0.6, coverage);
    vec2 cellPos = pos.xz * CU_CELL_FREQUENCY;
    ivec2 cell = ivec2(floor(cellPos));
    vec2 localPos = fract(cellPos);
    float maxReach = 1.7 * 0.86 * 1.12 * pow(0.94, 0.7) * coverage + 0.05;
    ivec2 firstCell = ivec2(ceil(localPos - maxReach - 0.82));
    ivec2 lastCell = ivec2(floor(localPos + maxReach - 0.18));
    float shape = -1.0;

    // Overlapping thermals share a condensation base but rise to different heights.
    for (int z = firstCell.y; z <= lastCell.y; ++z) {
        for (int x = firstCell.x; x <= lastCell.x; ++x) {
            ivec2 offset = ivec2(x, z);
            vec3 seed = hash_uintToFloat(hash_33_q3(uvec3(uvec2(cell + offset), 0x1919810u)));
            vec2 center = vec2(offset) + mix(vec2(0.18), vec2(0.82), seed.xy);
            vec2 delta = localPos - center;
            float top = mix(0.5, 0.94, smoothstep(0.1, 0.92, seed.z));
            top *= mix(0.25, 1.0, development);
            float radius = 0.86 * coverage * mix(0.8, 1.12, seed.x) * pow(top, 0.7);
            if (dot(delta, delta) > pow2(radius * 1.7 + 0.05) || heightFraction > top + 0.12) continue;

            float angle = seed.x * (2.0 * PI);
            vec2 axis = vec2(cos(angle), sin(angle));
            vec2 lean = (seed.xy - 0.5) * radius * heightFraction / top;
            vec2 thermalPos = delta - lean;
            float mainThermal = clouds_cu_thermal(thermalPos, heightFraction, top, radius);
            float sideHeight = top * mix(0.58, 0.86, seed.y);
            float sideThermal = clouds_cu_thermal(thermalPos - axis * radius * 0.43, heightFraction, sideHeight, radius * 0.82);
            float youngHeight = top * mix(0.36, 0.65, seed.x);
            float youngThermal = clouds_cu_thermal(thermalPos + axis * radius * 0.48, heightFraction, youngHeight, radius * 0.86);
            float thermals = clouds_cu_union(mainThermal, clouds_cu_union(sideThermal, youngThermal, radius * 0.12), radius * 0.12);
            shape = clouds_cu_union(shape, thermals / radius, 0.075);
        }
    }
    if (shape <= 0.0) return 0.0;

    vec3 curl = texture(usam_cumulusCurl, pos * 0.35).xyz;
    vec3 billowPos = pos * CU_BILLOW_FREQUENCY + curl * 0.09;
    billowPos.y *= 0.75;
    float billow = texture(usam_cumulusDetail1, billowPos).r;
    float entrainment = mix(0.7, 1.0, smoothstep(0.02, 0.15, heightFraction));
    shape -= (1.0 - sqrt(billow)) * 0.38 * entrainment;
    if (shape <= 0.0) return 0.0;

    // Erode the interface; the interior stays coherent instead of becoming wispy.
    vec3 detailPos = pos * CU_DETAIL_FREQUENCY + curl * 0.13;
    float erosion = (1.0 - texture(usam_cumulusDetail2, detailPos).r) * 0.07;
    erosion += (1.0 - texture(usam_cumulusDetail2, detailPos * 3.7).r) * 0.02;
    shape -= erosion * entrainment;
    float surface = smoothstep(0.0, 0.015, shape);
    float condensationHeight = (1.0 - sqrt(billow)) * 0.075 + erosion * 0.05;
    float condensation = smoothstep(condensationHeight, condensationHeight + 0.006, heightFraction);
    float liquidWater = mix(0.55, 1.35, smoothstep(0.0, 0.32, heightFraction));
    return surface * condensation * liquidWater * CU_DENSITY_SCALE;
}

float clouds_cu_column(vec3 origin, vec3 direction, vec2 layerRange, int steps, float spread, float jitter) {
    vec2 interval;
    if (!clouds_layerInterval(origin, direction, layerRange, interval)) return 0.0;
    float rayLength = min(interval.y - interval.x, 12.0);
    float column = 0.0;
    for (int i = 0; i < steps; ++i) {
        float start = (exp2(float(i) / float(steps) * spread) - 1.0) / (exp2(spread) - 1.0) * rayLength;
        float end = (exp2(float(i + 1) / float(steps) * spread) - 1.0) / (exp2(spread) - 1.0) * rayLength;
        vec3 pos = origin + direction * (interval.x + mix(start, end, jitter));
        float height = (length(pos) - layerRange.x) / (layerRange.y - layerRange.x);
        column += clouds_cu_density(pos, height) * (end - start);
    }
    return column;
}

#endif
