#ifndef INCLUDE_clouds_Cumulus_glsl
#define INCLUDE_clouds_Cumulus_glsl a

#include "Common.glsl"
#include "/util/Hash.glsl"

const float CU_CELL_FREQUENCY = 0.24 * exp2(SETTING_CLOUDS_LOW_BASE_FREQ);
const float CU_BILLOW_FREQUENCY = 0.65 * exp2(SETTING_CLOUDS_LOW_BILLOWY_FREQ);
const float CU_DETAIL_FREQUENCY = 2.1 * exp2(SETTING_CLOUDS_HIGH_BILLOWY_FREQ);
const float CU_DENSITY_SCALE = 240.0 * SETTING_CLOUDS_CU_DENSITY;
const float CU_DENSITY_MAX = 1.35 * CU_DENSITY_SCALE;

vec3 clouds_cu_wind(float time) {
    #if defined(SETTING_CLOUDS_CU_WIND) && !defined(SETTING_SCREENSHOT_MODE)
    float speed = exp2(SETTING_CLOUDS_CU_WIND_SPEED);
    return (vec3(1.0, 0.1, 0.7) * (time * 0.008)
        + vec3(-0.15, 0.05, 0.09) * (sin(time * 0.6 + 0.5) * 0.006)
        + vec3(0.23, -0.06, 0.12) * (sin(time * 0.4 + 0.2) * 0.009)) * speed;
    #else
    return vec3(0.0);
    #endif
}

float clouds_cu_lobe(vec2 offset, float height, float centerHeight, float radius, float verticalRadius) {
    return (1.0 - length(vec3(offset / radius, (height - centerHeight) / verticalRadius))) * radius;
}

float clouds_cu_union(float a, float b) {
    float overlap = max(0.075 - abs(a - b), 0.0);
    return max(a, b) + overlap * overlap / 0.3;
}

float clouds_cu_thermal(vec2 offset, float height, float top, float radius) {
    float stem = clouds_cu_lobe(offset, height, top * 0.36, radius * 0.68, top * 0.60);
    float head = clouds_cu_lobe(offset, height, top * 0.70, radius * 0.70, top * 0.30);
    return clouds_cu_union(stem, head);
}

float clouds_cu_density(vec3 rayPos, float heightFraction) {
    if (heightFraction <= 0.0 || heightFraction >= 1.0 || SETTING_CLOUDS_CU_COVERAGE <= 0.0) return 0.0;

    vec3 pos = vec3(rayPos.x, heightFraction * SETTING_CLOUDS_CU_THICKNESS, rayPos.z);
    pos += clouds_cu_wind(frameTimeCounter);

    vec2 cellPos = pos.xz * CU_CELL_FREQUENCY;
    ivec2 cell = ivec2(floor(cellPos));
    vec2 localPos = fract(cellPos);
    float coverageRadius = mix(0.12, 0.86, SETTING_CLOUDS_CU_COVERAGE);
    float maxReach = coverageRadius * 1.12 * 1.7 + 0.075;
    ivec2 first = ivec2(ceil(localPos - maxReach - 0.82));
    ivec2 last = ivec2(floor(localPos + maxReach - 0.18));
    float shape = -1.0;

    // Overlapping thermals share a condensation base but rise to different heights.
    for (int z = first.y; z <= last.y; ++z) {
        for (int x = first.x; x <= last.x; ++x) {
            ivec2 offset = ivec2(x, z);
            vec3 seed = hash_uintToFloat(hash_33_q3(uvec3(uvec2(cell + offset), 0x1919810u)));
            vec2 center = vec2(offset) + mix(vec2(0.18), vec2(0.82), seed.xy);
            vec2 delta = localPos - center;
            float radius = coverageRadius * mix(0.8, 1.12, seed.x);
            float top = mix(0.28, 0.94, smoothstep(0.1, 0.92, seed.z));
            if (dot(delta, delta) > pow2(radius * 1.7 + 0.075) || heightFraction > top + 0.12) continue;

            float angle = seed.x * (2.0 * PI);
            vec2 axis = vec2(cos(angle), sin(angle));
            vec2 lean = (seed.xy - 0.5) * radius * heightFraction / top;
            vec2 thermalPos = delta - lean;
            float base = clouds_cu_lobe(delta, heightFraction, top * 0.04, radius * 0.88, top * 0.16);
            float mainThermal = clouds_cu_thermal(thermalPos, heightFraction, top, radius);
            float sideHeight = top * mix(0.58, 0.86, seed.y);
            float sideThermal = clouds_cu_thermal(thermalPos - axis * radius * 0.43, heightFraction, sideHeight, radius * 0.82);
            float youngHeight = top * mix(0.36, 0.65, seed.x);
            float youngThermal = clouds_cu_thermal(thermalPos + axis * radius * 0.48, heightFraction, youngHeight, radius * 0.86);
            float thermals = clouds_cu_union(mainThermal, clouds_cu_union(sideThermal, youngThermal));
            shape = clouds_cu_union(shape, clouds_cu_union(base, thermals) / radius);
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
    float condensationHeight = 0.008 + (1.0 - sqrt(billow)) * 0.025 + erosion * 0.05;
    float condensation = smoothstep(condensationHeight, condensationHeight + 0.012, heightFraction);
    float liquidWater = mix(0.55, 1.35, smoothstep(0.0, 0.32, heightFraction));
    return surface * condensation * liquidWater * CU_DENSITY_SCALE;
}

float clouds_cu_column(vec3 origin, vec3 direction, vec2 layerRange, int steps, float jitter) {
    vec2 interval;
    if (CU_DENSITY_SCALE <= 0.0 || !clouds_layerInterval(origin, direction, layerRange, interval)) return 0.0;
    float rayLength = min(interval.y - interval.x, 12.0);
    // Anchor near-field shadow resolution to extinction, independent of the distant layer exit.
    float lengthScale = 0.25 / (CU_DENSITY_MAX * mmax3(CLOUDS_CU_EXTINCTION));
    float spread = log2(1.0 + rayLength / lengthScale);
    float column = 0.0;
    for (int i = 0; i < steps; ++i) {
        float start = (exp2(float(i) / float(steps) * spread) - 1.0) * lengthScale;
        float end = (exp2(float(i + 1) / float(steps) * spread) - 1.0) * lengthScale;
        vec3 pos = origin + direction * (interval.x + mix(start, end, jitter));
        float height = (length(pos) - layerRange.x) / (layerRange.y - layerRange.x);
        column += clouds_cu_density(pos, height) * (end - start);
        if (column * mmin3(CLOUDS_CU_EXTINCTION) > 12.0) break;
    }
    return column;
}

vec3 clouds_cu_transmittance(vec3 origin, vec3 direction, vec2 layerRange, vec3 extinction, inout uint rng) {
    vec2 interval;
    if (CU_DENSITY_SCALE <= 0.0 || !clouds_layerInterval(origin, direction, layerRange, interval)) return vec3(1.0);
    float majorant = CU_DENSITY_MAX * mmax3(extinction);
    float nearEnd = min(interval.y, interval.x + 8.0 / majorant);
    float distance = interval.x;
    vec3 transmittance = vec3(1.0);
    // Ratio tracking resolves near-field occluders without exponentiating an undersampled column.
    while (true) {
        float xi = hash_uintToFloat(hash_pcg_11(rng++));
        distance -= log(1.0 - xi) / majorant;
        if (distance >= nearEnd) break;
        vec3 pos = origin + direction * distance;
        float density = clouds_cu_density(pos, (length(pos) - layerRange.x) / (layerRange.y - layerRange.x));
        transmittance *= 1.0 - extinction * density / majorant;
        if (mmax3(transmittance) == 0.0) return transmittance;
        // Unbiased early exit for opaque near-field shadow paths.
        float survival = mmax3(transmittance) * 1e6;
        if (survival < 1.0) {
            if (hash_uintToFloat(hash_pcg_11(rng++)) >= survival) return vec3(0.0);
            transmittance /= survival;
        }
    }
    if (nearEnd < interval.y) {
        float jitter = hash_uintToFloat(hash_pcg_11(rng++));
        float column = clouds_cu_column(origin + direction * nearEnd, direction, layerRange, 16, jitter);
        transmittance *= exp(-extinction * column);
    }
    return transmittance;
}

#endif
