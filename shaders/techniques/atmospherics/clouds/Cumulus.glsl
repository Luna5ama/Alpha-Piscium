#ifndef INCLUDE_clouds_Cumulus_glsl
#define INCLUDE_clouds_Cumulus_glsl a

#include "Common.glsl"
#include "/util/Hash.glsl"

const float CU_CELL_FREQUENCY = 0.24 * exp2(SETTING_CLOUDS_LOW_BASE_FREQ);
const float CU_BILLOW_FREQUENCY = 0.65 * exp2(SETTING_CLOUDS_LOW_BILLOWY_FREQ);
const float CU_DETAIL_FREQUENCY = 2.1 * exp2(SETTING_CLOUDS_HIGH_BILLOWY_FREQ);
const float CU_DENSITY_SCALE = 160.0 * SETTING_CLOUDS_CU_DENSITY;

float clouds_cu_plume(vec2 offset, float height, float top, float radius) {
    float h = height / top;
    float width = radius * mix(1.55, 1.0, smoothstep(0.02, 0.72, h));
    return 1.0 - length(vec3(offset / width, (h - 0.28) / 0.72));
}

float clouds_cu_union(float a, float b, float width) {
    float overlap = max(width - abs(a - b), 0.0);
    return max(a, b) + overlap * overlap / (4.0 * width);
}

float clouds_cu_density(vec3 rayPos, float heightFraction) {
    if (heightFraction <= 0.0 || heightFraction >= 1.0) return 0.0;

    vec3 pos = vec3(rayPos.x, heightFraction * SETTING_CLOUDS_CU_THICKNESS, rayPos.z);
    #if defined(SETTING_CLOUDS_CU_WIND) && !defined(SETTING_SCREENSHOT_MODE)
    pos += uval_cuDetailWind;
    #endif

    float moisture = texture(usam_cumulusDetail2, vec3(pos.x * 0.045, 0.37, pos.z * 0.045)).r;
    float coverage = saturate(SETTING_CLOUDS_CU_COVERAGE * 2.0 + moisture * 2.5 - 1.1);
    if (coverage <= 0.0) return 0.0;
    float development = smoothstep(0.05, 0.6, coverage);
    vec3 curl = texture(usam_cumulusCurl, pos * 0.18).xyz;
    vec2 cellPos = pos.xz * CU_CELL_FREQUENCY + curl.xz * 0.12;
    ivec2 cell = ivec2(floor(cellPos));
    vec2 localPos = fract(cellPos);
    float maxReach = 1.90 * 1.05 * coverage + 0.04;
    ivec2 firstCell = ivec2(ceil(localPos - maxReach - 0.82));
    ivec2 lastCell = ivec2(floor(localPos + maxReach - 0.18));
    float shape = -1.0;

    // Plumes spread into a common lower mass instead of sitting on separate base slabs.
    for (int z = firstCell.y; z <= lastCell.y; ++z) {
        for (int x = firstCell.x; x <= lastCell.x; ++x) {
            ivec2 offset = ivec2(x, z);
            vec3 seed = hash_uintToFloat(hash_33_q3(uvec3(uvec2(cell + offset), 0x1919810u)));
            vec2 center = vec2(offset) + mix(vec2(0.18), vec2(0.82), seed.xy);
            vec2 delta = localPos - center;
            float top = mix(0.55, 0.98, smoothstep(0.1, 0.92, seed.z));
            top *= mix(0.25, 1.0, development);
            float radius = coverage * mix(0.78, 1.05, seed.x);
            if (dot(delta, delta) > pow2(radius * 1.90 + 0.04) || heightFraction > top * 1.10) continue;

            float angle = seed.x * (2.0 * PI);
            vec2 axis = vec2(cos(angle), sin(angle));
            vec2 lean = (seed.xy - 0.5) * radius * 0.5 * heightFraction / top;
            vec2 thermalPos = delta - lean;
            float mainLobe = clouds_cu_plume(thermalPos, heightFraction, top, radius * 0.66);
            float sideHeight = top * mix(0.58, 0.85, seed.y);
            float sideLobe = clouds_cu_plume(thermalPos - axis * radius * 0.66, heightFraction, sideHeight, radius * 0.58);
            float youngHeight = top * mix(0.32, 0.65, seed.x);
            float youngLobe = clouds_cu_plume(thermalPos + axis * radius * 0.72, heightFraction, youngHeight, radius * 0.61);
            float lobes = clouds_cu_union(mainLobe, clouds_cu_union(sideLobe, youngLobe, 0.16), 0.14);
            shape = clouds_cu_union(shape, lobes, 0.10);
        }
    }
    if (shape <= 0.0) return 0.0;
    float envelope = smoothstep(0.0, 0.12, shape);

    vec3 billowPos = pos * CU_BILLOW_FREQUENCY + curl * 0.12;
    billowPos.y *= 0.85;
    float billowMass = texture(usam_cumulusDetail1, billowPos * 0.43).r;
    float billow = texture(usam_cumulusDetail1, billowPos).r;
    float smallBillow = texture(usam_cumulusDetail1, billowPos * 2.7 + curl * 0.07).r;
    float entrainment = mix(0.60, 1.0, smoothstep(0.04, 0.24, heightFraction));
    shape += ((billowMass - 0.60) * 0.60 + (billow - 0.60) * 0.42 + (smallBillow - 0.55) * 0.20) * entrainment;
    if (shape <= 0.0) return 0.0;

    // Small entrained eddies modulate a finite transition into the cloud interior.
    vec3 detailPos = pos * CU_DETAIL_FREQUENCY + curl * 0.13;
    float erosion = (1.0 - texture(usam_cumulusDetail2, detailPos).r) * 0.12;
    erosion += (1.0 - texture(usam_cumulusDetail2, detailPos * 3.7).r) * 0.025;
    shape -= erosion * entrainment;
    float surface = smoothstep(0.0, 0.09, shape) * envelope;
    float condensationHeight = 0.008 + (1.0 - billowMass) * 0.06 + (1.0 - billow) * 0.025;
    float condensation = smoothstep(condensationHeight, condensationHeight + 0.045, heightFraction);
    float liquidWater = mix(0.38, 1.20, smoothstep(0.0, 0.38, heightFraction));
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
