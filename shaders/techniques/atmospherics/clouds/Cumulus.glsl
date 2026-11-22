#ifndef INCLUDE_clouds_Cumulus_glsl
#define INCLUDE_clouds_Cumulus_glsl

#include "/util/Hash.glsl"

const float CU_CELL_SIZE = 4.2 * exp2(-SETTING_CLOUDS_LOW_BASE_FREQ);
const float CU_RANGE = 50.0;
const float CU_DENSITY = 96.0 * SETTING_CLOUDS_CU_DENSITY;

// Geometry uses kilometres relative to sea level, independently of the air LUT's artistic scale.
vec3 clouds_cu_origin(vec3 sceneOffset) {
    dvec2 xz = (dvec2(cameraPositionInt.xz) + dvec2(cameraPositionFract.xz)) / double(1000);
    #ifdef SETTING_CLOUDS_CU_WIND
    double seconds = double(worldDay) * double(1200) + double(worldTime) / double(20);
    xz -= seconds * double(exp2(SETTING_CLOUDS_CU_WIND_SPEED)) * dvec2(6, 4) / double(1000);
    #endif
    // Both cell lattices and the regional pattern repeat on this period. Reduce before converting to float.
    vec2 local = vec2(mod(xz, double(CU_CELL_SIZE) * double(256)));
    float altitude = float(cameraPositionInt.y - 62) + cameraPositionFract.y;
    return vec3(local.x, altitude * 0.001, local.y) + sceneOffset * 0.001;
}

vec2 clouds_cu_altitudes() {
    return max(vec2(0.0), SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS * vec2(-0.19, 1.41));
}

bool clouds_cu_interval(vec3 origin, vec3 direction, float limit, out vec2 interval) {
    vec2 heights = clouds_cu_altitudes();
    interval = vec2(0.0, min(limit, CU_RANGE));
    if (SETTING_CLOUDS_CU_THICKNESS <= 0.0 || SETTING_CLOUDS_CU_COVERAGE <= 0.0 || CU_DENSITY <= 0.0) return false;
    if (direction.y == 0.0) return origin.y >= heights.x && origin.y <= heights.y && interval.y > 0.0;
    vec2 t = (heights - origin.y) / direction.y;
    interval.x = max(interval.x, min(t.x, t.y));
    interval.y = min(interval.y, max(t.x, t.y));
    return interval.x < interval.y;
}

struct CuField {
    float bound;
    float density;
    float bulk;
    float height;
    float development;
    vec3 normal;
};

// A thermal grows from the common condensation floor. A developed updraft leans
// and branches above its stem. Shallow thermals spread and end close to the floor.
float clouds_cu_thermal(vec3 p, float development, float variation) {
    float rise = smoothstep(0.10, 0.77, p.y);
    p.xz -= rise * vec2(mix(0.10, 0.55, variation) * development, (variation - 0.5) * 0.48);
    float root = 1.0 - length((p - vec3(0.0, 0.12, 0.0)) / vec3(1.04, 0.46, 0.96));
    float waist = mix(0.80, 0.55 + 0.12 * variation, development);
    float stem = 1.0 - length((p - vec3(0.04, 0.40, -0.03)) / vec3(waist, 0.44, waist));
    vec3 side = p - vec3(-0.48, 0.29 + 0.12 * variation, 0.13);
    float shoulder = 1.0 - length(side / vec3(0.52, 0.27, 0.51));
    side = p - vec3(0.12 + 0.23 * variation, 0.47, 0.40);
    shoulder = max(shoulder, 1.0 - length(side / vec3(0.44, 0.21, 0.48)));
    vec3 q = p - vec3(0.10 + 0.14 * variation, 0.62, 0.04);
    float head = 1.0 - length(q / vec3(0.70, 0.26, 0.65));
    vec3 bud = q - vec3(0.26, 0.13 + 0.07 * variation, -0.16);
    head = max(head, 1.0 - length(bud / vec3(0.46, 0.19, 0.44)));
    bud = q - vec3(-0.26, 0.15 - 0.09 * variation, 0.24);
    head = max(head, 1.0 - length(bud / vec3(0.40, 0.17, 0.40)));
    bud = q - vec3(-0.32, 0.03, -0.26);
    head = max(head, 1.0 - length(bud / vec3(0.46, 0.22, 0.45)));
    float column = max(max(root, shoulder), stem);
    float blend = max(0.22 - abs(column - head), 0.0);
    return max(column, head) + blend * blend;
}

float clouds_cu_thermalEnvelope(vec3 p) {
    return 2.0 * (1.0 - length((p - vec3(0.16, 0.34, 0.0)) / vec3(1.45, 0.84, 1.31))) - 0.003;
}

// Relief above a shared shallow layer. Each turret joins the body down to its local floor.
float clouds_cu_shallowBud(vec2 uv) {
    vec2 variation = hash_uintToFloat(hash_22_q1(uvec2(ivec2(floor(uv)))));
    float radius = mix(0.23, 0.47, variation.x);
    vec2 delta = (fract(uv) - 0.5 - (0.48 - radius) * (2.0 * variation.yx - 1.0)) / radius;
    delta.y *= 1.0 + 0.55 * variation.y;
    float dome = max(1.0 - dot(delta, delta), 0.0);
    return dome * mix(0.07, 0.47, variation.y * variation.y);
}

void clouds_cu_cluster(vec3 pos, vec3 direction, bool lower, float lod, bool density, inout CuField field) {
    float size = lower ? CU_CELL_SIZE * 0.5 : CU_CELL_SIZE;
    vec2 grid = pos.xz / size + (lower ? vec2(0.37, 0.61) : vec2(0.0));
    // Select the forward cell at an exact face. Every possible footprint is inset from these faces.
    vec2 cell = floor(grid + sign(direction.xz) * 0.000002);
    vec2 uv = grid - cell;
    vec2 toFace = mix(uv, 1.0 - uv, greaterThan(direction.xz, vec2(0.0)));
    vec2 exitDistance = toFace * size / max(abs(direction.xz), vec2(1e-20));
    if (direction.x != 0.0) field.bound = min(field.bound, max(exitDistance.x, 0.0));
    if (direction.z != 0.0) field.bound = min(field.bound, max(exitDistance.y, 0.0));

    uvec2 id = uvec2(ivec2(cell)) & uvec2(lower ? 511u : 255u);
    vec4 seed = hash_uintToFloat(hash_44_q1(uvec4(id + uvec2(871u, 2137u), lower ? 43u : 17u, 521u)));
    // Oblique, anisotropic coordinates group candidates into finite, broken streets.
    vec2 region = (vec2(id) + 0.5) * (lower ? 0.5 : 1.0);
    vec2 bandUV = vec2(dot(region, vec2(2.0, 1.0)), dot(region, vec2(-3.0, 6.0))) / 256.0;
    float band = textureLod(usam_cumulusBase, bandUV, 0.0).r;
    float coverage = SETTING_CLOUDS_CU_COVERAGE;
    float chance = clamp(coverage * (lower ? 0.96 : 1.34) + (band - 0.35) * 1.65, 0.0, 1.0);
    if (seed.w >= chance) return;

    float growth = smoothstep(0.15, 0.87, seed.z + (band - 0.35) * 0.54);
    float radius = size * mix(0.25, 0.48, mix(seed.y, coverage, 0.22));
    float margin = radius / size + 0.012;
    vec2 center = mix(vec2(margin), vec2(1.0 - margin), seed.xy);
    vec2 delta = (uv - center) * size;
    float base = max(0.0, SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS *
        (lower ? mix(-0.19, -0.035, band) : mix(0.015, 0.25, band)));
    float height = SETTING_CLOUDS_CU_THICKNESS * (lower ? mix(0.14, 0.40, growth) : mix(0.26, 1.16, growth * growth));
    float y = pos.y - base;
    float parentBox = max(length(delta) - radius, max(-y, y - height));
    if (parentBox > 0.0) {
        // Ray entry into the enclosing cylinder can be much farther than its
        // point distance, especially when traversing underneath a row of clouds.
        float entry = 0.0;
        if (y < 0.0 || y > height) {
            if (direction.y == 0.0 || (y < 0.0) != (direction.y > 0.0)) return;
            entry = ((y < 0.0 ? 0.0 : height) - y) / direction.y;
        }
        float c = dot(delta, delta) - radius * radius;
        if (c > 0.0) {
            float b = dot(delta, direction.xz);
            float disc = b * b - c * dot(direction.xz, direction.xz);
            if (b >= 0.0 || disc <= 0.0) return;
            entry = max(entry, c / (-b + sqrt(disc)));
        }
        field.bound = min(field.bound, entry);
        return;
    }

    field.bound = min(field.bound, parentBox);
    if (!density || parentBox >= 0.0) return;

    float angle = 6.2831853 * seed.x;
    vec2 axis = vec2(cos(angle), sin(angle));
    vec2 across = vec2(-axis.y, axis.x);
    vec3 p = vec3(dot(delta, axis) / radius, y / height, dot(delta, across) / radius);

    // Local coordinates keep all noises attached to their parent across tile and world-time wraps.
    vec3 noisePos = vec3(dot(delta, axis), y, dot(delta, across)) + 8.0 * seed.xyz;
    vec3 curl = textureLod(usam_cumulusCurl, noisePos * (0.57 * exp2(SETTING_CLOUDS_LOW_CURL_FREQ)), 0.0).xyz;
    vec3 drift = curl * (0.15 * exp2(SETTING_CLOUDS_LOW_BILLOWY_CURL_STR));
    #ifdef SETTING_CLOUDS_CU_WIND
    float phase = (float((worldDay % 4 + 4) % 4) + float(worldTime) / 24000.0) * (6.2831853 * 0.25);
    drift += 0.045 * sin(phase + 6.2831853 * seed.yzx);
    #endif
    vec3 warp = drift * vec3(0.76, 0.45, 0.76);
    float tall = lower ? 0.25 * growth : growth;
    vec3 mainSize = vec3(0.56, 1.0, 0.64);
    vec3 sideSize = vec3(0.56, mix(0.80, 0.38 + 0.29 * seed.x, tall), 0.53);
    vec3 youngSize = vec3(0.44, mix(0.49, 0.76, seed.y) * (1.0 - 0.36 * tall), 0.42);
    vec3 mainParent = (p - vec3(0.02 + 0.17 * seed.y, 0.0, -0.10)) / mainSize;
    vec3 sideParent = (p - vec3(-0.36, 0.0, 0.14)) / sideSize;
    vec3 youngParent = (p - vec3(0.44, 0.0, 0.17)) / youngSize;
    vec3 envelopes = vec3(clouds_cu_thermalEnvelope(mainParent), clouds_cu_thermalEnvelope(sideParent.zyx),
        clouds_cu_thermalEnvelope(youngParent));
    vec3 main = mainParent - warp / mainSize;
    vec3 side = sideParent - warp / sideSize;
    vec3 young = youngParent - warp / youngSize;
    vec3 thermals = min(envelopes, vec3(clouds_cu_thermal(main, tall, seed.x), clouds_cu_thermal(side.zyx, tall, seed.y),
        clouds_cu_thermal(young, tall, seed.z)));
    float shape = max(thermals.x, max(thermals.y, thermals.z));
    float foot = 1.0 - min(length(main.xz), min(length(side.xz), length(young.xz)));
    float shallow = (1.0 - smoothstep(0.22, 0.49, height / max(float(SETTING_CLOUDS_CU_THICKNESS), 0.001)))
        * (1.0 - smoothstep(0.53, 0.99, height / radius));
    const vec2 windDirection = vec2(0.83205, 0.55470);
    float alongWind = dot(delta, windDirection);
    float leeward = smoothstep(-0.28, 0.62, alongWind / radius + 0.19 * curl.x);
    float rim = 1.0 - smoothstep(0.10, 0.66, foot);
    float entrained = shallow * rim * (0.22 + 0.78 * leeward);
    if (shallow > 0.0) {
        float h = p.y + 0.16 * rim + 0.055 * entrained;
        vec2 uv = vec2(dot(delta, axis), dot(delta, across)) / height + 12.0 * seed.xy + 0.27 * curl.xz;
        vec2 smaller = mat2(0.8, 0.6, -0.6, 0.8) * uv / 0.68 + 0.43;
        float buds = max(clouds_cu_shallowBud(uv), 0.55 * clouds_cu_shallowBud(smaller));
        float activity = smoothstep(-0.10, 0.55, foot + 0.28 * curl.y - 0.22 * leeward);
        float top = 0.29 + 0.24 * foot + 0.13 * curl.y + buds * mix(0.35, 1.0, activity);
        shape = mix(shape, min(1.28 * foot, 2.5 * (top - h)), shallow);
    }
    if (shape <= 0.0) return;

    float billow = textureLod(usam_cumulusDetail1, noisePos * (0.75 * exp2(SETTING_CLOUDS_LOW_BILLOWY_FREQ)) + drift, 0.0).r;
    vec2 localWind = vec2(dot(axis, windDirection), dot(across, windDirection));
    vec3 edgePos = noisePos;
    edgePos.xz -= localWind * (0.62 * shallow * alongWind);
    float fine = textureLod(usam_cumulusDetail2, edgePos * (2.6 * exp2(SETTING_CLOUDS_HIGH_BILLOWY_FREQ))
        + curl * (0.11 * exp2(SETTING_CLOUDS_HIGH_BILLOWY_CURL_STR)), 0.0).r;
    float wisp = textureLod(usam_cumulusDetail2, edgePos * (5.1 * exp2(SETTING_CLOUDS_LOW_WISPS_FREQ))
        + curl * (0.05 * exp2(SETTING_CLOUDS_LOW_WISPS_CURL_STR)), 0.0).r;
    float erosion = (1.0 - sqrt(billow)) * 0.57 + (1.0 - fine) * 0.12;
    // The coarse edge is shared at every LOD. Finer evaluation only removes material.
    erosion += 0.018 + (1.0 - lod) * (1.0 - wisp) * mix(0.012, 0.065, entrained);
    erosion *= mix(0.63, 1.0, smoothstep(0.05, 0.45, p.y));
    float moistPocket = smoothstep(0.20, 0.46, billow + 0.48 * (fine - 0.28));
    erosion *= 1.0 - 0.40 * entrained * moistPocket;
    erosion += entrained * mix(0.60, 0.025, moistPocket);
    float floorHeight = min(0.16, (0.018 + 0.022 * curl.z + 0.065 * (1.0 - sqrt(billow))) / max(height, 0.001));
    floorHeight += entrained * (0.025 + 0.14 * (1.0 - moistPocket));
    float outer = min(-parentBox / min(radius, height) - 0.0005, (p.y - floorHeight) * 1.8);
    outer = min(outer, max(envelopes.x, max(envelopes.y, envelopes.z)));
    float bulk = clamp(min(shape, outer) / 0.15, 0.0, 1.0);
    float rho = clamp(min(shape - erosion, outer) / 0.15, 0.0, 1.0) * mix(0.44, 1.15, smoothstep(0.015, 0.35, p.y));
    float wetCore = smoothstep(0.11, 0.72, foot + 0.22 * (billow - 0.4) - 0.16 * leeward);
    float moisture = mix(1.0, mix(0.12 + 0.19 * billow, 1.0, wetCore), shallow);
    rho *= moisture;
    bulk *= moisture;
    if (rho > field.density) {
        field.density = rho;
        field.bulk = bulk;
        field.height = p.y;
        field.development = height / max(float(SETTING_CLOUDS_CU_THICKNESS), 0.001);
        vec2 slope = (axis * main.x + across * main.z) * (height / radius);
        field.normal = normalize(vec3(slope.x, 1.0, slope.y));
    }
}

CuField clouds_cu_field(vec3 pos, vec3 direction, float lod, bool density) {
    CuField field = CuField(CU_RANGE, 0.0, 0.0, 0.0, 0.0, vec3(0.0, 1.0, 0.0));
    clouds_cu_cluster(pos, direction, false, lod, density, field);
    clouds_cu_cluster(pos, direction, true, lod, density, field);
    return field;
}

// Distribute the remaining budget by the fourth power of distance fraction. A proven empty prefix can
// advance farther. The last iteration always accounts for the remaining interval.
float clouds_cu_bin(vec3 pos, vec3 direction, float target, float remaining, int i, int count, out float empty) {
    float d = clouds_cu_field(pos, direction, 0.0, false).bound;
    const float edgeStep = 0.42 / float(SETTING_CLOUDS_LOW_STEP_MIN);
    empty = clamp(d, 0.0, remaining);
    float allocated = remaining * (float(2 * i + 1) / float((count - i) * (count + i)));
    allocated *= float(i * i + (i + 1) * (i + 1)) / float(count * count + i * i);
    float wanted = d > edgeStep ? empty : min(target, edgeStep + 0.48 * abs(d));
    return min(remaining, max(allocated, wanted));
}

#endif
