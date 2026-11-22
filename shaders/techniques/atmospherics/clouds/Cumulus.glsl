#ifndef INCLUDE_clouds_Cumulus_glsl
#define INCLUDE_clouds_Cumulus_glsl

#include "/util/Hash.glsl"

const float CU_CELL_SIZE = 4.8 * exp2(-SETTING_CLOUDS_LOW_BASE_FREQ);
const float CU_RANGE = 50.0;
const float CU_DENSITY = 96.0 * SETTING_CLOUDS_CU_DENSITY;

// Geometry uses kilometres relative to sea level, independently of the air LUT's artistic scale.
vec3 clouds_cu_origin(vec3 sceneOffset) {
    dvec2 xz = (dvec2(cameraPositionInt.xz) + dvec2(cameraPositionFract.xz)) * 0.001lf;
    #ifdef SETTING_CLOUDS_CU_WIND
    double seconds = double(worldDay) * 1200.0lf + double(worldTime) * 0.05lf;
    xz -= seconds * double(exp2(SETTING_CLOUDS_CU_WIND_SPEED)) * dvec2(0.006, 0.004);
    #endif
    // Both cell lattices and the regional pattern repeat on this period. Reduce before converting to float.
    vec2 local = vec2(mod(xz, double(CU_CELL_SIZE) * 256.0lf));
    float altitude = float(cameraPositionInt.y - 62) + cameraPositionFract.y;
    return vec3(local.x, altitude * 0.001, local.y) + sceneOffset * 0.001;
}

vec2 clouds_cu_altitudes() {
    return max(vec2(0.0), SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS * vec2(-0.19, 1.28));
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

float clouds_cu_bud(vec3 p) {
    float parent = 1.0 - length(p);
    if (parent <= 0.0) return 0.0;
    float core = 1.0 - length((p + vec3(0.0, 0.17, 0.0)) / vec3(0.79, 0.81, 0.81));
    float a = 1.0 - length((p - vec3(-0.36, 0.25, -0.20)) / vec3(0.51, 0.56, 0.54));
    float b = 1.0 - length((p - vec3(0.34, 0.18, 0.21)) / vec3(0.54, 0.54, 0.52));
    float c = 1.0 - length((p - vec3(0.04, 0.57, -0.09)) / vec3(0.46, 0.45, 0.51));
    return min(parent - 0.004, max(max(core, c), max(a, b)));
}

float clouds_cu_crown(vec3 p, float parentInterior, vec3 drift) {
    if (parentInterior <= 0.0) return 0.0;
    vec3 q = p - drift;
    q.x -= 0.11 * (p.y + 0.2);
    float core = 1.0 - length((q - vec3(0.0, -0.19, 0.0)) / vec3(0.88, 0.86, 0.86));
    float left = clouds_cu_bud((q - vec3(-0.39, 0.25, -0.16)) / vec3(0.57, 0.53, 0.62));
    float right = clouds_cu_bud((q - vec3(0.36, 0.22, 0.20)) / vec3(0.55, 0.62, 0.60));
    float top = clouds_cu_bud((q - vec3(-0.08, 0.54, 0.11)) / vec3(0.51, 0.49, 0.52));
    return min(parentInterior - 0.004, max(max(core, top), max(left, right)));
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
    float chance = clamp(coverage * (lower ? 1.45 : 1.20) + (band - 0.35) * 1.55, 0.0, 1.0);
    if (seed.w >= chance) return;

    float growth = smoothstep(0.18, 0.94, seed.z + (band - 0.35) * 0.42);
    float radius = size * mix(0.17, 0.42, mix(seed.y, coverage, 0.36));
    float margin = radius / size + 0.026;
    vec2 center = mix(vec2(margin), vec2(1.0 - margin), seed.xy);
    vec2 delta = (uv - center) * size;
    float base = max(0.0, SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS *
        (lower ? mix(-0.19, -0.035, band) : mix(0.015, 0.25, band)));
    float height = SETTING_CLOUDS_CU_THICKNESS * (lower ? mix(0.085, 0.32, growth) : mix(0.13, 1.0, growth * growth));
    float y = pos.y - base;
    float parentBox = max(length(delta) - radius, max(-y, y - height));
    if (parentBox > 0.0) {
        field.bound = min(field.bound, parentBox);
        return;
    }

    float angle = 6.2831853 * seed.x;
    vec2 axis = vec2(cos(angle), sin(angle));
    vec2 across = vec2(-axis.y, axis.x);
    vec3 p = vec3(dot(delta, axis) / radius, y / height, dot(delta, across) / radius);
    float sideTop = mix(0.44, 0.79, seed.y);
    vec3 qb = (p - vec3(0.0, 0.10, 0.0)) / vec3(1.0, 0.19, 0.72);
    vec3 qm = (p - vec3(0.18 + 0.10 * seed.y, 0.50, 0.02)) / vec3(0.45, 0.51, 0.43);
    vec3 qs = (p - vec3(-0.32, 0.5 * sideTop, -0.06)) / vec3(0.42, 0.51 * sideTop, 0.39);
    vec3 qy = (p - vec3(0.49, 0.21, -0.13)) / vec3(0.37, 0.25, 0.34);
    vec4 interior = 1.0 - vec4(length(qb), length(qm), length(qs), length(qy));
    vec4 shortestAxis = min(radius * vec4(0.72, 0.43, 0.39, 0.34), height * vec4(0.19, 0.51, 0.51 * sideTop, 0.25));
    vec4 distances = -interior * shortestAxis;
    float bound = max(parentBox, min(min(distances.x, distances.y), min(distances.z, distances.w)));
    field.bound = min(field.bound, bound);
    if (bound >= 0.0 || !density) return;

    // Local coordinates keep all noises attached to their parent across tile and world-time wraps.
    vec3 noisePos = vec3(dot(delta, axis), y, dot(delta, across)) + 8.0 * seed.xyz;
    vec3 curl = textureLod(usam_cumulusCurl, noisePos * (0.57 * exp2(SETTING_CLOUDS_LOW_CURL_FREQ)), 0.0).xyz;
    vec3 drift = curl * (0.07 * exp2(SETTING_CLOUDS_LOW_BILLOWY_CURL_STR));
    #ifdef SETTING_CLOUDS_CU_WIND
    float phase = (float((worldDay % 4 + 4) % 4) + float(worldTime) / 24000.0) * (6.2831853 * 0.25);
    drift += 0.045 * sin(phase + 6.2831853 * seed.yzx);
    #endif
    float floorHeight = 0.025 + 0.020 * curl.z;
    float baseShape = min(interior.x - 0.004, (p.y - floorHeight) / 0.19);
    float shape = max(baseShape, clouds_cu_crown(qm, interior.y, drift));
    shape = max(shape, clouds_cu_crown(qs, interior.z, -drift.zxy));
    shape = max(shape, clouds_cu_crown(qy, interior.w, drift.yzx));
    if (shape <= 0.0) return;

    float billow = textureLod(usam_cumulusDetail1, noisePos * (0.75 * exp2(SETTING_CLOUDS_LOW_BILLOWY_FREQ)) + drift, 0.0).r;
    float fine = textureLod(usam_cumulusDetail2, noisePos * (2.6 * exp2(SETTING_CLOUDS_HIGH_BILLOWY_FREQ))
        + curl * (0.11 * exp2(SETTING_CLOUDS_HIGH_BILLOWY_CURL_STR)), 0.0).r;
    float wisp = textureLod(usam_cumulusDetail2, noisePos * (5.1 * exp2(SETTING_CLOUDS_LOW_WISPS_FREQ))
        + curl * (0.05 * exp2(SETTING_CLOUDS_LOW_WISPS_CURL_STR)), 0.0).r;
    float erosion = (1.0 - sqrt(billow)) * 0.24 + (1.0 - fine) * 0.062;
    erosion += mix((1.0 - wisp) * 0.025, 0.018, lod);
    float outer = -parentBox / min(radius, height) - 0.0005;
    float bulk = clamp(min(shape, outer) / 0.11, 0.0, 1.0);
    float rho = clamp(min(shape - erosion, outer) / 0.11, 0.0, 1.0) * mix(0.55, 1.15, smoothstep(0.015, 0.35, p.y));
    if (rho > field.density) {
        field.density = rho;
        field.bulk = bulk;
        field.height = p.y;
        field.development = height / max(float(SETTING_CLOUDS_CU_THICKNESS), 0.001);
        vec2 slope = (axis * qm.x + across * qm.z) * (height / radius);
        field.normal = normalize(vec3(slope.x, 1.0, slope.y));
    }
}

CuField clouds_cu_field(vec3 pos, vec3 direction, float lod, bool density) {
    CuField field = CuField(CU_RANGE, 0.0, 0.0, 0.0, 0.0, vec3(0.0, 1.0, 0.0));
    clouds_cu_cluster(pos, direction, false, lod, density, field);
    clouds_cu_cluster(pos, direction, true, lod, density, field);
    return field;
}

// Positive bounds skip empty space. Near a silhouette the integration bin shrinks before its jittered probe.
float clouds_cu_bin(vec3 pos, vec3 direction, float target, out bool empty) {
    float d = clouds_cu_field(pos, direction, 0.0, false).bound;
    const float edgeStep = 0.42 / float(SETTING_CLOUDS_LOW_STEP_MIN);
    empty = d > edgeStep;
    return empty ? d : min(target, edgeStep + 0.48 * abs(d));
}

#endif
