#ifndef INCLUDE_clouds_Transport_glsl
#define INCLUDE_clouds_Transport_glsl a

#include "Cumulus.glsl"

/*const*/
#define CT_N SETTING_CLOUDS_MS_RESOLUTION
#if CT_N == 64
#define CT_CELLS 262144
#define CT_TREE 299584
#define CT_GROUPS 4096
#define CT_COARSE_LEVEL 4
#else
#define CT_CELLS 32768
#define CT_TREE 37440
#define CT_GROUPS 512
#define CT_COARSE_LEVEL 3
#endif
#define CT_ANGLES 24
/*const*/

const float CT_W = SPHERE_SOLID_ANGLE / float(CT_ANGLES);
const float CT_STRETCH = 4.79585984; // 2 sinh(stretch) = 121 km from the camera in x/z.
const vec3 CT_EXTINCTION = vec3(0.18067367051236774, 0.18215559145291857, 0.19358580629452912);
const vec3 CT_SCATTERING = vec3(0.18067349140471628, 0.18215551958414714, 0.19358579492341665);
const vec3 CT_ABSORPTION = CT_EXTINCTION - CT_SCATTERING;

struct CTCoefficients {
    vec4 slow; // w Ds, density
    vec4 fast; // (1-w) Df
    vec4 weight;
    vec4 exchange;
};
struct CTPartial {
    vec4 rd;
    vec4 dd;
    vec4 rr;
    vec4 limit;
    vec4 bb;
    vec4 inputPower;
    vec4 outputPower;
};

#ifndef CT_STORAGE
#define CT_STORAGE readonly restrict buffer
#endif
layout(std430, binding = 6) CT_STORAGE CloudTransport {
    vec4 ctData[];
};

// Vec4 offsets preserve the readback layout without large sized SSBO arrays,
// which cause pathological compile times in the NVIDIA driver.
const int CT_EDGES_OFFSET = 32;
const int CT_PHASE_OFFSET = CT_EDGES_OFFSET + CT_N + 1;
const int CT_SUN_PHASE_OFFSET = CT_PHASE_OFFSET + CT_ANGLES * CT_ANGLES;
const int CT_MOON_PHASE_OFFSET = CT_SUN_PHASE_OFFSET + CT_ANGLES;
const int CT_MATERIAL_OFFSET = CT_MOON_PHASE_OFFSET + CT_ANGLES;
const int CT_COEFFICIENTS_OFFSET = CT_MATERIAL_OFFSET + 2 * CT_CELLS;
const int CT_RHS_OFFSET = CT_COEFFICIENTS_OFFSET + 4 * CT_TREE;
const int CT_ERROR_OFFSET = CT_RHS_OFFSET + 2 * CT_TREE;
const int CT_RADIANCE_OFFSET = CT_ERROR_OFFSET + 2 * CT_TREE;
const int CT_SOURCE_OFFSET = CT_RADIANCE_OFFSET + CT_ANGLES * CT_CELLS;
const int CT_SKY_OFFSET = CT_SOURCE_OFFSET + CT_ANGLES * CT_CELLS;
const int CT_LIFTED_OFFSET = CT_SKY_OFFSET + CT_ANGLES * CT_CELLS;
const int CT_PARTIALS_OFFSET = CT_LIFTED_OFFSET + CT_ANGLES * CT_CELLS;

#define ctStats(i) ctData[(i)]
#define ctEdges(i) ctData[CT_EDGES_OFFSET + (i)]
#define ctPhase(i) ctData[CT_PHASE_OFFSET + (i)]
#define ctSunPhase(i) ctData[CT_SUN_PHASE_OFFSET + (i)]
#define ctMoonPhase(i) ctData[CT_MOON_PHASE_OFFSET + (i)]
#define ctMaterialSun(i) ctData[CT_MATERIAL_OFFSET + 2 * (i)]
#define ctMaterialMoon(i) ctData[CT_MATERIAL_OFFSET + 2 * (i) + 1]
#define ctCoefficient(i, field) ctData[CT_COEFFICIENTS_OFFSET + 4 * (i) + (field)]
#define ctRhs(i) ctData[CT_RHS_OFFSET + (i)]
#define ctError(i) ctData[CT_ERROR_OFFSET + (i)]
#define ctRadiance(i) ctData[CT_RADIANCE_OFFSET + (i)]
#define ctSource(i) ctData[CT_SOURCE_OFFSET + (i)]
#define ctSky(i) ctData[CT_SKY_OFFSET + (i)]
#define ctLifted(i) ctData[CT_LIFTED_OFFSET + (i)]
#define ctPartial(i, field) ctData[CT_PARTIALS_OFFSET + 7 * (i) + (field)]

CTCoefficients ct_loadCoefficients(int i) {
    return CTCoefficients(ctCoefficient(i, 0), ctCoefficient(i, 1), ctCoefficient(i, 2), ctCoefficient(i, 3));
}

vec3 ct_toAP0(vec3 value) {
    return colors2_colorspaces_convert(COLORS2_WORKING_COLORSPACE, COLORS2_COLORSPACES_ACES_AP0, value);
}

vec3 ct_phaseCosines(float xi) {
    int row = 0;
    if (xi < SETTING_CLOUDS_CU_PHASE_RATIO) {
        row = 1;
        xi /= SETTING_CLOUDS_CU_PHASE_RATIO;
    } else {
        xi = (xi - SETTING_CLOUDS_CU_PHASE_RATIO) / (1.0 - SETTING_CLOUDS_CU_PHASE_RATIO);
    }
    float x = clamp(xi * 1024.0, 0.5, 1023.5) / 1024.0;
    return texture(usam_cloudTransport, vec2(x, (float(row) + 0.5) / 26.0)).rgb;
}

vec3 ct_boundary(int angle) {
    vec3 boundary = ct_toAP0(persistent_cloudsAmbLUT_fetch(ivec2(1 + angle, 32)).rgb);
    if ((angle & 2) != 0) return boundary;
    const vec3 ground = vec3(SETTING_ATM_GROUND_ALBEDO_R, SETTING_ATM_GROUND_ALBEDO_G, SETTING_ATM_GROUND_ALBEDO_B) / 255.0;
    vec3 albedo = colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_COLORSPACES_ACES_AP0, ground);
    return boundary + 2.0 * albedo * ct_toAP0(persistent_cloudsAmbLUT_fetch(ivec2(0, 32)).rgb);
}

vec3 ct_direction(int angle) {
    int axis = angle / 8;
    vec3 signV = 1.0 - 2.0 * vec3(ivec3(angle) >> ivec3(0, 1, 2) & 1);
    return signV * (vec3(1.0) + vec3(equal(ivec3(axis), ivec3(0, 1, 2)))) * inversesqrt(6.0);
}

int ct_angle(vec3 omega) {
    vec3 a = abs(omega);
    int axis = a.x >= a.y && a.x >= a.z ? 0 : (a.y >= a.z ? 1 : 2);
    return axis * 8 + int(omega.x < 0.0) + 2 * int(omega.y < 0.0) + 4 * int(omega.z < 0.0);
}

int ct_index(ivec3 p, int n) {
    return (p.z * n + p.y) * n + p.x;
}

ivec3 ct_position(int index, int n) {
    return ivec3(index % n, index / n % n, index / (n * n));
}

int ct_levelOffset(int level) {
    return (CT_CELLS - (CT_CELLS >> (3 * level))) * 8 / 7;
}

bool ct_inside(ivec3 p, int n) {
    return all(greaterThanEqual(p, ivec3(0))) && all(lessThan(p, ivec3(n)));
}

vec3 ct_edge(ivec3 p, int level) {
    p <<= level;
    return vec3(ctEdges(p.x).x, ctEdges(p.y).y, ctEdges(p.z).x);
}

vec3 ct_size(ivec3 p, int level) {
    return ct_edge(p + 1, level) - ct_edge(p, level);
}

vec3 ct_center(ivec3 p, int level) {
    return (ct_edge(p, level) + ct_edge(p + 1, level)) * 0.5;
}

float ct_volume(ivec3 p, int level) {
    vec3 h = ct_size(p, level);
    return h.x * h.y * h.z;
}

vec3 ct_worldCenter() {
    return vec3(cameraPosition.x, 0.0, cameraPosition.z) / float(SETTING_ATM_D_SCALE);
}

vec3 ct_bandFactor(vec3 a1, float h) {
    vec3 band = clamp(PI / (h * a1), vec3(0.125), vec3(8.0));
    vec3 x = (log2(band) + 3.0) * (63.0 / 6.0);
    float y = (5.5 + SETTING_CLOUDS_CU_PHASE_RATIO * 20.0) / 26.0;
    return vec3(texture(usam_cloudTransport, vec2((x.x + 0.5) / 1024.0, y)).r,
        texture(usam_cloudTransport, vec2((x.y + 0.5) / 1024.0, y)).g,
        texture(usam_cloudTransport, vec2((x.z + 0.5) / 1024.0, y)).b);
}

CTCoefficients ct_coefficients(float density, float h) {
    CTCoefficients c;
    c.slow = vec4(0.0, 0.0, 0.0, density);
    c.fast = vec4(0.0);
    c.weight = vec4(0.0);
    c.exchange = vec4(0.0);
    if (density == 0.0) return c;
    vec3 beta1 = mix(texelFetch(usam_cloudTransport, ivec2(0, 4), 0).rgb, texelFetch(usam_cloudTransport, ivec2(3, 4), 0).rgb, SETTING_CLOUDS_CU_PHASE_RATIO);
    vec3 beta2 = mix(texelFetch(usam_cloudTransport, ivec2(1, 4), 0).rgb, texelFetch(usam_cloudTransport, ivec2(4, 4), 0).rgb, SETTING_CLOUDS_CU_PHASE_RATIO);
    vec3 beta3 = mix(texelFetch(usam_cloudTransport, ivec2(2, 4), 0).rgb, texelFetch(usam_cloudTransport, ivec2(5, 4), 0).rgb, SETTING_CLOUDS_CU_PHASE_RATIO);
    vec3 a1 = density * (CT_EXTINCTION - CT_SCATTERING * beta1);
    vec3 a3 = density * (CT_EXTINCTION - CT_SCATTERING * beta3) * ct_bandFactor(a1, h);
    vec3 d0 = 1.0 / (3.0 * a1);
    vec3 sumD = 3.0 / (5.0 * a1) + 9.0 / (35.0 * a3);
    vec3 productD = 3.0 / (35.0 * a1 * a3);
    vec3 df = (sumD + sqrt(sumD * sumD - 4.0 * productD)) * 0.5;
    vec3 ds = productD / df;
    vec3 w = (df - d0) / (df - ds);
    c.slow.rgb = w * ds;
    c.fast.rgb = (1.0 - w) * df;
    c.weight.rgb = w;
    c.exchange.rgb = w * (1.0 - w) * density * CT_SCATTERING * (1.0 - beta2);
    return c;
}

// Integrated finite-volume conductance. Vacuum has an escape boundary in the
// preconditioner; it remains a zero-extinction streaming cell in transport.
void ct_face(ivec3 p, int level, ivec3 offset, out vec3 gs, out vec3 gf) {
    int n = CT_N >> level;
    int base = ct_levelOffset(level);
    CTCoefficients a = ct_loadCoefficients(base + ct_index(p, n));
    ivec3 q = p + offset;
    vec3 h = ct_size(p, level);
    float width = dot(h, abs(vec3(offset)));
    float area = h.x * h.y * h.z / width;
    float widthB = 0.0;
    CTCoefficients b;
    b.slow = vec4(0.0);
    if (ct_inside(q, n)) {
        b = ct_loadCoefficients(base + ct_index(q, n));
        widthB = dot(ct_size(q, level), abs(vec3(offset)));
    }
    if (b.slow.a > 0.0) {
        gs = area / (0.5 * width / a.slow.rgb + 0.5 * widthB / b.slow.rgb);
        gf = area / (0.5 * width / a.fast.rgb + 0.5 * widthB / b.fast.rgb);
    } else {
        gs = area / (0.5 * width / a.slow.rgb + 2.0 / a.weight.rgb);
        gf = area / (0.5 * width / a.fast.rgb + 2.0 / (1.0 - a.weight.rgb));
    }
}

ivec3 ct_neighbor(int face) {
    return ivec3(equal(ivec3(face / 2), ivec3(0, 1, 2))) * (1 - 2 * (face & 1));
}

void ct_diagonal(ivec3 p, int level, out vec3 ds, out vec3 df, out vec3 exchange) {
    int i = ct_levelOffset(level) + ct_index(p, CT_N >> level);
    CTCoefficients c = ct_loadCoefficients(i);
    float volume = ct_volume(p, level);
    exchange = c.exchange.rgb * volume;
    ds = c.weight.rgb * CT_ABSORPTION * c.slow.a * volume + exchange;
    df = (1.0 - c.weight.rgb) * CT_ABSORPTION * c.slow.a * volume + exchange;
    for (int f = 0; f < 6; ++f) {
        vec3 gs, gf;
        ct_face(p, level, ct_neighbor(f), gs, gf);
        ds += gs;
        df += gf;
    }
}

void ct_apply(ivec3 p, int level, out vec3 as, out vec3 af) {
    int n = CT_N >> level;
    int base = ct_levelOffset(level);
    int i = base + ct_index(p, n);
    as = vec3(0.0);
    af = vec3(0.0);
    if (ctCoefficient(i, 0).a == 0.0) return;
    vec3 ds, df, exchange;
    ct_diagonal(p, level, ds, df, exchange);
    as = ds * ctError(2 * i).rgb - exchange * ctError(2 * i + 1).rgb;
    af = df * ctError(2 * i + 1).rgb - exchange * ctError(2 * i).rgb;
    for (int f = 0; f < 6; ++f) {
        ivec3 offset = ct_neighbor(f);
        ivec3 q = p + offset;
        if (!ct_inside(q, n)) continue;
        int j = base + ct_index(q, n);
        if (ctCoefficient(j, 0).a == 0.0) continue;
        vec3 gs, gf;
        ct_face(p, level, offset, gs, gf);
        as -= gs * ctError(2 * j).rgb;
        af -= gf * ctError(2 * j + 1).rgb;
    }
}

void ct_blockSolve(vec3 bs, vec3 bf, vec3 ds, vec3 df, vec3 exchange, out vec3 xs, out vec3 xf) {
    vec3 determinant = ds * df - exchange * exchange;
    xs = (bs * df + bf * exchange) / determinant;
    xf = (bf * ds + bs * exchange) / determinant;
}

vec3 ct_delta(ivec3 p) {
    if (!ct_inside(p, CT_N)) return vec3(0.0);
    int i = ct_index(p, CT_N);
    if (ctCoefficient(i, 0).a == 0.0) return vec3(0.0);
    vec3 w = ctCoefficient(i, 2).rgb;
    return (w * ctError(2 * i).rgb + (1.0 - w) * ctError(2 * i + 1).rgb) * UNIFORM_PHASE;
}

vec3 ct_field(int index, bool correction) {
    return correction ? ctLifted(index).rgb : ctRadiance(index).rgb;
}

vec3 ct_streaming(ivec3 p, int angle, bool correction) {
    vec3 omega = ct_direction(angle);
    vec3 rate = abs(omega) / ct_size(p, 0);
    int i = ct_index(p, CT_N);
    vec3 result = (ctCoefficient(i, 0).a * CT_EXTINCTION + rate.x + rate.y + rate.z) * ct_field(angle * CT_CELLS + i, correction);
    for (int axis = 0; axis < 3; ++axis) {
        ivec3 offset = ivec3(equal(ivec3(axis), ivec3(0, 1, 2))) * ivec3(sign(omega));
        ivec3 q = p - offset;
        if (ct_inside(q, CT_N)) result -= rate[axis] * ct_field(angle * CT_CELLS + ct_index(q, CT_N), correction);
    }
    return result;
}

vec3 ct_external(int i, int angle) {
    return CT_SCATTERING * (ctCoefficient(i, 0).a * ctSky(angle * CT_CELLS + i).rgb
        + ctMaterialSun(i).rgb * ctSunPhase(angle).rgb + ctMaterialMoon(i).rgb * ctMoonPhase(angle).rgb);
}

// Positive, cubic-symmetric angular interpolation on the convex hull of the
// 24 (2,1,1) directions. Each basis integrates to 4pi/24 by symmetry.
void ct_angularWeights(vec3 omega, out ivec4 angles, out vec4 weights) {
    vec3 a = abs(omega);
    int hi = a.x >= a.y && a.x >= a.z ? 0 : (a.y >= a.z ? 1 : 2);
    int lo = a.x <= a.y && a.x <= a.z ? 0 : (a.y <= a.z ? 1 : 2);
    if (hi == lo) lo = (hi + 1) % 3;
    int mid = 3 - hi - lo;
    int signs = int(omega.x < 0.0) + 2 * int(omega.y < 0.0) + 4 * int(omega.z < 0.0);
    float axisT = 2.0 / a[hi];
    float pairT = 3.0 / (a[hi] + a[mid]);
    float triT = 4.0 / (a.x + a.y + a.z);
    if (triT <= min(axisT, pairT)) {
        angles = ivec4(signs, signs + 8, signs + 16, signs);
        weights = vec4(a * triT - 1.0, 0.0);
    } else if (pairT <= axisT) {
        float u = a[hi] * pairT - 1.0;
        float v = (omega[lo] * pairT + 1.0) * 0.5;
        int positive = signs & ~(1 << lo);
        int negative = positive | (1 << lo);
        angles = ivec4(8 * mid + negative, 8 * hi + negative, 8 * mid + positive, 8 * hi + positive);
        weights = vec4((1.0-u)*(1.0-v), u*(1.0-v), (1.0-u)*v, u*v);
    } else {
        int b = (hi + 1) % 3;
        int c = (hi + 2) % 3;
        float u = (omega[b] * axisT + 1.0) * 0.5;
        float v = (omega[c] * axisT + 1.0) * 0.5;
        int base = 8 * hi + (signs & (1 << hi));
        angles = ivec4(base | (1 << b) | (1 << c), base | (1 << c), base | (1 << b), base);
        weights = vec4((1.0-u)*(1.0-v), u*(1.0-v), (1.0-u)*v, u*v);
    }
}

vec3 ct_sample(vec3 pos, ivec4 angles, vec4 angularWeights) {
    pos -= ct_worldCenter();
    vec2 horizontal = asinh(pos.xz * 0.5) / CT_STRETCH;
    float y0 = ctEdges(0).y;
    float y1 = ctEdges(CT_N).y;
    vec3 grid = vec3((horizontal.x * 0.5 + 0.5) * CT_N, (pos.y - y0) / (y1 - y0) * CT_N, (horizontal.y * 0.5 + 0.5) * CT_N);
    // Interpolate between physical centers, not the inverse edge mapping.
    if (any(lessThan(grid, vec3(0.0))) || any(greaterThan(grid, vec3(CT_N)))) return vec3(0.0);
    ivec3 cell = clamp(ivec3(floor(grid)), ivec3(0), ivec3(CT_N - 1));
    ivec3 base = clamp(cell - ivec3(lessThan(pos, ct_center(cell, 0))), ivec3(0), ivec3(CT_N - 2));
    vec3 lower = ct_center(base, 0);
    vec3 upper = ct_center(base + 1, 0);
    vec3 f = clamp((pos - lower) / (upper - lower), vec3(0.0), vec3(1.0));
    vec3 result = vec3(0.0);
    for (int corner = 0; corner < 8; ++corner) {
        ivec3 offset = ivec3(corner) >> ivec3(0, 1, 2) & 1;
        vec3 w = mix(1.0 - f, f, vec3(offset));
        int i = ct_index(base + offset, CT_N);
        vec3 value = vec3(0.0);
        for (int a = 0; a < 4; ++a) value += angularWeights[a] * ctSource(angles[a] * CT_CELLS + i).rgb;
        result += value * w.x * w.y * w.z;
    }
    return colors2_constants_toWorkSpace(result);
}

#endif
