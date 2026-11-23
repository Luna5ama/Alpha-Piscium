#include "/util/Colors.glsl"

// Only setup uses the scratch area following the existing global block.
// Each of three 49 x 49 faces stores 11 control points plus implicit black and white.
layout(std430, binding = 0) coherent buffer SkibidiSetup {
    layout(offset = 16384) uvec2 coefficients[79233];
    uvec2 weightedLoss[4608];
};
layout(r32f) uniform coherent image2D uimg_fr32f;

#define SKIBIDI_GAMUT_SETUP
#include "/techniques/displaytransform/SkibidiGeometry.glsl"

float cuspAt(float hue, float redHue) {
    mat3 primaries = inverse(_SKM_CONES_TO_RGB);
    vec3 edge = hue > redHue ? primaries[1] : primaries[2];
    vec2 ray = _sk_direction(hue);
    float lo = 0.0, hi = 1.0;
    vec3 aab;
    for (int i = 0; i < 20; ++i) {
        float t = (lo + hi) * .5;
        vec3 c = primaries[0] + edge * t;
        aab = _sk_aab(vec3(_sk_response(c.r), _sk_response(c.g), _sk_response(c.b)));
        float crossZ = ray.x * aab.z - ray.y * aab.y;
        if ((hue > redHue) == (crossZ < 0.0))
            lo = t;
        else
            hi = t;
    }
    return pow(aab.x / (3.05 * _SKM_PEAK_R), _SKM_CZ);
}

float displaySeed(ivec2 p) {
    // The endpoint nodes approximate the limits approached from inside the cube.
    float level = clamp(float(p.y) / 31.0, .000001, .9999);
    float y = level * level, r = _sk_response(y);
    vec3 dir = _SKM_OPP_TO_CONES * _sk_direction(float(p.x) * (360.0 / 144.0));
    float upper = _sk_coneRayLimit(dir, vec3(-r), vec3(_SKM_PEAK_R - r));
    float lo = 0.0, hi = upper;
    for (int stepIndex = 1; stepIndex <= 8; ++stepIndex) {
        float x = upper * (float(stepIndex) / 8.0);
        vec3 rgb = _sk_displayRgb(r, y, dir, x);
        if (any(lessThan(rgb, vec3(0))) || any(greaterThan(rgb, vec3(1)))) {
            hi = x;
            break;
        }
        lo = x;
    }
    for (int i = 0; i < 18; ++i) {
        float x = (lo + hi) * .5;
        vec3 rgb = _sk_displayRgb(r, y, dir, x);
        if (all(greaterThanEqual(rgb, vec3(0))) && all(lessThanEqual(rgb, vec3(1))))
            lo = x;
        else
            hi = x;
    }
    return lo / (r * (1.0 - r / _SKM_PEAK_R));
}

void writeBoundary(ivec2 p) {
    if (p.y < 32)
        persistent_drtGeometry_store(p, vec4(displaySeed(p)));
    if (p.y == 32 && p.x <= 120) {
        float red = skRedCusp().x;
        float hue = p.x < 60 ? red * float(p.x) / 60.0 : mix(red, 90.0, float(p.x - 60) / 60.0);
        persistent_drtGeometry_store(p, vec4(cuspAt(hue, red)));
    }
}
vec3 trainingSI5(vec3 ratios, float t) {
    vec3 repaired = max(ratios, vec3(1e-6));
    repaired /= dot(repaired, _SKM_SOURCE_Y);
    float v = t * t;
    float sceneY = .18 * pow((v / (1.0 - v)) * (.82 / .18), .82);
    return clamp(_sk_si5(repaired, pow(_SKM_FL * sceneY, .42)), 0.0, 1.0);
}
void fitControlPoint(uint i) {
    if (i < 4608u)
        weightedLoss[i] = uvec2(0);
    int node = int(i) / 11, face = node / (49 * 49);
    vec3 ratios = vec3(1.0);
    vec2 axis = vec2((node / 49) % 49, node % 49) / 48.0;
    ratios[(face + 1) % 3] = axis.x * axis.x;
    ratios[(face + 2) % 3] = axis.y * axis.y;
    vec3 control = sqrt(trainingSI5(ratios, float(int(i) % 11 + 1) / 12.0));
    coefficients[i] = uvec2(packHalf2x16(control.rg), packHalf2x16(vec2(control.b, 0)));
}
vec3 coefficient(int node, int k) {
    uvec2 bits = coefficients[node * 11 + k];
    return vec3(unpackHalf2x16(bits.x), unpackHalf2x16(bits.y).x);
}
vec3 fittedCurve(vec3 ratios, float t) {
    int face = ratios.r >= ratios.g && ratios.r >= ratios.b ? 0 : int(ratios.b > ratios.g) + 1;
    vec2 uv = sqrt(vec2(ratios[(face + 1) % 3], ratios[(face + 2) % 3])) * 48.0;
    ivec2 xy = min(ivec2(uv), ivec2(47));
    vec2 f = uv - vec2(xy);
    int node = (face * 49 + xy.x) * 49 + xy.y;
    vec3 sum = vec3(pow(t, 12.0));
    float choose = 1.0;
    for (int k = 1; k < 12; ++k) {
        choose *= float(13 - k) / float(k);
        float weight = choose * pow(1.0 - t, float(12 - k)) * pow(t, float(k));
        vec3 a = mix(coefficient(node, k - 1), coefficient(node + 1, k - 1), f.y);
        vec3 b = mix(coefficient(node + 49, k - 1), coefficient(node + 50, k - 1), f.y);
        sum += weight * mix(a, b, f.x);
    }
    return sum;
}
void addLoss(ivec2 cell, float weight, float loss) {
    uint i = uint(cell.y * 96 + cell.x % 96);
    // 227520 samples at up to 16384 units fit in one uint even in the same bin.
    atomicAdd(weightedLoss[i].x, uint(round(16384.0 * weight * loss)));
    atomicAdd(weightedLoss[i].y, uint(round(16384.0 * weight)));
}
void accumulateSample(uint i) {
    const vec3 hex[6] =
        vec3[6](vec3(1, 0, 0), vec3(1, 1, 0), vec3(0, 1, 0), vec3(0, 1, 1), vec3(0, 0, 1), vec3(1, 0, 1));
    int ray = int(i) / 79, h = ray / 15;
    float a = float(h) / 32.0;
    vec3 hue = mix(hex[int(a)], hex[(int(a) + 1) % 6], fract(a));
    float s = mix(.001, 1.0, float(ray % 15) / 14.0);
    vec3 ratios = 1.0 - (1.0 - hue) * (1.0 - s * s);
    float t = (float(i % 79u) + .5) / 79.0;
    vec3 base = trainingSI5(ratios, t);
    vec3 bounded = max(ratios, vec3(1e-15));
    float purity = clamp(1.0 - 9.0 / (dot(bounded, vec3(1)) * dot(1.0 / bounded, vec3(1))), 0.0, 1.0);
    vec3 alt = mix(sqrt(base), fittedCurve(ratios, t), purity);
    alt *= alt;
    float y = dot(base, _SKM_DISPLAY_Y), altY = dot(alt, _SKM_DISPLAY_Y);
    vec3 delta = base - y;
    float norm = dot(delta, delta);
    float loss = clamp(1.0 - dot(alt - altY, delta) * y / max(altY * norm, 1e-35), 0.0, 1.0);
    float low = _sk_div(y - min(base.r, min(base.g, base.b)), y);
    float high = _sk_div(max(base.r, max(base.g, base.b)) - y, 1.0 - y);
    float rho = clamp(max(low, high), 0.0, 1.0);
    float weight = pow(rho, 4.0) * norm / (norm + 1e-12);
    vec2 v = skPlane(delta);
    float angle = norm > 0.0 ? fract(atan(v.y, v.x) / (2.0 * _SK_PI) + 1.0) : 0.0;
    vec2 pos = vec2(angle * 96.0, sqrt(clamp(y, 0.0, 1.0)) * 47.0);
    ivec2 cell = min(ivec2(pos), ivec2(95, 46));
    vec2 f = pos - vec2(cell);
    addLoss(cell, weight * (1.0 - f.x) * (1.0 - f.y), loss);
    addLoss(cell + ivec2(1, 0), weight * f.x * (1.0 - f.y), loss);
    addLoss(cell + ivec2(0, 1), weight * (1.0 - f.x) * f.y, loss);
    addLoss(cell + 1, weight * f.x * f.y, loss);
}
void writeOuterRadius(ivec2 p) {
    const float wx[5] = float[5](1, 4, 6, 4, 1);
    vec2 sum = vec2(0);
    // Two [1,2,1] smoothing passes wrap in hue and clamp after each lightness step.
    for (int dx = -2; dx <= 2; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
            for (int ey = -1; ey <= 1; ++ey) {
                int x = (p.x + dx + 96) % 96, y = clamp(clamp(p.y + dy, 0, 47) + ey, 0, 47);
                float w = wx[dx + 2] * float(2 - abs(dy)) * float(2 - abs(ey));
                sum += w * vec2(weightedLoss[y * 96 + x]);
            }
    persistent_drtGeometry_store(p + ivec2(0, 33), vec4(sum.y > 0.0 ? 1.0 - sum.x / sum.y : 1.0));
}

// Iris can run separate setup programs concurrently on resize. Keep the dependent
// writes and reads in one workgroup so each barrier covers all producers.
layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(1);

void main() {
    uint lane = gl_LocalInvocationIndex;
    for (uint i = lane; i < 145u * 33u; i += gl_WorkGroupSize.x)
        writeBoundary(ivec2(i % 145u, i / 145u));
    groupMemoryBarrier();
    barrier();

    for (uint i = lane; i < 79233u; i += gl_WorkGroupSize.x)
        fitControlPoint(i);
    groupMemoryBarrier();
    barrier();

    for (uint i = lane; i < 227520u; i += gl_WorkGroupSize.x)
        accumulateSample(i);
    groupMemoryBarrier();
    barrier();

    for (uint i = lane; i < 97u * 48u; i += gl_WorkGroupSize.x)
        writeOuterRadius(ivec2(i % 97u, i / 97u));
}
