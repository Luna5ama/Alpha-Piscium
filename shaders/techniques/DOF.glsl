#include "/util/Rand.glsl"
#include "/util/Math.glsl"

const float FOCAL_LENGTH = SETTING_DOF_FOCAL_LENGTH;
const float F_STOP = SETTING_DOF_F_STOP;
const float APERTURE_DIAMETER = FOCAL_LENGTH / F_STOP;
const float APERTURE_RADIUS = APERTURE_DIAMETER * 0.5;

float _dof_computeCoC(float depth) {
    float numerator = APERTURE_RADIUS * FOCAL_LENGTH * (global_focusDistance - depth);
    float denominator = depth * (global_focusDistance - FOCAL_LENGTH);
    return abs(numerator / denominator);
}

#define HEXGONAL_SPIRAL_COUNT(x) (3 * (x) * ((x) + 1))
#define SAMPLE_COUNT HEXGONAL_SPIRAL_COUNT(SETTING_DOF_QUALITY)
#define DISK_RADIUS SETTING_DOF_MAX_SAMPLE_RADIUS * 2.0
const float INV_SQRT_SAMPLE_COUNT = inversesqrt(float(SAMPLE_COUNT));
const float BASE_WEIGHT = rcp(SAMPLE_COUNT);

#if SETTING_APERTURE_SHAPE == 0 // Circular aperture
float _dof_intersectCoc(vec2 p, float radius) {
    float texelDistSq = dot(p, p);
    return smoothstep(pow2(radius + 0.01), pow2(radius - 0.01), texelDistSq);
}
#elif SETTING_APERTURE_SHAPE == 1 // Hexagonal aperture
float _dof_intersectCoc(vec2 p, float radius) {
    float q = abs(p.x) * (2.0 / radius);
    float r = (abs(p.x) + abs(p.y) * sqrt(3.0)) / radius;
    float s = (abs(p.x) - abs(p.y) * sqrt(3.0)) / radius;

    return smoothstep(2.01, 1.99, max(abs(q), max(abs(r), abs(s))));
}
#endif

vec4 _dof_sampleTap(vec2 centerTexelPos, vec2 sampleTexelPos, float centerCoc, float centerViewZ) {
    vec4 sampleData = transient_dofInput_fetch(sampleTexelPos);
    vec3 sampleColor = sampleData.rgb;
    float sampleViewZ = sampleData.a;
    float sampleCoc = _dof_computeCoC(sampleViewZ);
    sampleViewZ = min(far, sampleViewZ);
    vec2 texelPosDiff = sampleTexelPos - centerTexelPos;
    float weight = BASE_WEIGHT;
    weight *= _dof_intersectCoc(texelPosDiff, sampleCoc);
    weight *= saturate(exp2(-float(SETTING_DOF_MASKING_HEURISTIC) * exp2(-centerCoc) * (sampleViewZ - centerViewZ)));
    return vec4(sampleColor * weight, weight);
}

const mat2 _DOT_HEXGON_ROTATE = mat2(
    0.5, -sqrt(3.0) * 0.5,
    sqrt(3.0) * 0.5, 0.5
);

vec2 _dof_hexagonalSpiral(int index) {
    int layer = int(round(sqrt(float(index / 3.0))));
    int firstIdxInLayer = 3 * layer * (layer - 1) + 1;
    int side = (index - firstIdxInLayer) / layer;
    int idx = (index - firstIdxInLayer) % layer;
    float x = float(layer) * cos(float(side - 1) * PI / 3.0 + PI_HALF) + float(idx + 1) * cos(float(side + 1) * PI / 3.0 + PI_HALF);
    float y = -float(layer) * sin(float(side - 1) * PI / 3.0 + PI_HALF) - float(idx + 1) * sin(float(side + 1) * PI / 3.0 + PI_HALF);
    return _DOT_HEXGON_ROTATE * vec2(x, y);
}

const int SAMPLE_JITTER_GRID_LAYER = 4;
const int SAMPLE_JITTER_GRID = HEXGONAL_SPIRAL_COUNT(SAMPLE_JITTER_GRID_LAYER);

vec3 dof_sample(ivec2 texelPos) {
    vec2 centerTexelPos = vec2(texelPos) + 0.5;

    vec4 centerData = transient_dofInput_fetch(texelPos, 0);
    vec3 centerColor = centerData.rgb;
    float centerViewZ = centerData.a;
    float centerCoc = _dof_computeCoC(centerViewZ);
    vec4 sum = vec4(centerColor * BASE_WEIGHT, BASE_WEIGHT) * exp2(-centerCoc * 0.5);

    #if SETTING_APERTURE_SHAPE == 0 // Circular aperture
    float baseRotation = rand_stbnVec1(texelPos, frameCounter) * PI_2;
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float theta = GOLDEN_ANGLE * i + baseRotation;
        float r = sqrt(i + 0.5) * INV_SQRT_SAMPLE_COUNT;
        vec2 u = r * vec2(cos(theta), sin(theta));
        vec2 offset = u * DISK_RADIUS;

        vec2 sampleTexelPos = centerTexelPos + offset;
        sum += _dof_sampleTap(centerTexelPos, sampleTexelPos, centerCoc, centerViewZ);
    }
    #elif SETTING_APERTURE_SHAPE == 1 // Hexagonal aperture
    float jitterIndex = rand_stbnVec1(texelPos, frameCounter) * SAMPLE_JITTER_GRID;
    vec2 jitter = _dof_hexagonalSpiral(min(int(jitterIndex + 1), SAMPLE_JITTER_GRID)) / (SAMPLE_JITTER_GRID_LAYER + 0.5);

    vec2 centerOffset = jitter / (SETTING_DOF_QUALITY + 0.5) * DISK_RADIUS;
    vec2 centerSampleTexelPos = centerTexelPos + centerOffset;
    sum += _dof_sampleTap(centerTexelPos, centerSampleTexelPos, centerCoc, centerViewZ);

    for (int i = 0; i < SAMPLE_COUNT; i++) {
        vec2 offset = (_dof_hexagonalSpiral(i + 1) + jitter) / (SETTING_DOF_QUALITY + 0.5) * DISK_RADIUS;
        vec2 sampleTexelPos = centerTexelPos + offset;
        sum += _dof_sampleTap(centerTexelPos, sampleTexelPos, centerCoc, centerViewZ);
    }
    #endif

    return sum.rgb / sum.a;
}