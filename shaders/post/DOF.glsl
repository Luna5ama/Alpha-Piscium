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

#define SAMPLE_COUNT 8
#define DISK_RADIUS SETTING_DOF_MAX_SAMPLE_RADIUS
const float INV_SQRT_SAMPLE_COUNT = inversesqrt(float(SAMPLE_COUNT));
const float BASE_WEIGHT = rcp(SAMPLE_COUNT);

vec4 _dof_sampleTap(sampler2D inputTex, vec2 centerTexelPos, vec2 sampleTexelPos, float centerCoc, float centerViewZ) {
    vec2 sampleScreenPos = sampleTexelPos * global_mainImageSizeRcp;
    vec4 sampleData = texture(inputTex, sampleScreenPos);
    vec3 sampleColor = sampleData.rgb;
    float sampleViewZ = sampleData.a;
    float sampleCoc = _dof_computeCoC(sampleViewZ);
    vec2 texelPosDiff = sampleTexelPos - centerTexelPos;
    float texelDistSq = dot(texelPosDiff, texelPosDiff);
    float weight = BASE_WEIGHT;
    weight *= smoothstep(pow2(sampleCoc + 0.1), pow2(sampleCoc - 0.1), texelDistSq);
    weight *= saturate(exp2(-float(SETTING_DOF_MASKING_HEURISTIC) * exp2(-centerCoc) * (sampleViewZ - centerViewZ)));
    return vec4(sampleColor * weight, weight);
}

vec3 dof_sample(sampler2D inputTex, ivec2 texelPos) {
    vec2 centerTexelPos = vec2(texelPos) + 0.5;
    float baseRotation = rand_stbnVec1(texelPos, frameCounter) * PI_2;

    vec4 centerData = texelFetch(inputTex, texelPos, 0);
    vec3 centerColor = centerData.rgb;
    float centerViewZ = centerData.a;
    float centerCoc = _dof_computeCoC(centerViewZ);

    vec4 sum = vec4(centerColor * BASE_WEIGHT, BASE_WEIGHT);

    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float theta = GOLDEN_ANGLE * i + baseRotation;
        float r = sqrt(i + 0.5) * INV_SQRT_SAMPLE_COUNT;
        vec2 u = r * vec2(cos(theta), sin(theta));
        vec2 offset = u * DISK_RADIUS;

        vec2 sampleTexelPos = centerTexelPos + offset;
        sum += _dof_sampleTap(inputTex, centerTexelPos, sampleTexelPos, centerCoc, centerViewZ);
    }

    return sum.rgb / sum.a;
}