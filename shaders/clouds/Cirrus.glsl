#include "Common.glsl"
#include "/util/Rand.glsl"
#include "/util/noise/ValueNoise.glsl"
#include "/util/noise/GradientNoise.glsl"
#include "/util/Sampling.glsl"

#define CIRRUS_CLOUD_HEIGHT 12.0
#define CIRRUS_CLOUD_COVERAGE 0.5

float _clouds_cirrus_coverage(vec3 rayPos) {
    FBMParameters params;
    params.frequency = 0.02;
    params.persistence = 0.6;
    params.lacunarity = 2.5;
    params.octaveCount = 4u;
//    params.frequency = 0.05;
//    params.persistence = 0.5;
//    params.lacunarity = 3.0;
//    params.octaveCount = 4u;
//    params.frequency = 0.005;
//    params.persistence = 0.6;
//    params.lacunarity = 3.0;
//    params.octaveCount = 32u;
//    float coverage = ValueNoise_2D_value_fbm(params, rayPos.xz + vec2(16.0, 64.0));
    float coverage = GradientNoise_2D_value_fbm(params, rayPos.xz + vec2(16.0, 64.0));
    return pow2(linearStep(0.5 - CIRRUS_CLOUD_COVERAGE * 1.5, 1.0, coverage)) * CIRRUS_CLOUD_COVERAGE * 2.0;
}

float _clouds_cirrus_density_layer(vec2 texCoord) {
    return sampling_textureRepeat(usam_cirrus, texCoord, 1.0).x;

    vec2 dist0 = smoothstep(0.0, 0.5, abs(fract(texCoord - vec2(0.5, 0.5)) - vec2(0.5, 0.5)));
    vec2 dist1 = smoothstep(0.0, 0.5, abs(fract(texCoord - vec2(0.5, 0.5) + vec2(0.5, 0.5)) - vec2(0.5, 0.5)));
    vec2 dist2 = smoothstep(0.0, 0.5, abs(fract(texCoord - vec2(0.5, 0.5) + vec2(1.0, 0.5)) - vec2(0.5, 0.5)));
    vec2 dist3 = smoothstep(0.0, 0.5, abs(fract(texCoord - vec2(0.5, 0.5) + vec2(0.5, 1.0)) - vec2(0.5, 0.5)));

    float density = 0.0;
    density += texture(usam_cirrus, texCoord).r * dist0.x * dist0.y;
    density += texture(usam_cirrus, texCoord + vec2(0.5, 0.5)).r * dist1.x * dist1.y;
    density += texture(usam_cirrus, texCoord + vec2(1.0, 0.5)).r * dist2.x * dist2.y;
    density += texture(usam_cirrus, texCoord + vec2(0.5, 1.0)).r * dist3.x * dist3.y;

    return density;
}

float _clouds_cirrus_density_fbm(vec3 rayPos) {
//    return 1.0;
    float density = 0.0;
    density += _clouds_cirrus_density_layer(rayPos.xz * 0.1) * 0.5;
    density += _clouds_cirrus_density_layer((rayPos.xz + 114.0) * 0.02) * 1.0;
    density += _clouds_cirrus_density_layer((rayPos.xz + 69.0) * 0.004) * 2.0;
    return density;
}

float clouds_cirrus_density(vec3 rayPos) {
    return _clouds_cirrus_coverage(rayPos) * _clouds_cirrus_density_fbm(rayPos);
}

vec4 clouds_renderCirrus(inout CloudRaymarchParameters params) {
    vec4 result = vec4(0.0, 0.0, 0.0, 1.0);

    return result;
}