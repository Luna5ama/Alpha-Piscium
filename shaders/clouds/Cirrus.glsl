#include "Common.glsl"
#include "/util/Rand.glsl"
#include "/util/noise/ValueNoise.glsl"
#include "/util/noise/GradientNoise.glsl"
#include "/util/Sampling.glsl"

#define CIRRUS_CLOUD_HEIGHT 9.0
#define CIRRUS_CLOUD_COVERAGE 0.5

float _clouds_cirrus_coverage(vec3 rayPos) {
    FBMParameters params;

//    params.frequency = 0.02;
//    params.persistence = 0.6;
//    params.lacunarity = 2.5;
//    params.octaveCount = 4u;
//    float fbm = GradientNoise_2D_value_fbm(params, rayPos.xz + vec2(0.0, -24.0));

    params.frequency = 0.02;
    params.persistence = 0.5;
    params.lacunarity = 2.5;
    params.octaveCount = 4u;
    float fbm = ValueNoise_2D_value_fbm(params, rayPos.xz + vec2(72.0, 96.0));

    return pow2(linearStep(0.5 - CIRRUS_CLOUD_COVERAGE * 2.0, 1.0, fbm));
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
    FBMParameters curlParams;
    curlParams.frequency = 0.01;
    curlParams.persistence = 0.6;
    curlParams.lacunarity = 2.0;
    curlParams.octaveCount = 3u;
    vec2 curl = GradientNoise_2D_grad_fbm(curlParams, rayPos.xz + vec2(11.4, 51.4));

    float density = 0.0;
    density += _clouds_cirrus_density_layer((rayPos.xz + 0.114) * 0.04 + curl * 4.0) * 0.125;
    density += _clouds_cirrus_density_layer((rayPos.xz + 0.514) * 0.01 + curl * -0.8) * 0.25;
    density += _clouds_cirrus_density_layer(rayPos.xz * 0.005 + curl * 0.3) * 0.5;
    return density;
}

float clouds_cirrus_density(vec3 rayPos) {
    return _clouds_cirrus_coverage(rayPos) * _clouds_cirrus_density_fbm(rayPos);
}

vec4 clouds_renderCirrus(inout CloudRaymarchParameters params) {
    vec4 result = vec4(0.0, 0.0, 0.0, 1.0);

    return result;
}