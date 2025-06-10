#include "Common.glsl"
#include "/util/Rand.glsl"
#include "/util/noise/ValueNoise.glsl"
#include "/util/noise/GradientNoise.glsl"
#include "/util/Sampling.glsl"

#define CIRRUS_CLOUD_HEIGHT 9.0
#define CIRRUS_CLOUD_COVERAGE 0.5

float _clouds_cirrus_coverage(vec3 rayPos) {
    FBMParameters earthParams;
    earthParams.frequency = 0.005;
    earthParams.persistence = 0.8;
    earthParams.lacunarity = 3.0;
    earthParams.octaveCount = 2u;
    float earthCoverage = ValueNoise_2D_value_fbm(earthParams, rayPos.xz + vec2(-140.0, 120.0));
    earthCoverage = pow2(linearStep(0.0, 0.6, earthCoverage));

    FBMParameters shapeParams;
    shapeParams.frequency = 0.05;
    shapeParams.persistence = 0.6;
    shapeParams.lacunarity = 2.5;
    shapeParams.octaveCount = 2u;
    float shapeCoverage = GradientNoise_2D_value_fbm(shapeParams, rayPos.xz + vec2(0.0, -12.0));
    shapeCoverage = pow3(linearStep(0.5 - CIRRUS_CLOUD_COVERAGE * 1.5, 1.0, shapeCoverage));

//    FBMParameters puffyParams;
//    puffyParams.frequency = 0.5;
//    puffyParams.persistence = 0.8;
//    puffyParams.lacunarity = 3.0;
//    puffyParams.octaveCount = 1u;
//    float puffyCoverage = GradientNoise_2D_value_fbm(puffyParams, rayPos.xz);
    float puffyCoverage = GradientNoise_2D_value(rayPos.xz * 0.8);
    puffyCoverage = 1.0 - pow3(1.0 - linearStep(-1.0, 1.0, puffyCoverage + 0.2));

    return earthCoverage * shapeCoverage * 1.0;
}

float _clouds_cirrus_density_layer(vec2 texCoord) {
    return texture(usam_cirrus, texCoord).x;
}

float _clouds_cirrus_density_fbm(vec3 rayPos) {
    FBMParameters curlParams;
    curlParams.frequency = 0.008;
    curlParams.persistence = -0.7;
    curlParams.lacunarity = 1.5;
    curlParams.octaveCount = 3u;
    vec2 curl = GradientNoise_2D_grad_fbm(curlParams, rayPos.xz + vec2(11.4, 51.4));

    float density = 0.0;
    density += _clouds_cirrus_density_layer((rayPos.xz + 0.114) * 0.04 + curl * 2.0) * 0.125;
    density += _clouds_cirrus_density_layer((rayPos.xz + 0.514) * 0.01 + curl * 0.8) * 0.25;
    density += _clouds_cirrus_density_layer(rayPos.xz * 0.005 + curl * 0.4) * 0.5;
    return density * 3.0;
}

float clouds_cirrus_density(vec3 rayPos) {
    return _clouds_cirrus_coverage(rayPos) * _clouds_cirrus_density_fbm(rayPos);
}

vec4 clouds_renderCirrus(inout CloudRaymarchParameters params) {
    vec4 result = vec4(0.0, 0.0, 0.0, 1.0);

    return result;
}