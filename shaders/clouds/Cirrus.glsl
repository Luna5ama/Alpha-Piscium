#include "Common.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/util/noise/ValueNoise.glsl"
#include "/util/noise/GradientNoise.glsl"
#include "/util/Sampling.glsl"

float _clouds_cirrus_density_layer(vec2 texCoord) {
    return texture(usam_cirrus, texCoord).x;
}

float clouds_cirrus_density(vec3 rayPos) {
    FBMParameters curlParams;
    curlParams.frequency = 0.01;
    curlParams.persistence = 0.8;
    curlParams.lacunarity = 2.0;
    curlParams.octaveCount = 3u;
    mat2 rotatioMatrix = mat2_rotate(GOLDEN_ANGLE);
    vec2 curl = GradientNoise_2D_grad_fbm(curlParams, rotatioMatrix, rayPos.xz);

    FBMParameters shapeParams;
    shapeParams.frequency = 0.04;
    shapeParams.persistence = 0.6;
    shapeParams.lacunarity = 1.9;
    shapeParams.octaveCount = 3u;
    mat2 shapeRotMat = mat2_rotate(PI_QUARTER);
    float coverage = GradientNoise_2D_value_fbm(shapeParams, shapeRotMat, rayPos.xz + vec2(-8.0, -4.0) + curl * 16.0);
    coverage = pow3(linearStep(0.5 - SETTING_CLOUDS_CI_COVERAGE * 1.5, 1.0, coverage));


    float density = 0.0;
    density += _clouds_cirrus_density_layer((rayPos.xz + 0.114) * 0.12 + curl * 2.5) * 0.125;
    density += _clouds_cirrus_density_layer((rayPos.xz + 0.514) * 0.08 + curl * 0.8) * 0.25;
    density += pow2(saturate(_clouds_cirrus_density_layer(rayPos.xz * 0.02 + curl * 0.4) * 2.0)) * 0.5;
    density *= 2.0 * SETTING_CLOUDS_CI_DENSITY;

    return coverage * density;
}