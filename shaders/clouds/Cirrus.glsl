#include "Common.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/util/noise/ValueNoise.glsl"
#include "/util/noise/GradientNoise.glsl"
#include "/util/Sampling.glsl"

float _clouds_ci_density_layer(vec2 texCoord) {
    return texture(usam_cirrus, texCoord).x;
}

float clouds_ci_density(vec3 rayPos) {
    FBMParameters curlParams;
    curlParams.frequency = 0.01;
    curlParams.persistence = 0.8;
    curlParams.lacunarity = 2.0;
    curlParams.octaveCount = 3u;
    mat2 rotatioMatrix = mat2_rotate(GOLDEN_ANGLE);
    vec2 curl = GradientNoise_2D_grad_fbm(curlParams, rotatioMatrix, rayPos.xz);

    FBMParameters shapeParams;
    shapeParams.frequency = 0.05;
    shapeParams.persistence = 0.6;
    shapeParams.lacunarity = 2.1;
    shapeParams.octaveCount = 3u;
    mat2 shapeRotMat = mat2_rotate(PI_QUARTER);
    float coverage = GradientNoise_2D_value_fbm(shapeParams, shapeRotMat, rayPos.xz + curl * 15.0);
    coverage = pow3(linearStep(0.5 - SETTING_CLOUDS_CI_COVERAGE * 1.5, 1.0, coverage));

    float density = 0.0;
    density += _clouds_ci_density_layer(rayPos.xz * 0.2 + curl * 4.3) * 0.1;
    density += _clouds_ci_density_layer(rayPos.xz * 0.08 + curl * 2.1) * 0.1;
    density += _clouds_ci_density_layer(rayPos.xz * 0.01 + curl * 0.2) * 0.6;
    density *= 0.25 * SETTING_CLOUDS_CI_DENSITY;

    return coverage * density;
}