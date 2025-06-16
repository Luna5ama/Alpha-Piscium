#include "Common.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/util/noise/ValueNoise.glsl"
#include "/util/noise/GradientNoise.glsl"

float _clouds_cu_coverage(vec3 rayPos) {
    //    FBMParameters earthParams;
    //    earthParams.frequency = 0.008;
    //    earthParams.persistence = 0.8;
    //    earthParams.lacunarity = 3.0;
    //    earthParams.octaveCount = 2u;
    //    float earthCoverage = ValueNoise_2D_value_fbm(earthParams, rayPos.xz + vec2(120.0, -350.0));
    //    earthCoverage = pow2(linearStep(0.0, 0.6, earthCoverage));
    float earthCoverage = 1.0;

    FBMParameters shapeParams;
    shapeParams.frequency = 0.05;
    shapeParams.persistence = 0.6;
    shapeParams.lacunarity = 2.5;
    shapeParams.octaveCount = 2u;
    float shapeCoverage = GradientNoise_2D_value_fbm(shapeParams, mat2_identity(), rayPos.xz + vec2(0.0, -12.0));
    shapeCoverage = pow3(linearStep(0.5 - SETTING_CLOUDS_CU_COVERAGE * 1.5, 1.0, shapeCoverage));

    return earthCoverage * shapeCoverage;
}

float _clouds_cu_density_fbm(vec3 rayPos) {
    return 1.0;
}

float clouds_cu_density(vec3 rayPos) {
    return _clouds_cu_coverage(rayPos) * _clouds_cu_density_fbm(rayPos);
}