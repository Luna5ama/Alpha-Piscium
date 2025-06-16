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
    shapeParams.frequency = 0.2;
    shapeParams.persistence = 0.5;
    shapeParams.lacunarity = 2.0;
    shapeParams.octaveCount = 4u;
    mat2 rotationMatrix = mat2_rotate(GOLDEN_RATIO);
    float shapeCoverage = GradientNoise_2D_value_fbm(shapeParams, rotationMatrix, rayPos.xz);
    shapeCoverage = linearStep(1.0 - SETTING_CLOUDS_CU_COVERAGE * 2.0, 1.0, shapeCoverage);

    return shapeCoverage;
}

float _clouds_cu_density_fbm(vec3 rayPos) {
    return SETTING_CLOUDS_CU_DENSITY;
}

float clouds_cu_density(vec3 rayPos) {
    return _clouds_cu_coverage(rayPos) * _clouds_cu_density_fbm(rayPos);
}