#include "Common.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/util/noise/ValueNoise.glsl"
#include "/util/noise/GradientNoise.glsl"

float clouds_cu_coverage(vec3 rayPos) {
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
    vec3 samplePos = rayPos;
    float shapeCoverage = GradientNoise_3D_value_fbm(shapeParams, samplePos);
    shapeCoverage = linearStep(1.0 - SETTING_CLOUDS_CU_COVERAGE * 2.0, 1.0, shapeCoverage);

    return shapeCoverage;
}

float clouds_cu_density(vec3 rayPos) {
    FBMParameters densityParams;
    densityParams.frequency = 2.5;
    densityParams.persistence = 0.6;
    densityParams.lacunarity = 2.6;
    densityParams.octaveCount = 3u;
    float density = GradientNoise_3D_value_fbm(densityParams, rayPos);
    density = linearStep(-1.0, 1.0, density);

    return density;
}