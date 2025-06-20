#include "Common.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/util/noise/ValueNoise.glsl"
#include "/util/noise/GradientNoise.glsl"

float clouds_cu_coverage(vec3 rayPos, float heightFraction) {
    //    FBMParameters earthParams;
    //    earthParams.frequency = 0.008;
    //    earthParams.persistence = 0.8;
    //    earthParams.lacunarity = 3.0;
    //    earthParams.octaveCount = 2u;
    //    float earthCoverage = ValueNoise_2D_value_fbm(earthParams, rayPos.xz + vec2(120.0, -350.0));
    //    earthCoverage = pow2(linearStep(0.0, 0.6, earthCoverage));
    float earthCoverage = 1.0;

    FBMParameters shapeParams;
    shapeParams.frequency = 0.1;
    shapeParams.persistence = 0.5;
    shapeParams.lacunarity = 2.0;
    shapeParams.octaveCount = 4u;
    mat2 rotationMatrix = mat2_rotate(GOLDEN_RATIO);
    float baseCoverage = GradientNoise_2D_value_fbm(shapeParams, rotationMatrix, rayPos.xz + vec2(110.0, 0.0));
    baseCoverage = linearStep(1.0 - SETTING_CLOUDS_CU_COVERAGE * 2.0, 1.0, baseCoverage);

    // https://www.desmos.com/calculator/5ttnmabiq2
    const float a0 = -0.0716611475481;
    const float a1 = 12.5479827742;
    const float a2 = -55.4236072975;
    const float a3 = 105.501792431;
    const float a4 = -92.0798828757;
    const float a5 = 29.5155575344;

    float xzDist = length(rayPos.xz);
    float x0 = 1.0;
    float x1 = linearStep(saturate(0.0 + xzDist * 0.0004), 1.0, heightFraction);
    float x2 = heightFraction * heightFraction;
    float x3 = heightFraction * x2;
    float x4 = heightFraction * x3;
    float x5 = heightFraction * x4;

    vec2 xa = vec2(x0, x1);
    vec4 xb = vec4(x2, x3, x4, x5);

    const vec2 aa = vec2(a0, a1);
    const vec4 ab = vec4(a2, a3, a4, a5);

    float heightCurve = saturate(dot(aa, xa) + dot(ab, xb));

    float coverage = baseCoverage;
    coverage = saturate(coverage + heightCurve - 1.0);

    return coverage;
}

float clouds_cu_density(vec3 rayPos) {
    FBMParameters curlParams;
    curlParams.frequency = 0.01;
    curlParams.persistence = 0.7;
    curlParams.lacunarity = 3.9;
    curlParams.octaveCount = 2u;
    vec3 curl = GradientNoise_3D_grad_fbm(curlParams, rayPos);

    FBMParameters densityParams;
    densityParams.frequency = 2.5;
    densityParams.persistence = 0.6;
    densityParams.lacunarity = 2.0;
    densityParams.octaveCount = 2u;
    float density = GradientNoise_3D_value_fbm(densityParams, rayPos + curl * 2.0) * 1.0;

    FBMParameters valueNoiseParams;
    valueNoiseParams.frequency = 6.9;
    valueNoiseParams.persistence = 0.7;
    valueNoiseParams.lacunarity = 3.1;
    valueNoiseParams.octaveCount = 3u;
    density += ValueNoise_3D_value_fbm(valueNoiseParams, rayPos + curl * 1.0) * 0.25;

    density = linearStep(-1.0, 1.0, density);

    return density;
}