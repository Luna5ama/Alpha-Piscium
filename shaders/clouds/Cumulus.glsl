#include "Common.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/util/noise/ValueNoise.glsl"
#include "/util/noise/GradientNoise.glsl"

const float _CU_DENSITY_EPSILON = 0.0001;
const float _CU_COVERAGE_FACTOR = 1.0 - SETTING_CLOUDS_CU_COVERAGE * 2.0;

bool clouds_cu_density(vec3 rayPos, float heightFraction, out float densityOut) {
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
    shapeParams.persistence = 0.8;
    shapeParams.lacunarity = 2.1;
    shapeParams.octaveCount = 3u;
    mat2 rotationMatrix = mat2_rotate(GOLDEN_RATIO);
    float coverage = GradientNoise_2D_value_fbm(shapeParams, rotationMatrix, rayPos.xz + vec2(0.0, -32.0));
    float xzDist = length(rayPos.xz);
    const float DISTANCE_DECAY = 0.005;
    coverage *= exp2(-xzDist * DISTANCE_DECAY);
    coverage = linearStep(_CU_COVERAGE_FACTOR, 1.0, coverage);
    coverage = saturate(coverage * (1.0 + pow2(SETTING_CLOUDS_CU_COVERAGE)));
    coverage = pow2(coverage);

    // https://www.desmos.com/calculator/bdcmyniav9
    const float a0 = -0.0248956145304;
    const float a1 = 9.7248812371;
    const float a2 = -31.1921421103;
    const float a3 = 38.7372454749;
    const float a4 = -17.3174088441;

    float x1 = heightFraction;
    float x2 = heightFraction * heightFraction;
    float x3 = heightFraction * x2;
    float x4 = heightFraction * x3;

    vec4 xs = vec4(x1, x2, x3, x4);

    const vec4 as = vec4(a1, a2, a3, a4);

    float heightCurve = saturate(dot(as, xs) + a0);

    float base = coverage;
    base = saturate(base + heightCurve - 1.0);

    if (base > _CU_DENSITY_EPSILON) {
        FBMParameters curlParams;
        curlParams.frequency = 0.01;
        curlParams.persistence = 0.7;
        curlParams.lacunarity = 3.9;
        curlParams.octaveCount = 2u;
        vec3 curl = GradientNoise_3D_grad_fbm(curlParams, rayPos);

        FBMParameters densityParams;
        densityParams.frequency = 1.7;
        densityParams.persistence = 0.7;
        densityParams.lacunarity = 2.5;
        densityParams.octaveCount = 2u;
        float detail = GradientNoise_3D_value_fbm(densityParams, rayPos + curl * 2.0) * 2.0;

        FBMParameters valueNoiseParams;
        valueNoiseParams.frequency = 6.9;
        valueNoiseParams.persistence = 0.7;
        valueNoiseParams.lacunarity = 2.3;
        valueNoiseParams.octaveCount = 2u;
        detail += ValueNoise_3D_value_fbm(valueNoiseParams, rayPos + curl * 1.0) * 0.5;

        detail = linearStep(-1.0, 1.0, detail);

        densityOut = base;
        densityOut = linearStep(detail * heightFraction * 0.5, 1.0, densityOut);
        densityOut *= 1.0 - heightFraction;

        if (densityOut > _CU_DENSITY_EPSILON) {
            return true;
        }
    }

    return false;
}