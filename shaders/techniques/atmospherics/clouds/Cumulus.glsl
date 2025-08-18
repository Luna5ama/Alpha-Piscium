#include "Common.glsl"
#include "/util/AxisAngle.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/util/noise/ValueNoise.glsl"
#include "/util/noise/GradientNoise.glsl"

const float _CU_DENSITY_EPSILON = 0.0001;

float _clouds_cu_heightCurve1(vec4 xs) {
    // https://www.desmos.com/calculator/2c5574fcdc
    const float a0 = 0.477365819688;
//    const float a1 = 5.24645878304;
//    const float a2 = -17.5442672791;
//    const float a3 = 21.3254889271;
//    const float a4 = -9.55952308113;

    const vec4 as = vec4(5.24645878304, -17.5442672791, 21.3254889271, -9.55952308113);
    return saturate(dot(as, xs) + a0);
}


float _clouds_cu_heightCurve2(vec4 xs) {
    // https://www.desmos.com/calculator/2c5574fcdc
    const float a0 = 0.615;
//    const float a1 = -2.4;
//    const float a2 = 2.9;
    const vec2 as = vec2(-3.2, 2.8);
    return saturate(dot(as, xs.xy) + a0);
}

bool clouds_cu_density(vec3 rayPos, float heightFraction, bool detail, out float densityOut) {
    mat2 rotation2D = mat2_rotate(GOLDEN_ANGLE);
    AxisAngle rotation3D = AxisAngle_init(normalize(vec3(1.0, -1.0, 1.0)), GOLDEN_ANGLE);
    const float CUMULUS_FACTOR = SETTING_CLOUDS_CU_WEIGHT;

    FBMParameters baseCoverageParams;
    baseCoverageParams.frequency = 0.04;
    baseCoverageParams.persistence = 0.8;
    baseCoverageParams.lacunarity = 2.5;
    baseCoverageParams.octaveCount = 2u;
    vec2 baseCoveragePos = rayPos.xz;
    baseCoveragePos += vec2(-10.0, 12.0);
    float baseCoverage = GradientNoise_2D_value_fbm(baseCoverageParams, rotation2D, baseCoveragePos);
    const float BASE_COVERAGE_BIAS = -SETTING_CLOUDS_CI_COVERAGE;
    const float BASE_COVERAGE_RANGE = 0.8;
    baseCoverage = linearStep(BASE_COVERAGE_BIAS - BASE_COVERAGE_RANGE, BASE_COVERAGE_BIAS + BASE_COVERAGE_RANGE, baseCoverage);
    baseCoverage = pow4(baseCoverage);
    densityOut = baseCoverage;

    FBMParameters coverageParams;
    coverageParams.frequency = 0.5;
    coverageParams.persistence = -0.5;
    coverageParams.lacunarity = -1.1;
    coverageParams.octaveCount = 4u;
    vec3 coveragePos = rayPos;
    coveragePos.y *= 0.5;
    float coverage = GradientNoise_3D_value_fbm(coverageParams, rotation3D, coveragePos) * 2.0;
    const float _CU_COVERAGE_FACTOR = 1.0 - SETTING_CLOUDS_CU_COVERAGE - (1.0 - CUMULUS_FACTOR) * 0.3;
    const float SIGMOID_K = mix(0.5, 8.0, pow2(CUMULUS_FACTOR));
    coverage = rcp(1.0 + exp2(-coverage * SIGMOID_K)); // Sigmoid
    coverage = linearStep(_CU_COVERAGE_FACTOR, 1.0, coverage);

    float x1 = heightFraction;
    float x2 = heightFraction * heightFraction;
    float x3 = heightFraction * x2;
    float x4 = heightFraction * x3;
    vec4 xs = vec4(x1, x2, x3, x4);
    float heightCurve = _clouds_cu_heightCurve1(xs);

    densityOut *= coverage;
    densityOut = saturate(densityOut + heightCurve - 1.0);

    if (densityOut > _CU_DENSITY_EPSILON) {
        #if !defined(SETTING_SCREENSHOT_MODE) && defined(SETTING_CLOUDS_CU_WIND)
        rayPos += uval_cuDetailWind;
        #endif

        FBMParameters curlParams;
        curlParams.frequency = 0.2;
        curlParams.persistence = -0.8;
        curlParams.lacunarity = 1.2;
        curlParams.octaveCount = 2u;
        vec3 curl = GradientNoise_3D_grad_fbm(curlParams, rotation3D, rayPos);
        curl *= 0.5 + 0.5 * heightFraction;

        // Carve out basic cloud shape
        FBMParameters densityParams;
        densityParams.frequency = 1.6;
        densityParams.persistence = -0.65;
        densityParams.lacunarity = 2.8;
        densityParams.octaveCount = 2u;
        float detail1 = GradientNoise_3D_value_fbm(densityParams, rotation3D, rayPos + curl * 0.3);
        float detail1SigmoidK = mix(12.0, 32.0, _clouds_cu_heightCurve2(xs));
        detail1SigmoidK *= CUMULUS_FACTOR * 0.9 + 0.1;
        detail1 = rcp(1.0 + exp2(-detail1 * detail1SigmoidK)); // Controls carve out hardness
        detail1 *= mix(0.2, 0.7, heightFraction); // Controls how much to carve out
        densityOut = linearStep(saturate(detail1), 1.0, densityOut);

        if (densityOut > _CU_DENSITY_EPSILON) {
            if (detail) {
                // Add some high frequency detail to edges
                FBMParameters valueNoiseParams;
                valueNoiseParams.frequency = 4.9;
                valueNoiseParams.persistence = 0.7;
                valueNoiseParams.lacunarity = 2.9;
                valueNoiseParams.octaveCount = 2u;
                float detail2 = GradientNoise_3D_value_fbm(valueNoiseParams, rotation3D, rayPos + curl * -0.2) * 2.0;
                detail2 = mix(saturate(detail2 * 0.5 + 0.5), abs(detail2), heightFraction * 0.6);
                detail2 = pow2(detail2);
                detail2 *= mix(0.3, 0.6, heightFraction);
                detail2 *= CUMULUS_FACTOR * 0.9 + 0.1;
                densityOut = linearStep(saturate(detail2), 1.0, densityOut);
            }

            densityOut *= 1.0 - pow2(heightFraction);
            densityOut *= (1.0 - pow2(1.0 - CUMULUS_FACTOR)) * 0.5 + 0.5;

            if (densityOut > _CU_DENSITY_EPSILON) {
                return true;
            }
        }
    }

    densityOut = 0.0;
    return false;
}