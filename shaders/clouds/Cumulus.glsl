#include "Common.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/util/noise/ValueNoise.glsl"
#include "/util/noise/GradientNoise.glsl"

const float _CU_DENSITY_EPSILON = 0.0001;

bool clouds_cu_density(vec3 rayPos, float heightFraction, out float densityOut) {

    densityOut = 0.0;
    //    FBMParameters earthParams;
    //    earthParams.frequency = 0.008;
    //    earthParams.persistence = 0.8;
    //    earthParams.lacunarity = 3.0;
    //    earthParams.octaveCount = 2u;
    //    float earthCoverage = ValueNoise_2D_value_fbm(earthParams, rayPos.xz + vec2(120.0, -350.0));
    //    earthCoverage = pow2(linearStep(0.0, 0.6, earthCoverage));
    float earthCoverage = 1.0;

    FBMParameters shapeParams;
    shapeParams.frequency = 0.036;
    shapeParams.persistence = -1.7;
    shapeParams.lacunarity = 2.6;
    shapeParams.octaveCount = 4u;
    mat2 rotationMatrix = mat2_rotate(GOLDEN_ANGLE);
    float coverage = GradientNoise_3D_value_fbm(shapeParams, rayPos + vec3(0.0, 12.0, 0.0));
    const float CUMULUS_FACTOR = 0.7;
    const float _CU_COVERAGE_FACTOR = 1.0 - (pow(SETTING_CLOUDS_CU_COVERAGE, 0.3 + CUMULUS_FACTOR * 1.5));
    const float SIGMOID_K = mix(0.2, 4.0, pow2(CUMULUS_FACTOR));
    coverage = rcp(1.0 + exp2(-coverage * SIGMOID_K)); // Sigmoid
    coverage = linearStep(_CU_COVERAGE_FACTOR, 1.0, coverage);
    coverage = pow2(coverage);

    // Make the top small
    // https://www.desmos.com/calculator/2c5574fcdc
    const float a0 = 0.477365819688;
    const float a1 = 5.24645878304;
    const float a2 = -17.5442672791;
    const float a3 = 21.3254889271;
    const float a4 = -9.55952308113;

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
        curlParams.frequency = 0.3;
        curlParams.persistence = -0.8;
        curlParams.lacunarity = 2.6;
        curlParams.octaveCount = 2u;
        vec2 curl2D = GradientNoise_2D_grad_fbm(curlParams, rotationMatrix, rayPos.xz);
        vec3 curl = vec3(curl2D.x, 0.0, curl2D.y);
        curl *= 0.5 + 1.0 * heightFraction;

        densityOut = base;

        // Carve out basic cloud shape
        FBMParameters densityParams;
        densityParams.frequency = 2.4;
        densityParams.persistence = -0.65;
        densityParams.lacunarity = 2.1;
        densityParams.octaveCount = 2u;
        float detail1 = GradientNoise_3D_value_fbm(densityParams, rayPos + curl * 0.4) * 2.0;
        detail1 = smoothstep(-1.0, 1.0, detail1);
        detail1 *= mix(0.5, 0.8, heightFraction);
        detail1 *= pow2(CUMULUS_FACTOR) * 0.95 + 0.05;
        densityOut = linearStep(saturate(detail1), 1.0, densityOut);

        if (densityOut > _CU_DENSITY_EPSILON) {
            // Add some high frequency detail to edges
            FBMParameters valueNoiseParams;
            valueNoiseParams.frequency = 8.2;
            valueNoiseParams.persistence = 0.7;
            valueNoiseParams.lacunarity = 3.1;
            valueNoiseParams.octaveCount = 2u;
            float detail2 = GradientNoise_3D_value_fbm(valueNoiseParams, rayPos + curl * -0.2);
            detail2 = mix(saturate(detail2 * 0.5 + 0.5), abs(detail2), heightFraction * 0.6);
            detail2 *= mix(0.8, 1.0, heightFraction);
            detail2 *= pow2(CUMULUS_FACTOR) * 0.95 + 0.05;
            densityOut = linearStep(saturate(detail2), 1.0, densityOut);

            densityOut *= 1.0 - heightFraction;
            densityOut *= (1.0 - pow2(1.0 - CUMULUS_FACTOR)) * 0.8 + 0.2;

            if (densityOut > _CU_DENSITY_EPSILON) {
                return true;
            }
        }
    }

    return false;
}