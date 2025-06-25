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
    shapeParams.frequency = 0.015;
    shapeParams.persistence = -1.8;
    shapeParams.lacunarity = 2.6;
    shapeParams.octaveCount = 4u;
    mat2 rotationMatrix = mat2_rotate(GOLDEN_ANGLE);
        float coverage = GradientNoise_2D_value_fbm(shapeParams, rotationMatrix, rayPos.xz + vec2(3.0, -6.0));
    float xzDist = length(rayPos.xz);
    const float DISTANCE_DECAY = 0.002;
    const float CUMULUS_FACTOR = 0.8;
    const float SIGMOID_K = mix(0.2, 2.0, pow2(CUMULUS_FACTOR));
    coverage = rcp(1.0 + exp2(-coverage * SIGMOID_K)); // Sigmoid
    coverage = linearStep(_CU_COVERAGE_FACTOR, 1.0, coverage);
    coverage *= exp2(-xzDist * DISTANCE_DECAY);
    coverage = pow2(coverage);

    // https://www.desmos.com/calculator/2c5574fcdc
    const float a0 = 0.379724148315;
    const float a1 = 5.83589350096;
    const float a2 = -18.9521206139;
    const float a3 = 24.4587627067;
    const float a4 = -11.7641185037;

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
        curlParams.frequency = 0.04;
        curlParams.persistence = 0.7;
        curlParams.lacunarity = 2.6;
        curlParams.octaveCount = 2u;
        vec3 curSamplePos = rayPos;
        curSamplePos.y *= 0.8;
        vec3 curl = GradientNoise_3D_grad_fbm(curlParams, curSamplePos);
        curl *= 0.8 + heightFraction * 0.5;

        densityOut = base;

        FBMParameters densityParams;
        densityParams.frequency = 0.6;
        densityParams.persistence = -0.65;
        densityParams.lacunarity = 2.7;
        densityParams.octaveCount = 2u;
        float detail1 = GradientNoise_3D_value_fbm(densityParams, rayPos + curl * 1.2) * 1.5;
        detail1 = linearStep(-1.0, 1.0, detail1);
        detail1 *= mix(0.7, 1.8, pow2(heightFraction));
        detail1 *= mix(0.1, 1.0, CUMULUS_FACTOR);
        densityOut = linearStep(saturate(detail1), 1.0, densityOut);

        if (densityOut > _CU_DENSITY_EPSILON) {
            FBMParameters valueNoiseParams;
            valueNoiseParams.frequency = 5.2;
            valueNoiseParams.persistence = 0.7;
            valueNoiseParams.lacunarity = 2.6;
            valueNoiseParams.octaveCount = 2u;
            float detail2 = GradientNoise_3D_value_fbm(valueNoiseParams, rayPos + curl * 0.6);
            detail2 = mix(saturate(linearStep(-1.0, 1.0, detail2) - 0.3), abs(detail2), heightFraction * 0.8);
            detail2 *= mix(0.4, 0.6, heightFraction);
            detail2 *= mix(0.1, 1.0, CUMULUS_FACTOR);
            densityOut = linearStep(saturate(detail2), 1.0, densityOut);

            densityOut *= pow3(1.0 - heightFraction);

            if (densityOut > _CU_DENSITY_EPSILON) {
                return true;
            }
        }
    }

    return false;
}