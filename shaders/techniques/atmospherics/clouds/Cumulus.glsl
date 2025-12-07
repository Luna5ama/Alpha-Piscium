/*
    References:
        [QUI13a] Quilez, Inigo. "Voronoi - smooth ". 2014.
            MIT License. Copyright (c) 2014 Inigo Quilez.
            https://www.shadertoy.com/view/ldB3zc#

        You can find full license texts in /licenses
*/
#include "Common.glsl"
#include "/util/AxisAngle.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/util/noise/ValueNoise.glsl"
#include "/util/noise/GradientNoise.glsl"

float _clouds_cu_heightCurve1(vec4 xs, float c, float d) {
    // https://www.desmos.com/calculator/2c5574fcdc
    const float a0 = -0.00990250069855;
    const vec4 as = vec4(0.304580242379, -3.16188989373, 3.90881835439, -3.24747329187);
    return exp2(dot(as, xs) + a0);
}

/*
c_{i}
b_{i}
-1.7856344
1.0547554
-167.4822
339.0876
-175.19631
*/
float _clouds_cu_heightCurveWisp(vec4 xs) {
    // https://www.desmos.com/calculator/2c5574fcdc
    const float a0 = -1.7856344;
    const vec4 as = vec4(1.0547554, -167.4822, 339.0876, -175.19631);
    return exp2(dot(as, xs) + a0) + 0.001;
}

/*
c_{i}
-6.4614601
54.648608
-179.31287
217.69984
-89.768473
*/
float _clouds_cu_heightCurveBillowy(vec4 xs) {
    // https://www.desmos.com/calculator/2c5574fcdc
    const float a0 = -6.4614601;
    const vec4 as = vec4(54.648608, -179.31287, 217.69984, -89.768473);
    return exp2(dot(as, xs) + a0);
}

float worleyNoise(vec2 x, uint seed) {
    uvec2 centerCellID = uvec2(ivec2(floor(x)));
    vec2 centerOffset = fract(x);

    // Initialize results
    float f1 = 0.0;
    float f2 = 0.0;
    float m = 1.0;

    for (int ix = -1; ix <= 1; ++ix) {
        for (int iy = -1; iy <= 1; ++iy) {
            ivec2 idOffset = ivec2(ix, iy);
            uvec2 cellID = centerCellID + (idOffset + 2);

            uvec3 hashPos = uvec3(cellID, seed);
            vec3 hashValF = hash_uintToFloat(hash_33_q3(hashPos));
            vec2 cellCenter = hashValF.xy + vec2(idOffset);

            float cellDistance = distance(centerOffset, cellCenter);
            float regularV = pow3(hashValF.z) * smoothstep(0.0, 1.0, 1.0 - cellDistance);
            
            const float w = 0.5;
            float h = hashValF.z * smoothstep(-1.0, 1.0, (m - cellDistance) / w);
            m = mix(m, cellDistance, h) - h * (1.0 - h) * w / (1.0 + 3.0 * w);

            if (f1 < regularV) {
                f2 = f1;
                f1 = regularV;
            } else if (f2 < regularV) {
                f2 = regularV;
            }
        }
    }
    return saturate(1.0 - m - f2);
}

float coverageNoise(vec2 pos) {
    float higherOctave = texture(usam_cumulusBase, pos / 32.0).x;
    float amp = 1.0;
    float freq = 0.35;
    float sum = 0.0;
    for (uint i = 0u; i < 1u; i++) {
        float n = worleyNoise(pos * freq, 0x1919810u + i * 0x114514u);
        sum += n * amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    return sum + higherOctave * 0.75;
}

float detailNoiseB(vec3 pos) {
    pos.y *= 1.2;
    return saturate(0.98 - texture(usam_cumulusDetail1, pos * 0.3).x);
}

float detailNoiseW(vec3 pos) {
    pos.y *= 1.4;
    return texture(usam_cumulusDetail1, pos * 0.5).x;
}

vec3 detailCurlNoise(vec3 pos) {
    return texture(usam_cumulusCurl, pos * 0.1).xyz * 1.0;
}

bool clouds_cu_density(vec3 rayPos, float heightFraction, bool detail, out float densityOut) {
    const float CUMULUS_FACTOR = SETTING_CLOUDS_CU_WEIGHT;

    vec2 baseCoveragePos = rayPos.xz;

    float baseCoverage = coverageNoise(baseCoveragePos);
    const float COVERAGE = SETTING_CLOUDS_CU_COVERAGE;
    float COVERAGE_P = pow2(COVERAGE);
    float COVERAGE_SQRT = sqrt(COVERAGE);

    baseCoverage = max(baseCoverage - (1.0 - COVERAGE) * 1.1, 0.0);
    densityOut = baseCoverage * (1.0 - pow2(1.0 - COVERAGE));

    float x1 = heightFraction;
    float x2 = x1 * x1;
    float x3 = x1 * x2;
    float x4 = x1 * x3;
    vec4 xs = vec4(x1, x2, x3, x4);

    // TODO: expose these as settings
    const float CONE_FACTOR = 0.5;
    const float CONE_TOP_FACTOR = 6.0;
    const float TOP_CURVE_FACTOR = 50.0;
    const float BOTTOM_CURVE_FACTOR = 100.0;

    densityOut *= 1.5 - pow(heightFraction, rcp(CONE_FACTOR));
    densityOut *= exp2(-(heightFraction) * CONE_TOP_FACTOR);
    densityOut *= saturate(1.0 - exp2(TOP_CURVE_FACTOR * (heightFraction - 1.0)));
    densityOut *= saturate(1.0 - exp2(-BOTTOM_CURVE_FACTOR * heightFraction));

    const float CU_BASE_DENSITY_THRESHOLD = 0.02;
//    return densityOut > CU_BASE_DENSITY_THRESHOLD;

    if (densityOut > CU_BASE_DENSITY_THRESHOLD) {
        #if !defined(SETTING_SCREENSHOT_MODE) && defined(SETTING_CLOUDS_CU_WIND)
        rayPos += uval_cuDetailWind;
        #endif

        float bottomDetail = 1.0;

        vec3 curlPos = rayPos;
        curlPos.y *= 1.3;
        vec3 detailCurl = detailCurlNoise(curlPos);
        detailCurl *= 0.1 + 0.1 * pow2(heightFraction);

        float detail1Billowy = detailNoiseB(rayPos + detailCurl * 0.6);

        detail1Billowy = pow2(detail1Billowy);
        bottomDetail = 1.0 - detail1Billowy * smoothstep(0.1, 0.0, heightFraction) * 2.0;
        float hc3 = _clouds_cu_heightCurveBillowy(xs);
        detail1Billowy *= hc3;
        detail1Billowy *= COVERAGE_SQRT;
        densityOut = linearStep(saturate(detail1Billowy), 1.0, densityOut);

        float detail1Wisp = detailNoiseW(rayPos + detailCurl * 2.3);

        detail1Wisp = pow2(detail1Wisp);
        detail1Wisp *= COVERAGE_SQRT;
        float hc2 = _clouds_cu_heightCurveWisp(xs);
        detail1Wisp *= hc2;

        densityOut = linearStep(saturate(detail1Wisp), 1.0, densityOut);

        // TODO: add another detail layer

        float hardEdgeBlend = smoothstep(0.0, 0.3, heightFraction);
        float minDetailDensity = mix(0.001, 0.02, hardEdgeBlend);
        float edgeDesnityRange = mix(0.08, 0.0015, hardEdgeBlend);
        densityOut *= smoothstep(minDetailDensity, minDetailDensity + edgeDesnityRange, densityOut);

        // Make cloud top more dense
        densityOut *= 1.0 + heightFraction * 10.0;
        // Add some shapes to the bottom, using multipcation because subtraction only affect sides
        densityOut *= bottomDetail;

        if (densityOut > 0.0) {
            return true;
        }
    }

    densityOut = 0.0;
    return false;
}