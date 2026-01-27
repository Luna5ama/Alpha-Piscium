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

const float _LOW_BASE_FREQ = exp2(SETTING_CLOUDS_LOW_BASE_FREQ);
const float _LOW_CURL_FREQ = exp2(SETTING_CLOUDS_LOW_CURL_FREQ);
const float _LOW_BILLOWY_FREQ = exp2(SETTING_CLOUDS_LOW_BILLOWY_FREQ - 1.0);
const float _LOW_BILLOWY_CURL_STR = exp2(SETTING_CLOUDS_LOW_BILLOWY_CURL_STR - 1.0);
const float _HIGH_BILLOWY_FREQ = exp2(SETTING_CLOUDS_HIGH_BILLOWY_FREQ);
const float _HIGH_BILLOWY_CURL_STR = exp2(SETTING_CLOUDS_HIGH_BILLOWY_CURL_STR);
const float _LOW_WISPS_FREQ = exp2(SETTING_CLOUDS_LOW_WISPS_FREQ);
const float _LOW_WISPS_CURL_STR = exp2(SETTING_CLOUDS_LOW_WISPS_CURL_STR);


/*
b_{i}
-1.755622
3.6801126
-163.09651
320.59657
-163.29147
0.01273534
*/
float _clouds_cu_heightCurveWisp(vec4 xs) {
    // https://www.desmos.com/calculator/2c5574fcdc
    const float a0 = -1.755622;;
    const vec4 as = vec4(3.6801126, -163.09651, 320.59657, -163.29147);
    const float a5 = 0.01273534;
    return exp2(dot(as, xs) + a0) + a5;
}

/*
c_{i}
-2.8940248
20.938597
-83.765418
114.27168
-53.99707
*/
float _clouds_cu_heightCurveBillowy(vec4 xs) {
    // https://www.desmos.com/calculator/2c5574fcdc
    const float a0 = -2.8940248;
    const vec4 as = vec4(20.938597, -83.765418, 114.27168, -53.99707);
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
            float v = pow2(hashValF.z) * saturate(1.0 - cellDistance);

            const float w = 0.35;
            const float wRcp = 1.4285714286; // 1.0 / w * 0.5
            float h = hashValF.z * saturate((m - cellDistance) * wRcp + 0.5);
            float wh = w * h;
            m = mix(m, cellDistance, h) - fma(wh, -h, wh) / (1.0 + 3.0 * w);

            if (f1 < v) {
                f2 = f1;
                f1 = v;
            } else if (f2 < v) {
                f2 = v;
            }
        }
    }

    return saturate(1.0 - m - f2);
}

float coverageNoise(vec2 pos) {
    pos *= _LOW_BASE_FREQ;
    float higherOctave = texture(usam_cumulusBase, pos / 32.0).x;
    const float freq = 0.35;
    float baseNoise = worleyNoise(pos * freq, 0x1919810u);
    return baseNoise + higherOctave * 0.75;
}

float detailNoiseB(vec3 pos, vec3 curl) {
    vec3 lowFreqPos = pos + curl * _LOW_BILLOWY_CURL_STR;
    lowFreqPos *= _LOW_BILLOWY_FREQ;
    float lowFreq = texture(usam_cumulusDetail1, lowFreqPos).x;
    vec3 highFreqPos = pos + curl * _HIGH_BILLOWY_CURL_STR;
    highFreqPos *= _HIGH_BILLOWY_FREQ;
    float highFreq = texture(usam_cumulusDetail2, highFreqPos).x;
    return pow3(1.0 - lowFreq) * 0.6 + pow2(0.5 - highFreq) * 0.5;
}

float detailNoiseW(vec3 pos) {
    pos *= 0.5;
    pos *= _LOW_WISPS_FREQ;
    return texture(usam_cumulusDetail2, pos).x;
}

vec3 detailCurlNoise(vec3 pos) {
    pos *= 0.5;
    pos *= _LOW_CURL_FREQ;
    return texture(usam_cumulusCurl, pos).xyz;
}

bool clouds_cu_density(vec3 rayPos, float heightFraction, bool detail, out float densityOut, out float densityLodOut) {
    vec2 baseCoveragePos = rayPos.xz;

    float baseCoverage = coverageNoise(baseCoveragePos);
    const float COVERAGE = SETTING_CLOUDS_CU_COVERAGE;
    float COVERAGE_P = pow2(COVERAGE);
    float COVERAGE_SQRT = sqrt(COVERAGE);

    baseCoverage = max(baseCoverage - (1.0 - COVERAGE_P) * 0.8, 0.0);
    densityOut = baseCoverage * (1.0 - pow2(1.0 - COVERAGE));

    float x1 = heightFraction;
    float x2 = x1 * x1;
    float x3 = x1 * x2;
    float x4 = x1 * x3;
    vec4 xs = vec4(x1, x2, x3, x4);

    // TODO: expose these as settings
    const float CONE_FACTOR = mix(5.0, 0.1, SETTING_CLOUDS_LOW_CONE_FACTOR);
    const float CONE_TOP_FACTOR = mix(4.0, 10.0, SETTING_CLOUDS_LOW_CONE_FACTOR);
    const float TOP_CURVE_FACTOR = float(SETTING_CLOUDS_LOW_TOP_CURVE_FACTOR);
    const float BOTTOM_CURVE_FACTOR = float(SETTING_CLOUDS_LOW_BOTTOM_CURVE_FACTOR);

    densityOut *= 1.5 - pow(heightFraction, CONE_FACTOR);
    densityOut *= exp2(-(heightFraction) * CONE_TOP_FACTOR);
    densityOut *= saturate(1.0 - exp2(TOP_CURVE_FACTOR * (heightFraction - 1.0)));
    densityOut *= saturate(1.0 - exp2(-BOTTOM_CURVE_FACTOR * heightFraction));
    densityLodOut = densityOut;
//    densityLodOut *= 1.0 + heightFraction * 16.0;

    const float CU_BASE_DENSITY_THRESHOLD = 0.02;
//    return densityOut > CU_BASE_DENSITY_THRESHOLD;

    if (densityOut > CU_BASE_DENSITY_THRESHOLD) {
        #if !defined(SETTING_SCREENSHOT_MODE) && defined(SETTING_CLOUDS_CU_WIND)
        rayPos += uval_cuDetailWind;
        #endif

        #define DETAIL_NOISE 1

        vec3 curlPos = rayPos;
        curlPos.y *= 1.3;
        vec3 detailCurl = detailCurlNoise(curlPos);
        detailCurl *= 0.2 + 0.3 * pow2(heightFraction);
        detailCurl *= linearStep(mix(CU_BASE_DENSITY_THRESHOLD, 1.0, pow5(1.0 - heightFraction)), 0.0, densityOut);

        float detail1Billowy = detailNoiseB(rayPos, detailCurl);

        float bottomDetail = 1.0 - detail1Billowy * smoothstep(0.1, 0.0, heightFraction) * 2.0;
        float hc3 = _clouds_cu_heightCurveBillowy(xs);
        detail1Billowy *= hc3;
        detail1Billowy *= COVERAGE_SQRT;
        #if DETAIL_NOISE
        densityOut = linearStep(saturate(detail1Billowy), 1.0, densityOut);
        #endif

        float detail1Wisp = detailNoiseW(rayPos + detailCurl * 2.0 * _LOW_WISPS_CURL_STR);

        detail1Wisp = pow2(detail1Wisp);
        detail1Wisp *= COVERAGE_SQRT;
        float hc2 = _clouds_cu_heightCurveWisp(xs);
        detail1Wisp *= hc2;

        #if DETAIL_NOISE
        densityOut = linearStep(saturate(detail1Wisp), 1.0, densityOut);
        #endif

        float hardEdgeBlend = linearStep(0.0, 0.3, heightFraction);
        float minDetailDensity = mix(0.001, 0.02, hardEdgeBlend);
        float edgeDesnityRange = mix(0.08, 0.0015, hardEdgeBlend);
        densityOut *= smoothstep(minDetailDensity, minDetailDensity + edgeDesnityRange, densityOut);

        // Make cloud top more dense
        // TODO: Expose height density modifier as setting
        densityOut *= 1.0 + heightFraction * 16.0;
        // Add some shapes to the bottom, using multipcation because subtraction only affect sides
        densityOut *= bottomDetail;

        if (densityOut > 0.0) {
            return true;
        }
    }

    densityOut = 0.0;
    return false;
}