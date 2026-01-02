#ifndef INCLUDE_util_Sampling_glsl
#define INCLUDE_util_Sampling_glsl a
/*
    References:
        [QUI15] Quilez, Inigo. "Texture Repetition". 2015.
            https://iquilezles.org/articles/texturerepetition/
        [QUI17] Quilez, Inigo. "Texture Repetition V". 2017.
            MIT License. Copyright (c) 2017 Inigo Quilez
            https://www.shadertoy.com/view/Xtl3zf
        [DJO12] Djonov, Phill. "Bicubic Filtering in Fewer Taps". 2012.
            https://vec3.net/posts/bicubic-filtering-in-fewer-taps
        [MJP19] MJP. "An HLSL function for sampling a 2D texture with Catmull-Rom filtering, using 9 texture samples instead of 16". 2019.
            https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1

        You can find full license texts in /licenses
*/

#include "Hash.glsl"
#include "Math.glsl"

vec4 sampling_bSplineWeights(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return vec4(
        (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0,
        (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
        (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0,
        t3 / 6.0
    );
}

vec4 sampling_catmullRomWeights(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return vec4(
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2
    );
}

vec4 sampling_mitchellNetravaliWeights(float t, float B, float C) {
    float t2 = t * t;
    float t3 = t2 * t;

    vec4 c1 = vec4(B / -6.0 - C, 0.5 * B + 2.0 * C, -0.5 * B - C, B / 6.0);
    vec3 c2 = vec3(-1.5 * B - C + 2.0, 2.0 * B + C - 3.0, B / -3.0 + 1.0);
    vec4 c3 = vec4(1.5 * B + C - 2.0, -2.5 * B - 2.0 * C + 3.0, 0.5 * B + C, B / 6.0);
    vec2 c4 = vec2(B / 6.0 + C, -C);

    vec4 catmull = sampling_catmullRomWeights(t);

    return vec4(
        dot(c1, vec4(t3, t2, t, 1.0)),
        dot(c2, vec3(t3, t2, 1.0)),
        dot(c3, vec4(t3, t2, t, 1.0)),
        dot(c4, vec2(t3, t2))
    );
}

vec4 sampling_gaussianWeights(float t, float sigma) {
    vec4 x = vec4(t) + vec4(1.0, 0.0, -1.0, -2.0);
    return exp(-pow2(x) / (2.0 * pow2(sigma)));
}

vec4 _sampling_sincn(vec4 x) {
    return mix(vec4(1.0), sin(PI * x) / (PI * x), greaterThan(abs(x), vec4(0.00001)));
}

vec4 _sampling_lanczoc2(vec4 x) {
    x = clamp(x, -2.0, 2.0);
    return _sampling_sincn(x) * _sampling_sincn(0.5 * x);
}

vec4 sampling_lanczoc2Weights(float t) {
    vec4 x = vec4(t) + vec4(1.0, 0.0, -1.0, -2.0);
    return _sampling_lanczoc2(x);
}

vec4 _sampling_lanczoc22(vec4 x) {
    return _sampling_sincn(x) * _sampling_sincn(0.5 * x);
}

vec4 sampling_lanczoc2Weights(float t, float w) {
    vec4 x = vec4(t) + (vec4(1.0, 0.0, -1.0, -2.0) + 0.5) * w - 0.5;
    //    x = clamp(x, -2.0 / w, 2.0 / w);
    return _sampling_lanczoc22(x);
}

#if DERIVATIVE_AVAILABLE
// [QUI17]
vec3 sampling_textureRepeatGrad(sampler2D t, vec2 uv, float v) {
    float k = texture(iChannel1, 0.005 * uv).x; // cheap (cache friendly) lookup

    vec2 duvdx = dFdx(uv);
    vec2 duvdy = dFdy(uv);

    float l = k * 8.0;
    float f = fract(l);

    float ia = floor(l);
    float ib = ia + 1.0;

    vec2 offa = sin(vec2(3.0, 7.0) * ia); // can replace with any other hash
    vec2 offb = sin(vec2(3.0, 7.0) * ib); // can replace with any other hash

    vec3 cola = textureGrad(t, uv + v * offa, duvdx, duvdy).xyz;
    vec3 colb = textureGrad(t, uv + v * offb, duvdx, duvdy).xyz;

    return mix(cola, colb, smoothstep(0.2, 0.8, f - 0.1 * sum3(cola - colb)));
}
#endif


vec4 texture_tiling(sampler2D t, vec2 texCoord) {
    vec2 dist0 = smoothstep(0.0, 0.5, abs(fract(texCoord - vec2(0.5, 0.5)) - vec2(0.5, 0.5)));
    vec2 dist1 = smoothstep(0.0, 0.5, abs(fract(texCoord - vec2(0.5, 0.5) + vec2(0.5, 0.5)) - vec2(0.5, 0.5)));
    vec2 dist2 = smoothstep(0.0, 0.5, abs(fract(texCoord - vec2(0.5, 0.5) + vec2(1.0, 0.5)) - vec2(0.5, 0.5)));
    vec2 dist3 = smoothstep(0.0, 0.5, abs(fract(texCoord - vec2(0.5, 0.5) + vec2(0.5, 1.0)) - vec2(0.5, 0.5)));

    vec4 result = vec4(0.0);
    result += texture(t, texCoord) * dist0.x * dist0.y;
    result += texture(t, texCoord + vec2(0.5, 0.5)) * dist1.x * dist1.y;
    result += texture(t, texCoord + vec2(1.0, 0.5)) * dist2.x * dist2.y;
    result += texture(t, texCoord + vec2(0.5, 1.0)) * dist3.x * dist3.y;

    return result;
}

// [QUI17]
vec4 sampling_textureRepeat(sampler2D t, vec2 uv, float tsize, float v) {
    float k = textureLod(noisetex, tsize * uv, 0.0).x; // cheap (cache friendly) lookup

    float l = k * 8.0;
    float f = fract(l);

    float ia = floor(l);
    float ib = ia + 1.0;

    vec2 offa = sin(vec2(3.0, 7.0) * ia); // can replace with any other hash
    vec2 offb = sin(vec2(3.0, 7.0) * ib); // can replace with any other hash

    vec4 cola = textureLod(t, uv + v * offa, 0.0);
    vec4 colb = textureLod(t, uv + v * offb, 0.0);

    return mix(cola, colb, smoothstep(0.2, 0.8, f - 0.1 * sum4(cola - colb)));
}

// From https://github.com/GameTechDev/TAA and https://www.iryoku.com/downloads/Filmic-SMAA-v8.pptx
vec4 sampling_catmullBicubic5Tap(sampler2D texSampler, vec2 texelPos, float sharpness, vec2 texRcpSize){
    vec2 t = fract(texelPos - 0.5);
    vec2 centerUV = (floor(texelPos - 0.5) + vec2(0.5f, 0.5f)) * texRcpSize;

    // 5-tap bicubic sampling (for Hermite/Carmull-Rom filter) -- (approximate from original 16->9-tap bilinear fetching)
    vec2 f = t;
    vec2 f2 = t * t;
    vec2 f3 = t * t * t;

    float s = sharpness;
    vec2 w0 = -s * f3 + 2.0 * s * f2 - s * f;
    vec2 w1 = (2.0 - s) * f3 + (s - 3.0) * f2 + 1.0;
    vec2 w2 = (s - 2.0) * f3 + (3.0 - 2.0 * s) * f2 + s * f;
    vec2 w3 = s * f3 - s * f2;

    vec2 w12 = w1 + w2;

    vec2 tc12 = centerUV + (w2 / w12) * texRcpSize;
    vec2 tc0 = centerUV - 1.0f * texRcpSize;
    vec2 tc3 = centerUV + 2.0f * texRcpSize;

    vec4 c1 = vec4(texture(texSampler, vec2(tc12.x, tc0.y)));
    vec4 c2 = vec4(texture(texSampler, vec2(tc0.x, tc12.y)));
    vec4 c3 = vec4(texture(texSampler, vec2(tc12.x, tc12.y)));
    vec4 c4 = vec4(texture(texSampler, vec2(tc3.x, tc12.y)));
    vec4 c5 = vec4(texture(texSampler, vec2(tc12.x, tc3.y)));

    float weight1 = w12.x * w0.y;
    float weight2 = w0.x * w12.y;
    float weight3 = w12.x * w12.y;
    float weight4 = w3.x * w12.y;
    float weight5 = w12.x * w3.y;

    vec4 color = weight1 * c1;
    color += weight2 * c2;
    color += weight3 * c3;
    color += weight4 * c4;
    color += weight5 * c5;

    return color * rcp(weight1 + weight2 + weight3 + weight4 + weight5);
}

struct CatmullRomBicubic5TapData {
    vec3 uv1AndWeight;
    vec3 uv2AndWeight;
    vec3 uv3AndWeight;
    vec3 uv4AndWeight;
    vec3 uv5AndWeight;
};

// From https://github.com/GameTechDev/TAA and https://www.iryoku.com/downloads/Filmic-SMAA-v8.pptx
CatmullRomBicubic5TapData sampling_catmullRomBicubic5Tap_init(vec2 texelPos, float sharpness, vec2 texRcpSize){
    vec2 t = fract(texelPos - 0.5);
    vec2 centerUV = (floor(texelPos - 0.5) + 0.5) * texRcpSize;

    // 5-tap bicubic sampling (for Hermite/Carmull-Rom filter) -- (approximate from original 16->9-tap bilinear fetching)
    vec2 f = t;
    vec2 f2 = t * t;
    vec2 f3 = t * t * t;

    float s = sharpness;
    vec2 w0 = -s * f3 + 2.0 * s * f2 - s * f;
    vec2 w1 = (2.0 - s) * f3 + (s - 3.0) * f2 + 1.0;
    vec2 w2 = (s - 2.0) * f3 + (3.0 - 2.0 * s) * f2 + s * f;
    vec2 w3 = s * f3 - s * f2;

    vec2 w12 = w1 + w2;

    vec2 tc12 = centerUV + (w2 / w12) * texRcpSize;
    vec2 tc0 = centerUV - 1.0f * texRcpSize;
    vec2 tc3 = centerUV + 2.0f * texRcpSize;

    float weight1 = w12.x * w0.y;
    float weight2 = w0.x * w12.y;
    float weight3 = w12.x * w12.y;
    float weight4 = w3.x * w12.y;
    float weight5 = w12.x * w3.y;

    float weightSum = weight1 + weight2 + weight3 + weight4 + weight5;
    float weightSumRcp = rcp(weightSum);

    CatmullRomBicubic5TapData params;
    params.uv1AndWeight = vec3(vec2(tc12.x, tc0.y), weight1 * weightSumRcp);
    params.uv2AndWeight = vec3(vec2(tc0.x, tc12.y), weight2 * weightSumRcp);
    params.uv3AndWeight = vec3(vec2(tc12.x, tc12.y), weight3 * weightSumRcp);
    params.uv4AndWeight = vec3(vec2(tc3.x, tc12.y), weight4 * weightSumRcp);
    params.uv5AndWeight = vec3(vec2(tc12.x, tc3.y), weight5 * weightSumRcp);
    return params;
}

vec4 sampling_catmullBicubic5Tap_sum(vec4 c1, vec4 c2, vec4 c3, vec4 c4, vec4 c5, CatmullRomBicubic5TapData params){
    vec4 color = params.uv1AndWeight.z * c1;
    color += params.uv2AndWeight.z * c2;
    color += params.uv3AndWeight.z * c3;
    color += params.uv4AndWeight.z * c4;
    color += params.uv5AndWeight.z * c5;
    return color;
}

vec4 sampling_catmullBicubic5Tap(sampler2D texSampler, vec2 uv, float sharpness){
    vec2 texSize = vec2(textureSize(texSampler, 0));
    vec2 texelPos = uv * texSize;
    vec2 texRcpSize = rcp(texSize);
    return sampling_catmullBicubic5Tap(texSampler, texelPos, sharpness, texRcpSize);
}


// [DJO12]
// Optimized 4-tap B-Spline bicubic sampling using bilinear filtering
// Reduces 16 texture fetches to 4 by leveraging hardware bilinear interpolation
vec4 sampling_bSplineBicubic4Tap(sampler2D texSampler, vec2 texelPos, vec2 texRcpSize) {
    vec2 f = fract(texelPos - 0.5);
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;

    // Compute the B-Spline weights (optimized: w2 computed from sum constraint)
    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w3 = 0.5 * (f3 - f2);
    vec2 w2 = 1.0 - w0 - w1 - w3;

    // Combine weights for bilinear filtering
    vec2 s0 = w0 + w1;
    vec2 s1 = w2 + w3;

    // Compute texture coordinates that leverage bilinear filtering
    vec2 centerUV = (floor(texelPos - 0.5) + 0.5) * texRcpSize;
    vec2 t0 = centerUV + (w1 / s0 - 1.0) * texRcpSize;
    vec2 t1 = centerUV + (w3 / s1 + 1.0) * texRcpSize;

    // 4 bilinear samples instead of 16 point samples
    return (texture(texSampler, vec2(t0.x, t0.y)) * s0.x
    +  texture(texSampler, vec2(t1.x, t0.y)) * s1.x) * s0.y
    + (texture(texSampler, vec2(t0.x, t1.y)) * s0.x
    +  texture(texSampler, vec2(t1.x, t1.y)) * s1.x) * s1.y;
}

// Convenience wrapper that takes UV coordinates
vec4 sampling_bSplineBicubic4Tap(sampler2D texSampler, vec2 uv) {
    vec2 texSize = vec2(textureSize(texSampler, 0));
    vec2 texelPos = uv * texSize;
    vec2 texRcpSize = rcp(texSize);
    return sampling_bSplineBicubic4Tap(texSampler, texelPos, texRcpSize);
}

struct BSplineBicubic4TapData {
    vec2 uv00;
    vec2 uv10;
    vec2 uv01;
    vec2 uv11;
    vec2 weight0;
    vec2 weight1;
};

// Initialize B-Spline sampling parameters for reuse across multiple textures
BSplineBicubic4TapData sampling_bSplineBicubic4Tap_init(vec2 texelPos, vec2 texRcpSize) {
    vec2 f = fract(texelPos - 0.5);
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;

    // Compute the B-Spline weights
    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w3 = 0.5 * (f3 - f2);
    vec2 w2 = 1.0 - w0 - w1 - w3;

    // Combine weights for bilinear filtering
    vec2 s0 = w0 + w1;
    vec2 s1 = w2 + w3;

    // Compute texture coordinates
    vec2 centerUV = (floor(texelPos - 0.5) + 0.5) * texRcpSize;
    vec2 t0 = centerUV + (w1 / s0 - 1.0) * texRcpSize;
    vec2 t1 = centerUV + (w3 / s1 + 1.0) * texRcpSize;

    BSplineBicubic4TapData params;
    params.uv00 = vec2(t0.x, t0.y);
    params.uv10 = vec2(t1.x, t0.y);
    params.uv01 = vec2(t0.x, t1.y);
    params.uv11 = vec2(t1.x, t1.y);
    params.weight0 = s0;
    params.weight1 = s1;
    return params;
}

// Sum pre-fetched samples with B-Spline weights
vec4 sampling_bSplineBicubic4Tap_sum(vec4 c00, vec4 c10, vec4 c01, vec4 c11, BSplineBicubic4TapData params) {
    return (c00 * params.weight0.x + c10 * params.weight1.x) * params.weight0.y
    + (c01 * params.weight0.x + c11 * params.weight1.x) * params.weight1.y;
}



vec2 sampling_indexToGatherOffset(uint index) {
    //   _______ _______
    //  |       |       |
    //  |  x(0) |  y(1) |
    //  |_______o_______|  o gather location
    //  |       |       |
    //  |  w(3) |  z(2) |
    //  |_______|_______|
    // vec4 ofsetPosXs = vec4(-1.0, 1.0, 1.0, -1.0);
    // 0 - 00 = -1
    // 1 - 01 =  1
    // 2 - 10 =  1
    // 3 - 11 = -1
    // 0 + 1 = 1 -> 01
    // 1 + 1 = 2 -> 10
    // 2 + 1 = 3 -> 11
    // 3 + 1 = 4 -> 00

    // vec4 ofsetPosYs = vec4(1.0, 1.0, -1.0, -1.0);
    // vec2 ofsetPos = vec2(texelPosXs[index], texelPosYs[index]);

    uvec2 xyBits;
    xyBits.x = ((index + 1u) >> 1u) & 1u;
    xyBits.y = (~(index >> 1u)) & 1u;
    return vec2(xyBits) * 2.0 - 1.0;
}

// [MJP19] - 9-tap Catmull-Rom bicubic sampling
// More efficient than 16-tap full sampling while maintaining higher quality
// See: http://vec3.ca/bicubic-filtering-in-fewer-taps/ for details
// Uses bilinear filtering to evaluate 9 taps with only 9 texture fetches
vec4 sampling_catmullRomBicubic9Tap(sampler2D texSampler, vec2 texelPos, vec2 texRcpSize) {
    // Sample position and starting texel
    vec2 samplePos = texelPos;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    // Fractional offset from starting texel
    vec2 f = samplePos - texPos1;

    // Catmull-Rom weights for each axis
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);

    // Combine w1 and w2 for bilinear filtering optimization
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / w12;

    // Compute UV coordinates (in texture space)
    vec2 texPos0 = (texPos1 - 1.0) * texRcpSize;
    vec2 texPos3 = (texPos1 + 2.0) * texRcpSize;
    vec2 texPos12 = (texPos1 + offset12) * texRcpSize;

    // 9 texture samples with bilinear filtering
    vec4 result = vec4(0.0);
    result += texture(texSampler, vec2(texPos0.x, texPos0.y)) * w0.x * w0.y;
    result += texture(texSampler, vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += texture(texSampler, vec2(texPos3.x, texPos0.y)) * w3.x * w0.y;

    result += texture(texSampler, vec2(texPos0.x, texPos12.y)) * w0.x * w12.y;
    result += texture(texSampler, vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += texture(texSampler, vec2(texPos3.x, texPos12.y)) * w3.x * w12.y;

    result += texture(texSampler, vec2(texPos0.x, texPos3.y)) * w0.x * w3.y;
    result += texture(texSampler, vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += texture(texSampler, vec2(texPos3.x, texPos3.y)) * w3.x * w3.y;

    return result;
}

// Convenience wrapper that takes UV coordinates (0..1 range)
vec4 sampling_catmullRomBicubic9Tap(sampler2D texSampler, vec2 uv) {
    vec2 texSize = vec2(textureSize(texSampler, 0));
    vec2 texelPos = uv * texSize;
    vec2 texRcpSize = rcp(texSize);
    return sampling_catmullRomBicubic9Tap(texSampler, texelPos, texRcpSize);
}

// Data structure for 9-tap sampling initialization (for reuse across multiple textures)
struct CatmullRomBicubic9TapData {
    vec2 uv00;
    vec2 uv12_0;
    vec2 uv30;
    vec2 uv01_2;
    vec2 uv12_12;
    vec2 uv31_2;
    vec2 uv03;
    vec2 uv12_3;
    vec2 uv33;
    vec2 weight0;
    vec2 weight12;
    vec2 weight3;
};

// Initialize 9-tap Catmull-Rom sampling parameters for reuse across multiple textures
CatmullRomBicubic9TapData sampling_catmullRomBicubic9Tap_init(vec2 texelPos, vec2 texRcpSize) {
    // Sample position and starting texel
    vec2 samplePos = texelPos;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    // Fractional offset from starting texel
    vec2 f = samplePos - texPos1;

    // Catmull-Rom weights for each axis
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);

    // Combine w1 and w2 for bilinear filtering optimization
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / w12;

    // Compute UV coordinates (in texture space)
    vec2 texPos0 = (texPos1 - 1.0) * texRcpSize;
    vec2 texPos3 = (texPos1 + 2.0) * texRcpSize;
    vec2 texPos12 = (texPos1 + offset12) * texRcpSize;

    CatmullRomBicubic9TapData params;
    params.uv00 = vec2(texPos0.x, texPos0.y);
    params.uv12_0 = vec2(texPos12.x, texPos0.y);
    params.uv30 = vec2(texPos3.x, texPos0.y);
    params.uv01_2 = vec2(texPos0.x, texPos12.y);
    params.uv12_12 = vec2(texPos12.x, texPos12.y);
    params.uv31_2 = vec2(texPos3.x, texPos12.y);
    params.uv03 = vec2(texPos0.x, texPos3.y);
    params.uv12_3 = vec2(texPos12.x, texPos3.y);
    params.uv33 = vec2(texPos3.x, texPos3.y);
    params.weight0 = w0;
    params.weight12 = w12;
    params.weight3 = w3;
    return params;
}

// Sum pre-fetched 9 samples with Catmull-Rom weights
vec4 sampling_catmullRomBicubic9Tap_sum(vec4 c00, vec4 c12_0, vec4 c30,
                                        vec4 c01_2, vec4 c12_12, vec4 c31_2,
                                        vec4 c03, vec4 c12_3, vec4 c33,
                                        CatmullRomBicubic9TapData params) {
    vec4 result = vec4(0.0);
    result += c00 * params.weight0.x * params.weight0.y;
    result += c12_0 * params.weight12.x * params.weight0.y;
    result += c30 * params.weight3.x * params.weight0.y;

    result += c01_2 * params.weight0.x * params.weight12.y;
    result += c12_12 * params.weight12.x * params.weight12.y;
    result += c31_2 * params.weight3.x * params.weight12.y;

    result += c03 * params.weight0.x * params.weight3.y;
    result += c12_3 * params.weight12.x * params.weight3.y;
    result += c33 * params.weight3.x * params.weight3.y;

    return result;
}

#endif