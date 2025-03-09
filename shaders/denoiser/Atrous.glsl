#include "Common.glsl"
#include "/general/NDPacking.glsl"
#include "/util/Colors.glsl"

float normalWeight(vec3 centerNormal, vec3 sampleNormal, float phiN) {
    float sdot = saturate(dot(centerNormal, sampleNormal));
    return pow(sdot, phiN);
}

float viewZWeight(float centerViewZ, float sampleViewZ, float phiZ) {
    return phiZ / (phiZ + pow2(centerViewZ - sampleViewZ));
}

float luminanceWeight(float centerLuminance, float sampleLuminance, float phiL) {
    return exp(-(abs(centerLuminance - sampleLuminance) * phiL));
}

void atrousSample(
sampler2D filterInput, usampler2D packedNZ,
vec3 centerNormal, float centerViewZ, float centerLuminance,
float phiN, float phiZ, float phiL,
ivec2 texelPos, float sampleWeight,
inout vec4 colorSum, inout float weightSum
) {
    if (all(greaterThanEqual(texelPos, ivec2(0))) && all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 sampleColor = texelFetch(filterInput, texelPos, 0);
        vec3 sampleNormal;
        float sampleViewZ;
        ndpacking_unpack(texelFetch(packedNZ, texelPos, 0).xy, sampleNormal, sampleViewZ);

        float sampleLuminance = colors_srgbLuma(sampleColor.rgb);

        float weight = sampleWeight;
        weight *= normalWeight(centerNormal, sampleNormal, phiN);
        weight *= viewZWeight(centerViewZ, sampleViewZ, phiZ);
        weight *= luminanceWeight(centerLuminance, sampleLuminance, phiL);

        colorSum += sampleColor * vec4(vec3(weight), weight * weight);
        weightSum += weight;
    }
}

vec4 svgf_atrous(sampler2D filterInput, usampler2D packedNZ, ivec2 texelPos, ivec2 radiusOffset, float sigmaL) {
    vec4 outputColor = vec4(0.0);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 centerFilterData = texelFetch(filterInput, texelPos, 0);
        vec3 centerColor = centerFilterData.rgb;

        float centerVariance = 0.0;
        float a = texelFetchOffset(filterInput, texelPos, 0, ivec2(-1, 0)).a;
        float c = texelFetchOffset(filterInput, texelPos, 0, ivec2(1, 0)).a;
        float g = texelFetchOffset(filterInput, texelPos, 0, ivec2(0, -1)).a;
        float i = texelFetchOffset(filterInput, texelPos, 0, ivec2(0, 1)).a;
        centerVariance += (a + c + g + i) * 0.0625;

        float b = texelFetchOffset(filterInput, texelPos, 0, ivec2(-1, -1)).a;
        float d = texelFetchOffset(filterInput, texelPos, 0, ivec2(1, -1)).a;
        float f  = texelFetchOffset(filterInput, texelPos, 0, ivec2(-1, 1)).a;
        float h = texelFetchOffset(filterInput, texelPos, 0, ivec2(1, 1)).a;
        centerVariance += (b + d + f + h) * 0.125;

        float e = centerFilterData.a;
        centerVariance += e * 0.25;

        float centerLuminance = colors_srgbLuma(centerColor);

        vec3 centerNormal;
        float centerViewZ;
        ndpacking_unpack(texelFetch(packedNZ, texelPos, 0).xy, centerNormal, centerViewZ);

        float phiN = SETTING_DENOISER_FILTER_NORMAL_STRICTNESS;
        float phiZ = max(0.01 * pow2(centerViewZ), 0.5);
        float phiL = (1.0 / sigmaL) * max(sqrt(centerVariance), 1e-10);
        phiL = 1.0 / phiL;

        vec4 colorSum = centerFilterData * 1.0;
        float weightSum = 1.0;

        atrousSample(
            filterInput, packedNZ,
            centerNormal, centerViewZ, centerLuminance,
            phiN, phiZ, phiL,
            -2 * radiusOffset + texelPos, 0.25,
            colorSum, weightSum
        );

        atrousSample(
            filterInput, packedNZ,
            centerNormal, centerViewZ, centerLuminance,
            phiN, phiZ, phiL,
            -1 * radiusOffset + texelPos, 0.5,
            colorSum, weightSum
        );


        atrousSample(
            filterInput, packedNZ,
            centerNormal, centerViewZ, centerLuminance,
            phiN, phiZ, phiL,
            1 * radiusOffset + texelPos, 0.5,
            colorSum, weightSum
        );

        atrousSample(
            filterInput, packedNZ,
            centerNormal, centerViewZ, centerLuminance,
            phiN, phiZ, phiL,
            2 * radiusOffset + texelPos, 0.25,
            colorSum, weightSum
        );

        outputColor = colorSum / vec4(vec3(weightSum), weightSum * weightSum);
        outputColor = max(outputColor, 0.0);
    }

    return outputColor;
}