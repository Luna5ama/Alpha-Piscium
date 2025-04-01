#include "Common.glsl"
#include "/util/Dither.glsl"
#include "/util/NZPacking.glsl"
#include "/util/Rand.glsl"
#include "/util/Colors.glsl"

#if ATROUS_PASS == 1
#define ATROUS_AXIS_X a
#define ATROUS_RADIUS 2
#define ATROUS_INPUT usam_temp2
#define ATROUS_OUTPUT uimg_temp1

#elif ATROUS_PASS == 2
#define ATROUS_AXIS_Y a
#define ATROUS_RADIUS 2
#define ATROUS_INPUT usam_temp1
#define ATROUS_OUTPUT uimg_temp2


#elif ATROUS_PASS == 3
#define ATROUS_AXIS_X a
#define ATROUS_RADIUS 4
#define ATROUS_INPUT usam_temp2
#define ATROUS_OUTPUT uimg_temp1

#elif ATROUS_PASS == 4
#define ATROUS_AXIS_Y a
#define ATROUS_RADIUS 4
#define ATROUS_INPUT usam_temp1
#define ATROUS_OUTPUT uimg_temp2


#elif ATROUS_PASS == 5
#define ATROUS_AXIS_X a
#define ATROUS_RADIUS 8
#define ATROUS_INPUT usam_temp2
#define ATROUS_OUTPUT uimg_temp1

#elif ATROUS_PASS == 6
#define ATROUS_AXIS_Y a
#define ATROUS_RADIUS 8
#define ATROUS_INPUT usam_temp1
#define ATROUS_OUTPUT uimg_temp2


#elif ATROUS_PASS == 7
#define ATROUS_AXIS_X a
#define ATROUS_RADIUS 16
#define ATROUS_INPUT usam_temp2
#define ATROUS_OUTPUT uimg_temp1

#elif ATROUS_PASS == 8
#define ATROUS_AXIS_Y a
#define ATROUS_RADIUS 16
#define ATROUS_INPUT usam_temp1
#define ATROUS_OUTPUT uimg_temp2


#elif ATROUS_PASS == 9
#define ATROUS_AXIS_X a
#define ATROUS_RADIUS 32
#define ATROUS_INPUT usam_temp2
#define ATROUS_OUTPUT uimg_temp1

#elif ATROUS_PASS == 10
#define ATROUS_AXIS_Y a
#define ATROUS_RADIUS 32
#define ATROUS_INPUT usam_temp1
#define ATROUS_OUTPUT uimg_temp2
#endif

#if ATROUS_PASS == 10
#define ATROUS_UPDATE_HISTORY a
#endif

#ifdef ATROUS_AXIS_X
layout(local_size_x = 128, local_size_y = 1) in;
#define ATROUS_AXIS_VEC ivec2(1, 0)
#else
layout(local_size_x = 1, local_size_y = 128) in;
#define ATROUS_AXIS_VEC ivec2(0, 1)
#endif
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D ATROUS_INPUT;
uniform sampler2D usam_temp6;
uniform usampler2D usam_packedZN;
layout(rgba16f) uniform writeonly image2D ATROUS_OUTPUT;
#ifdef ATROUS_UPDATE_HISTORY
layout(rgba32ui) uniform restrict uimage2D uimg_svgfHistory;
#endif

float normalWeight(vec3 centerNormal, vec3 sampleNormal, float phiN) {
    float sdot = saturate(dot(centerNormal, sampleNormal));
    return pow(sdot, phiN);
}

float viewZWeight(float centerViewZ, float sampleViewZ, float phiZ) {
    return phiZ / (phiZ + pow2(centerViewZ - sampleViewZ));
}

float luminanceWeight(float centerLuminance, float sampleLuminance, float phiL) {
    return exp(-(sqrt(abs(centerLuminance - sampleLuminance)) * phiL));
}

#define SHARED_DATA_SIZE (128 + ATROUS_RADIUS * 4)
#define SHARED_DATA_OFFSET (ATROUS_RADIUS * 2)
ivec2 svgf_texelPos;
shared uvec2 shared_colorData[SHARED_DATA_SIZE];

void loadShared(uint index) {
    if (index < SHARED_DATA_SIZE) {
        ivec2 loadTexelPos = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy);
        loadTexelPos += ATROUS_AXIS_VEC * (int(index) - SHARED_DATA_OFFSET);
        loadTexelPos = clamp(loadTexelPos, ivec2(0), global_mainImageSizeI - 1);
        vec4 color = texelFetch(ATROUS_INPUT, loadTexelPos, 0);
        uvec2 packedColor = uvec2(packHalf2x16(color.rg), packHalf2x16(color.ba));
        shared_colorData[index] = packedColor;
    }
}

vec4 readSharedColor(int offset) {
    uvec2 packedColor = shared_colorData[gl_LocalInvocationIndex + offset + SHARED_DATA_OFFSET];
    return vec4(unpackHalf2x16(packedColor.x), unpackHalf2x16(packedColor.y));
}

void atrousSample(
vec3 centerNormal, float centerViewZ, float centerLuminance,
float phiN, float phiZ, float phiL,
int offset, float sampleWeight,
inout vec4 colorSum, inout float weightSum
) {
    ivec2 texelPos = svgf_texelPos + offset * ATROUS_AXIS_VEC * ATROUS_RADIUS;
    if (all(greaterThanEqual(texelPos, ivec2(0))) && all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 sampleColor = readSharedColor(offset);
        vec3 sampleNormal;
        float sampleViewZ;
        nzpacking_unpack(texelFetch(usam_packedZN, texelPos + ivec2(0, global_mainImageSizeI.y), 0).xy, sampleNormal, sampleViewZ);

        float sampleLuminance = colors_srgbLuma(sampleColor.rgb);

        float weight = sampleWeight;
        weight *= normalWeight(centerNormal, sampleNormal, phiN);
        weight *= viewZWeight(centerViewZ, sampleViewZ, phiZ);
        weight *= luminanceWeight(centerLuminance, sampleLuminance, phiL);

        colorSum += sampleColor * vec4(vec3(weight), weight * weight);
        weightSum += weight;
    }
}

vec4 svgf_atrous(ivec2 texelPos) {
    svgf_texelPos = texelPos;

    loadShared(gl_LocalInvocationIndex);
    loadShared(gl_LocalInvocationIndex + 128);
    barrier();

    vec4 outputColor = vec4(0.0);

    if (all(lessThan(svgf_texelPos, global_mainImageSizeI))) {
        vec3 centerNormal;
        float centerViewZ;
        nzpacking_unpack(texelFetch(usam_packedZN, svgf_texelPos + ivec2(0, global_mainImageSizeI.y), 0).xy, centerNormal, centerViewZ);

        if (centerViewZ != -65536.0) {
            vec4 centerFilterData = readSharedColor(0);
            vec3 centerColor = centerFilterData.rgb;

            float centerVariance = centerFilterData.a;
            float centerLuminance = colors_srgbLuma(centerColor);

            float hDecay = texelFetch(usam_temp6, svgf_texelPos, 0).r;
            float sigmaL = ATROUS_RADIUS * SETTING_DENOISER_FILTER_COLOR_STRICTNESS * pow2(hDecay) * 0.02;
            float phiN = SETTING_DENOISER_FILTER_NORMAL_STRICTNESS;
            float phiZ = max((1.0 / SETTING_DENOISER_FILTER_DEPTH_STRICTNESS) * pow2(centerViewZ), 0.5);
            float phiL = sigmaL / max(sqrt(centerVariance), 0.01);

            vec4 colorSum = centerFilterData * 1.0;
            float weightSum = 1.0;

            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                phiN, phiZ, phiL,
                -2, mix(1.0, 0.25, hDecay),
                colorSum, weightSum
            );

            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                phiN, phiZ, phiL,
                -1, mix(1.0, 0.5, hDecay),
                colorSum, weightSum
            );


            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                phiN, phiZ, phiL,
                1, mix(1.0, 0.5, hDecay),
                colorSum, weightSum
            );

            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                phiN, phiZ, phiL,
                2, mix(1.0, 0.25, hDecay),
                colorSum, weightSum
            );

            outputColor = colorSum / vec4(vec3(weightSum), weightSum * weightSum);
            outputColor = max(outputColor, 0.0);
        }
    }

    return outputColor;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    vec4 outputColor = svgf_atrous(texelPos);
    outputColor = dither_fp16(outputColor, rand_IGN(texelPos, frameCounter + ATROUS_PASS));

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        imageStore(ATROUS_OUTPUT, texelPos, outputColor);

        #ifdef ATROUS_UPDATE_HISTORY
        uvec4 packedData = imageLoad(uimg_svgfHistory, texelPos);
        vec3 color;
        vec3 fastColor;
        vec2 moments;
        float hLen;
        svgf_unpack(packedData, color, fastColor, moments, hLen);
        color = outputColor.rgb;
        svgf_pack(packedData, color, fastColor, moments, hLen);
        imageStore(uimg_svgfHistory, texelPos, packedData);
        #endif
    }
}