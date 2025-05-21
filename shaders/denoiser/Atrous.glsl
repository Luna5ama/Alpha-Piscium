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
shared uvec4 shared_data[SHARED_DATA_SIZE];

void initSharedData(uint index) {
    if (index < SHARED_DATA_SIZE) {
        ivec2 loadTexelPos = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy);
        loadTexelPos += ATROUS_AXIS_VEC * (int(index) - SHARED_DATA_OFFSET);
        loadTexelPos = clamp(loadTexelPos, ivec2(0), global_mainImageSizeI - 1);
        vec4 color = texelFetch(ATROUS_INPUT, loadTexelPos, 0);
        uvec4 packedData = uvec4(0u);
        packedData.x = packHalf2x16(color.rg);
        packedData.y = packHalf2x16(color.ba);
        vec3 normal;
        float viewZ;
        nzpacking_unpack(texelFetch(usam_packedZN, loadTexelPos + ivec2(0, global_mainImageSizeI.y), 0).xy, normal, viewZ);
        packedData.z = packSnorm4x8(vec4(normal, 0.0));
        packedData.w = floatBitsToUint(viewZ);
        shared_data[index] = packedData;
    }
}

void loadSharedData(int offset, out vec4 color, out vec3 normal, out float viewZ) {
    uvec4 packedData = shared_data[gl_LocalInvocationIndex + offset + SHARED_DATA_OFFSET];
    color = vec4(unpackHalf2x16(packedData.x), unpackHalf2x16(packedData.y));
    normal = unpackSnorm4x8(packedData.z).xyz;
    viewZ = uintBitsToFloat(packedData.w);
}

void atrousSample(
vec3 centerNormal, float centerViewZ, float centerLuminance,
float phiN, float phiZ, float phiL,
int offset, float baseWeight,
inout vec4 colorSum, inout float weightSum
) {
    int realOffset = offset * ATROUS_RADIUS;
    ivec2 texelPos = svgf_texelPos + realOffset * ATROUS_AXIS_VEC;
    if (all(greaterThanEqual(texelPos, ivec2(0))) && all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 sampleColor;
        vec3 sampleNormal;
        float sampleViewZ;
        loadSharedData(realOffset, sampleColor, sampleNormal, sampleViewZ);

        float sampleLuminance = colors_srgbLuma(sampleColor.rgb);

        float weight = baseWeight;
        weight *= normalWeight(centerNormal, sampleNormal, phiN);
        weight *= viewZWeight(centerViewZ, sampleViewZ, phiZ);
        weight *= luminanceWeight(centerLuminance, sampleLuminance, phiL * float(abs(realOffset)));

        colorSum += sampleColor * vec4(vec3(weight), weight * weight);
        weightSum += weight;
    }
}

vec4 svgf_atrous(ivec2 texelPos) {
    svgf_texelPos = texelPos;

    initSharedData(gl_LocalInvocationIndex);
    initSharedData(gl_LocalInvocationIndex + 128);
    barrier();

    vec4 outputColor = vec4(0.0);

    if (all(lessThan(svgf_texelPos, global_mainImageSizeI))) {
        vec4 centerFilterData;
        vec3 centerNormal;
        float centerViewZ;
        loadSharedData(0, centerFilterData, centerNormal, centerViewZ);

        if (centerViewZ != -65536.0) {
            vec3 centerColor = centerFilterData.rgb;

            float centerVariance = centerFilterData.a;
            float centerLuminance = colors_srgbLuma(centerColor);

            vec2 hLenV = texelFetch(usam_temp6, svgf_texelPos, 0).xy;
            float sigmaL = pow2(hLenV.x);
            sigmaL *= mix(SETTING_DENOISER_FILTER_INIT_COLOR_WEIGHT, SETTING_DENOISER_FILTER_FINAL_COLOR_WEIGHT, hLenV.y);
            sigmaL *= 0.01;

            float phiN = SETTING_DENOISER_FILTER_NORMAL_WEIGHT;
            float phiZ = max((1.0 / SETTING_DENOISER_FILTER_DEPTH_WEIGHT) * pow2(centerViewZ), 0.5);
            float phiL = sigmaL / max(sqrt(centerVariance), 0.01);

            vec4 colorSum = centerFilterData * 1.0;
            float weightSum = 1.0;

            float kernelDecay = mix(SETTING_DENOISER_FILTER_KERNEL_INIT_SIGMA, SETTING_DENOISER_FILTER_KERNEL_FINAL_SIGMA, pow2(hLenV.y));
            float baseWeight1 = exp2(-1.0 * kernelDecay);
            float baseWeight2 = exp2(-2.0 * kernelDecay);

            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                phiN, phiZ, phiL,
                -2, baseWeight2,
                colorSum, weightSum
            );

            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                phiN, phiZ, phiL,
                -1, baseWeight1,
                colorSum, weightSum
            );

            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                phiN, phiZ, phiL,
                1, baseWeight1,
                colorSum, weightSum
            );

            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                phiN, phiZ, phiL,
                2, baseWeight2,
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
        packedData.x = packUnorm4x8(colors_SRGBToLogLuv(outputColor.rgb));
        imageStore(uimg_svgfHistory, texelPos, packedData);
        #endif
    }
}