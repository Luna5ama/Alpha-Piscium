#include "Common.glsl"
#include "/util/Dither.glsl"
#include "/util/NZPacking.glsl"
#include "/util/Rand.glsl"
#include "/util/Colors.glsl"
#include "/techniques/HiZ.glsl"

#define ATROUS_THREAD_SIZE 128

#if ATROUS_PASS == 1
#define ATROUS_TAP_COUNT 2
#define ATROUS_AXIS_X a
#define ATROUS_RADIUS 2
#define ATROUS_INPUT usam_temp2
#define ATROUS_OUTPUT uimg_temp1

#elif ATROUS_PASS == 2
#define ATROUS_TAP_COUNT 2
#define ATROUS_AXIS_Y a
#define ATROUS_RADIUS 2
#define ATROUS_INPUT usam_temp1
#define ATROUS_OUTPUT uimg_temp2


#elif ATROUS_PASS == 3
#define ATROUS_TAP_COUNT 4
#define ATROUS_AXIS_X a
#define ATROUS_RADIUS 8
#define ATROUS_INPUT usam_temp2
#define ATROUS_OUTPUT uimg_temp1

#elif ATROUS_PASS == 4
#define ATROUS_TAP_COUNT 4
#define ATROUS_AXIS_Y a
#define ATROUS_RADIUS 8
#define ATROUS_INPUT usam_temp1
#define ATROUS_OUTPUT uimg_temp2
#endif

#define SHARED_DATA_OFFSET (ATROUS_RADIUS * ATROUS_TAP_COUNT)
#define SHARED_DATA_SIZE (ATROUS_THREAD_SIZE + SHARED_DATA_OFFSET * 2)

#ifdef ATROUS_AXIS_X
layout(local_size_x = ATROUS_THREAD_SIZE, local_size_y = 1) in;
#define ATROUS_AXIS_VEC ivec2(1, 0)
#else
layout(local_size_x = 1, local_size_y = ATROUS_THREAD_SIZE) in;
#define ATROUS_AXIS_VEC ivec2(0, 1)
#endif


const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D ATROUS_OUTPUT;

shared uvec4 shared_data[SHARED_DATA_SIZE];

ivec2 atrous_texelPos;
float atrous_normalWeight = 0.0;
float atrous_viewZWeight = 0.0;
float atrous_luminanceWeight = 0.0;

void loadGlobalData(ivec2 loadTexelPos, out vec4 color, out vec3 normal, out float viewZ) {
    color = texelFetch(ATROUS_INPUT, loadTexelPos, 0);
    uvec4 packedData = uvec4(0u);
    nzpacking_unpack(texelFetch(usam_packedZN, loadTexelPos + ivec2(0, global_mainImageSizeI.y), 0).xy, normal, viewZ);
    normal = mat3(gbufferModelViewInverse) * normal;
}

void initSharedData(uint index) {
    if (index < SHARED_DATA_SIZE) {
        int pos1d = int(index) - SHARED_DATA_OFFSET;
        ivec2 loadTexelPos = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy);
        loadTexelPos += ATROUS_AXIS_VEC * pos1d;
        loadTexelPos += int(rand_stbnUnitVec211(loadTexelPos, frameCounter) * ATROUS_RADIUS);
        loadTexelPos = clamp(loadTexelPos, ivec2(0), global_mainImageSizeI - 1);
        vec4 color;
        vec3 normal;
        float viewZ;
        loadGlobalData(loadTexelPos, color, normal, viewZ);
        uvec4 packedData = uvec4(0u);
        packedData.x = packHalf2x16(color.rg);
        packedData.y = packHalf2x16(color.ba);
        packedData.z = packSnorm4x8(vec4(normal, 0.0));
        packedData.w = floatBitsToUint(viewZ);
        shared_data[index] = packedData;
    }
}

void loadSharedData(int offset, out vec4 color, out vec3 normal, out float viewZ) {
    int loadIndex = int(gl_LocalInvocationIndex) + offset + SHARED_DATA_OFFSET;
    loadIndex = clamp(loadIndex, 0, SHARED_DATA_SIZE - 1);
    uvec4 packedData = shared_data[loadIndex];
    color = vec4(unpackHalf2x16(packedData.x), unpackHalf2x16(packedData.y));
    normal = unpackSnorm4x8(packedData.z).xyz;
    viewZ = uintBitsToFloat(packedData.w);
}

float normalWeight(vec3 centerNormal, vec3 sampleNormal, float phi) {
    float sdot = saturate(dot(centerNormal, sampleNormal));
    return pow(sdot, phi);
}

float viewZWeight(float centerViewZ, float sampleViewZ, float phi) {
    return phi / (phi + pow2(centerViewZ - sampleViewZ));
}

float luminanceWeight(float centerLuminance, float sampleLuminance, float phi) {
    return exp2(abs(centerLuminance - sampleLuminance) * phi);
}

void atrousSample(
vec3 centerNormal, float centerViewZ, float centerLuminance,
int offset,
inout vec4 colorSum, inout float weightSum
) {
    int realOffset = offset * ATROUS_RADIUS;
    ivec2 texelPos = atrous_texelPos + realOffset * ATROUS_AXIS_VEC;
    if (all(greaterThanEqual(texelPos, ivec2(0))) && all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 sampleColor;
        vec3 sampleNormal;
        float sampleViewZ;
        loadSharedData(realOffset, sampleColor, sampleNormal, sampleViewZ);

        float sampleLuminance = colors_sRGB_luma(sampleColor.rgb);

        float weight = 1.0;
        weight *= normalWeight(centerNormal, sampleNormal, atrous_normalWeight);
        weight *= viewZWeight(centerViewZ, sampleViewZ, atrous_viewZWeight);
        weight *= luminanceWeight(centerLuminance, sampleLuminance, atrous_luminanceWeight * float(abs(realOffset)));

        colorSum += sampleColor * vec4(vec3(weight), weight * weight);
        weightSum += weight;
    }
}

const float _ATROUS_MIN_VARIANCE_FACTOR = exp2(-SETTING_DENOISER_MIN_VARIANCE_FACTOR);

vec4 atrous_atrous(ivec2 texelPos) {
    atrous_texelPos = texelPos;

    initSharedData(gl_LocalInvocationIndex);
    initSharedData(gl_LocalInvocationIndex + ATROUS_THREAD_SIZE);
    barrier();

    vec4 outputColor = vec4(0.0);

    if (all(lessThan(atrous_texelPos, global_mainImageSizeI))) {
        vec4 centerFilterData;
        vec3 centerNormal;
        float centerViewZ;
        loadGlobalData(atrous_texelPos, centerFilterData, centerNormal, centerViewZ);

        if (centerViewZ != -65536.0) {
            vec3 centerColor = centerFilterData.rgb;
            float centerVariance = centerFilterData.a;
            float centerLuminance = colors_sRGB_luma(centerColor);

            float sigmaL = 0.001 * SETTING_DENOISER_FILTER_COLOR_WEIGHT;

            atrous_normalWeight = SETTING_DENOISER_FILTER_NORMAL_WEIGHT;
            atrous_viewZWeight = max((1.0 / SETTING_DENOISER_FILTER_DEPTH_WEIGHT) * pow2(centerViewZ), 0.5);
            atrous_luminanceWeight = -sigmaL * inversesqrt(max(centerVariance, _ATROUS_MIN_VARIANCE_FACTOR));

            vec4 colorSum = centerFilterData * 1.0;
            float weightSum = 1.0;

            #if ATROUS_TAP_COUNT >= 8
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                -8,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 7
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                -7,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 6
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                -6,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 5
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                -5,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 4
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                -4,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 3
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                -3,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 2
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                -2,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 1
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                -1,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 1
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                1,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 2
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                2,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 3
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                3,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 4
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                4,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 5
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                5,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 6
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                6,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 7
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                7,
                colorSum, weightSum
            );
            #endif

            #if ATROUS_TAP_COUNT >= 8
            atrousSample(
                centerNormal, centerViewZ, centerLuminance,
                8,
                colorSum, weightSum
            );
            #endif

            outputColor = colorSum / vec4(vec3(weightSum), weightSum * weightSum);
            outputColor = max(outputColor, 0.0);
        }
    }

    return outputColor;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    vec4 outputColor = atrous_atrous(texelPos);
//    outputColor = dither_fp16(outputColor, rand_IGN(texelPos, frameCounter + ATROUS_PASS));

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        imageStore(ATROUS_OUTPUT, texelPos, outputColor);
    }
}