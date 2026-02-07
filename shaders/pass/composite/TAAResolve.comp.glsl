#include "/util/Coords.glsl"
#include "/util/Colors.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Dither.glsl"
#include "/util/Rand.glsl"
#include "/util/Sampling.glsl"
#include "/techniques/gi/Common.glsl"
#include "/util/AgxInvertible.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict writeonly image2D uimg_temp1;
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;

// Shared memory with padding for 4x4 tap (-2 to +2)
// Each work group is 16x16, need +2 padding on each side for Lanczos2 4x4 taps
shared vec3 shared_colorData[20][20];
shared vec4 shared_weightsX;
shared vec4 shared_weightsY;
shared float shared_kernelDist2[9];

struct ColorAABB {
    vec3 minVal;
    vec3 maxVal;
    vec3 moment1;
    vec3 moment2;
    float weightSum;
};

ColorAABB initAABB(vec3 colorYCoCg, float weight) {
    ColorAABB box;
    box.minVal = colorYCoCg;
    box.maxVal = colorYCoCg;
    box.moment1 = colorYCoCg * weight;
    box.moment2 = colorYCoCg * colorYCoCg * weight;
    box.weightSum = weight;
    return box;
}

void updateAABB(vec3 color, float weight, inout ColorAABB box) {
    box.minVal = min(box.minVal, color);
    box.maxVal = max(box.maxVal, color);
    box.moment1 += color * weight;
    box.moment2 += color * color * weight;
    box.weightSum += weight;
}

float kernelWeight(vec2 centerPos, vec2 samplePos, float param) {
    vec2 diff = abs(samplePos - centerPos);
    float dist2 = dot(diff, diff);
    return exp(param * dist2);
}

void loadSharedColorData(uvec2 groupOriginTexelPos, uint index) {
    if (index < 400u) { // 20 * 20 = 400
        uvec2 sharedXY = uvec2(index % 20u, index / 20u);
        ivec2 srcXY = ivec2(groupOriginTexelPos) + ivec2(sharedXY) - 2;
        srcXY = clamp(srcXY, ivec2(0), ivec2(uval_mainImageSize - 1));
        vec3 colorData = texelFetch(usam_main, srcXY, 0).rgb;
        shared_colorData[sharedXY.y][sharedXY.x] = colors_RGBToYCoCg(colorData);
    }
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    vec2 texelCenter = vec2(texelPos) + vec2(0.5);
    vec2 screenPos = texelCenter * uval_mainImageSizeRcp;

    // Calculate work group origin for shared memory loading
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4u; // * 16

    // Load shared memory cooperatively (20x20 = 400 elements, 16x16 = 256 threads)
    loadSharedColorData(workGroupOrigin, gl_LocalInvocationIndex);
    loadSharedColorData(workGroupOrigin, gl_LocalInvocationIndex + 256u);

    if (gl_LocalInvocationIndex == 0u) {
        vec2 pixelPosFract = fract(uval_taaJitter);

        #if SETTING_TAA_CURR_FILTER == 0
        shared_weightsX = sampling_bSplineWeights(pixelPosFract.x);
        shared_weightsY = sampling_bSplineWeights(pixelPosFract.y);
        #elif SETTING_TAA_CURR_FILTER == 1
        shared_weightsX = sampling_catmullRomWeights(pixelPosFract.x);
        shared_weightsY = sampling_catmullRomWeights(pixelPosFract.y);
        #elif SETTING_TAA_CURR_FILTER == 2
        shared_weightsX = sampling_lanczoc2Weights(pixelPosFract.x);
        shared_weightsY = sampling_lanczoc2Weights(pixelPosFract.y);
        #endif

        for (int i = 0; i < 9; ++i) {
            vec2 offset = vec2(i % 3, i / 3) - 1.0;
            vec2 diff = offset - 0.5 - uval_taaJitter;
            shared_kernelDist2[i] = dot(diff, diff);
        }
    }

    barrier();

    vec2 unjitterTexelPos = texelCenter + uval_taaJitter;
    vec2 unjitterScreenPos = screenPos + uval_taaJitter * uval_mainImageSizeRcp;

    // Looks like this is fast enough without shared memory
    float currViewZ = texelFetch(usam_gbufferSolidViewZ, texelPos, 0).r;
    ivec2 offsetNeg = max(ivec2(-1, -1) + texelPos, ivec2(0)) - texelPos;
    ivec2 offsetPos = min(ivec2(1, 1) + texelPos, uval_mainImageSizeI - 1) - texelPos;
    currViewZ = max(texelFetch(usam_gbufferSolidViewZ, texelPos + ivec2(offsetNeg.x, 0), 0).r, currViewZ);
    currViewZ = max(texelFetch(usam_gbufferSolidViewZ, texelPos + ivec2(offsetPos.x, 0), 0).r, currViewZ);
    currViewZ = max(texelFetch(usam_gbufferSolidViewZ, texelPos + ivec2(0, offsetNeg.y), 0).r, currViewZ);
    currViewZ = max(texelFetch(usam_gbufferSolidViewZ, texelPos + ivec2(0, offsetPos.y), 0).r, currViewZ);
    currViewZ = max(texelFetch(usam_gbufferSolidViewZ, texelPos + ivec2(offsetNeg.x, offsetNeg.y), 0).r, currViewZ);
    currViewZ = max(texelFetch(usam_gbufferSolidViewZ, texelPos + ivec2(offsetPos.x, offsetNeg.y), 0).r, currViewZ);
    currViewZ = max(texelFetch(usam_gbufferSolidViewZ, texelPos + ivec2(offsetNeg.x, offsetPos.y), 0).r, currViewZ);
    currViewZ = max(texelFetch(usam_gbufferSolidViewZ, texelPos + ivec2(offsetPos.x, offsetPos.y), 0).r, currViewZ);

    GBufferData gData = gbufferData_init();
    gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelPos, 0), gData);
    vec3 currViewPos = coords_toViewCoord(screenPos, currViewZ, global_camProjInverse);
    vec4 curr2PrevViewPos = coord_viewCurrToPrev(vec4(currViewPos, 1.0), gData.isHand);
    vec4 curr2PrevClipPos = global_prevCamProj * curr2PrevViewPos;
    uint clipFlag = uint(curr2PrevClipPos.z > 0.0);
    clipFlag &= uint(all(lessThan(abs(curr2PrevClipPos.xy), curr2PrevClipPos.ww)));
    curr2PrevClipPos /= curr2PrevClipPos.w;
    vec2 prevScreenPos = curr2PrevClipPos.xy * 0.5 + 0.5;

    float lastFrameAccum = 0.0;
    vec3 prevColor = vec3(0.0);
    if (bool(clipFlag)) {
        vec2 prevTexelPos = prevScreenPos * uval_mainImageSize;
        vec4 prevResult = vec4(0.0);
        {
            #if SETTING_TAA_HISTORY_FILTER == 0
            prevResult = history_taa_sample(prevScreenPos);
            #elif SETTING_TAA_HISTORY_FILTER == 1
            CatmullRomBicubic5TapData tapData5 = sampling_catmullRomBicubic5Tap_init(prevTexelPos, 0.5, uval_mainImageSizeRcp);
            prevResult = sampling_catmullBicubic5Tap_sum(
                history_taa_sample(tapData5.uv1AndWeight.xy),
                history_taa_sample(tapData5.uv2AndWeight.xy),
                history_taa_sample(tapData5.uv3AndWeight.xy),
                history_taa_sample(tapData5.uv4AndWeight.xy),
                history_taa_sample(tapData5.uv5AndWeight.xy),
                tapData5
            );
            #elif SETTING_TAA_HISTORY_FILTER == 2
            CatmullRomBicubic9TapData tapData9 = sampling_catmullRomBicubic9Tap_init(prevTexelPos, uval_mainImageSizeRcp);
            prevResult = sampling_catmullRomBicubic9Tap_sum(
                history_taa_sample(tapData9.uv00),
                history_taa_sample(tapData9.uv12_0),
                history_taa_sample(tapData9.uv30),
                history_taa_sample(tapData9.uv01_2),
                history_taa_sample(tapData9.uv12_12),
                history_taa_sample(tapData9.uv31_2),
                history_taa_sample(tapData9.uv03),
                history_taa_sample(tapData9.uv12_3),
                history_taa_sample(tapData9.uv33),
                tapData9
            );
            #else
            vec2 centerPixel = prevTexelPos - 0.5;
            vec2 centerPixelOrigin = floor(centerPixel);
            vec2 pixelPosFract = centerPixel - centerPixelOrigin;

            #if SETTING_TAA_HISTORY_FILTER == 3
            vec4 weightX = sampling_catmullRomWeights(pixelPosFract.x);
            vec4 weightY = sampling_catmullRomWeights(pixelPosFract.y);
            #elif SETTING_TAA_HISTORY_FILTER == 4
            vec4 weightX = sampling_lanczoc2Weights(pixelPosFract.x);
            vec4 weightY = sampling_lanczoc2Weights(pixelPosFract.y);
            #endif

            ivec2 gatherTexelPos = ivec2(centerPixelOrigin) + ivec2(1);
            float weightSum = 0.0;
            for (int iy = 0; iy < 4; ++iy) {
                for (int ix = 0; ix < 4; ++ix) {
                    ivec2 offset = ivec2(ix, iy) - 2;
                    vec4 sampleData = history_taa_fetch(gatherTexelPos + offset);
                    float weight = weightX[ix] * weightY[iy];
                    weightSum += weight;
                    prevResult += sampleData * weight;
                }
            }
            prevResult /= weightSum;
            #endif
        }
        prevColor = saturate(prevResult.rgb);
        lastFrameAccum = prevResult.a;
    }
    float newFrameAccum = lastFrameAccum + 1.0;

    vec3 currColor;
    {
        vec2 centerPixel = unjitterTexelPos - 0.5;
        vec2 centerPixelOrigin = floor(centerPixel);
        vec2 pixelPosFract = centerPixel - centerPixelOrigin;

        vec4 weightX = shared_weightsX;
        vec4 weightY = shared_weightsY;

        ivec2 gatherTexelPos = ivec2(centerPixelOrigin) + ivec2(1);
        ivec2 localOrigin = gatherTexelPos - ivec2(workGroupOrigin);

        vec3 colorResult = vec3(0.0);
        float weightSum = 0.0;
        for (int iy = 0; iy < 4; ++iy) {
            for (int ix = 0; ix < 4; ++ix) {
                ivec2 offset = ivec2(ix, iy) - 2;
                ivec2 localPos = localOrigin + offset + 2; // +2 for padding
                vec3 sampleColor = shared_colorData[localPos.y][localPos.x];
                float weight = weightX[ix] * weightY[iy];
                weightSum += weight;
                colorResult += sampleColor * weight;
            }
        }
        currColor = colors_YCoCgToRGB(colorResult / weightSum);
    }
    currColor = saturate(currColor);

    vec4 taaResetFactor = global_taaResetFactor;
    newFrameAccum *= taaResetFactor.z;

    {
        vec3 currColorYCoCg = colors_RGBToYCoCg(currColor);
        const float distanceFactor = 0.01;
        float kernelParam = -taaResetFactor.x * rcp(1.0 - (currViewPos.z * distanceFactor));
        ColorAABB box = initAABB(currColorYCoCg, exp(kernelParam * shared_kernelDist2[4]));

        ivec2 localTexelPos = texelPos - ivec2(workGroupOrigin) + 2; // +2 for padding

        updateAABB(shared_colorData[localTexelPos.y - 1][localTexelPos.x - 1], exp(kernelParam * shared_kernelDist2[0]), box);
        updateAABB(shared_colorData[localTexelPos.y - 1][localTexelPos.x], exp(kernelParam * shared_kernelDist2[1]), box);
        updateAABB(shared_colorData[localTexelPos.y - 1][localTexelPos.x + 1], exp(kernelParam * shared_kernelDist2[2]), box);

        updateAABB(shared_colorData[localTexelPos.y][localTexelPos.x - 1], exp(kernelParam * shared_kernelDist2[3]), box);
        updateAABB(shared_colorData[localTexelPos.y][localTexelPos.x + 1], exp(kernelParam * shared_kernelDist2[5]), box);

        updateAABB(shared_colorData[localTexelPos.y + 1][localTexelPos.x - 1], exp(kernelParam * shared_kernelDist2[6]), box);
        updateAABB(shared_colorData[localTexelPos.y + 1][localTexelPos.x], exp(kernelParam * shared_kernelDist2[7]), box);
        updateAABB(shared_colorData[localTexelPos.y + 1][localTexelPos.x + 1], exp(kernelParam * shared_kernelDist2[8]), box);

        vec3 mean = box.moment1 / box.weightSum;
        vec3 mean2 = box.moment2 / box.weightSum;
        vec3 variance = mean2 - mean * mean;
        vec3 stddev = sqrt(abs(variance));

        const float varianceAABBSize = 1.0;
        vec3 varianceAABBMin = mean - stddev * varianceAABBSize;
        vec3 varianceAABBMax = mean + stddev * varianceAABBSize;
        varianceAABBMin = clamp(varianceAABBMin, box.minVal, currColorYCoCg);
        varianceAABBMax = clamp(varianceAABBMax, currColorYCoCg, box.maxVal);

        vec3 prevColorYCoCg = colors_RGBToYCoCg(prevColor);

        const float clippingEps = FLT_MIN;
        vec3 delta = prevColorYCoCg - mean;
        delta /= max(1.0, length(delta / stddev));

        vec3 prevColorYCoCgAABBClamped = clamp(prevColorYCoCg, box.minVal, box.maxVal);
        vec3 prevColorYCoCgVarianceAABBClamped = clamp(prevColorYCoCgAABBClamped, varianceAABBMin, varianceAABBMax);
        vec3 prevColorYCoCgEllipsoid = clamp(mean + delta, box.minVal, box.maxVal);
        prevColorYCoCgEllipsoid = clamp(prevColorYCoCgEllipsoid, varianceAABBMin, varianceAABBMax);

        float clampMethod = taaResetFactor.y;

        vec3 prevColorYCoCgClamped = mix(prevColorYCoCgEllipsoid, prevColorYCoCgVarianceAABBClamped, linearStep(0.0, 0.5, clampMethod));
        prevColorYCoCgClamped = mix(prevColorYCoCgClamped, prevColorYCoCgAABBClamped, linearStep(0.5, 1.0, clampMethod));

        prevColor = mix(prevColor, colors_YCoCgToRGB(prevColorYCoCgClamped), 1.0);
    }

    #ifdef SETTING_SCREENSHOT_MODE
    float MIN_ACCUM_FRAMES = 1.0;
    float MAX_ACCUM_FRAMES = 1024.0;
    #else
    float MIN_ACCUM_FRAMES = 1.0;
    float MAX_ACCUM_FRAMES = mix(2.0, 128.0, pow3(global_motionFactor.w));
    if (gData.isHand) {
        MAX_ACCUM_FRAMES *= 0.5;
    }
    #endif

    newFrameAccum = clamp(newFrameAccum, MIN_ACCUM_FRAMES, MAX_ACCUM_FRAMES);

    float finalCurrWeight = 1.0 / newFrameAccum;
    #ifndef SETTING_TAA
    finalCurrWeight = 1.0;
    #endif

    vec3 finalColor = mix(prevColor, currColor, finalCurrWeight);
    vec4 outputData = vec4(finalColor, newFrameAccum);

    float ditherNoise = rand_stbnVec1(rand_newStbnPos(texelPos, 0u), frameCounter);
    outputData = dither_fp16(outputData, ditherNoise);
    transient_taaOutput_store(texelPos, outputData);
}