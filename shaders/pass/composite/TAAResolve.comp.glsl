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

void updateAABB(vec3 colorSRGB, float weight, inout ColorAABB box) {
    vec3 colorYCoCg = colors_SRGBToYCoCg(colorSRGB);
    box.minVal = mix(box.minVal, min(box.minVal, colorYCoCg), weight);
    box.maxVal = mix(box.maxVal, max(box.maxVal, colorYCoCg), weight);
    box.moment1 += colorYCoCg * weight;
    box.moment2 += colorYCoCg * colorYCoCg * weight;
    box.weightSum += weight;
}

float kernelWeight(vec2 centerPos, vec2 samplePos, float param) {
    vec2 diff = abs(samplePos - centerPos);
    float dist2 = dot(diff, diff);
    return exp(param * dist2);
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    vec2 texelCenter = vec2(texelPos) + vec2(0.5);
    vec2 screenPos = texelCenter * uval_mainImageSizeRcp;

    GBufferData gData = gbufferData_init();
    gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

    vec2 unjitterTexelPos = texelCenter + global_taaJitter;
    vec3 currColor = sampling_catmullBicubic5Tap(usam_main, unjitterTexelPos, 0.5, uval_mainImageSizeRcp).rgb;
    currColor = saturate(currColor);

    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    vec3 currViewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec4 prevViewPos = coord_viewCurrToPrev(vec4(currViewPos, 1.0), gData.isHand);
    vec4 prevClipPos = global_prevCamProj * prevViewPos;
    prevClipPos /= prevClipPos.w;
    vec2 prevScreenPos = prevClipPos.xy * 0.5 + 0.5;
    vec2 prevTexelPos = prevScreenPos * uval_mainImageSize;

    CatmullBicubic5TapData tapData = sampling_catmullBicubic5Tap_init(prevTexelPos, 0.5, uval_mainImageSizeRcp);
    vec4 prevResult = sampling_catmullBicubic5Tap_sum(
        history_taa_sample(tapData.uv1AndWeight.xy),
        history_taa_sample(tapData.uv2AndWeight.xy),
        history_taa_sample(tapData.uv3AndWeight.xy),
        history_taa_sample(tapData.uv4AndWeight.xy),
        history_taa_sample(tapData.uv5AndWeight.xy),
        tapData
    );
    vec3 prevColor = saturate(prevResult.rgb);

    float lastFrameAccum = prevResult.a;
    float newFrameAccum = lastFrameAccum + 1.0;

    vec2 pixelPosDiff = (screenPos - prevScreenPos) * uval_mainImageSize;
    vec3 cameraDelta = uval_cameraDelta;
    float cameraSpeed = length(cameraDelta);
    float prevCameraSpeed = length(global_prevCameraDelta);
    float cameraSpeedDiff = abs(cameraSpeed - prevCameraSpeed);
    float pixelSpeed = length(pixelPosDiff);

    float speedSum = 0.0;
    speedSum += sqrt(cameraSpeedDiff) * 8.0;
    speedSum += sqrt(cameraSpeed) * 0.05;
    speedSum += sqrt(pixelSpeed) * 0.2;

    vec3 prevFrontVec = coords_dir_viewToWorldPrev(vec3(0.0, 0.0, -1.0));
    vec3 currFrontVec = coords_dir_viewToWorld(vec3(0.0, 0.0, -1.0));
    float frontVecDiff = dot(prevFrontVec, currFrontVec);

    float extraReset = global_taaResetFactor;
    extraReset *= (1.0 - saturate(pixelSpeed * 1.0));

    #ifdef SETTING_SCREENSHOT_MODE
    #if SETTING_SCREENSHOT_MODE_SKIP_INITIAL
    extraReset *= float(frameCounter > SETTING_SCREENSHOT_MODE_SKIP_INITIAL);
    #endif
    #else
    extraReset *= (1.0 - saturate(cameraSpeedDiff * 64.0));
    extraReset *= (1.0 - saturate(cameraSpeed * 1.0));
    #endif

    {
        vec3 currColorYCoCg = colors_SRGBToYCoCg(currColor);
        float clampWeight = exp2(-speedSum);
        float param = pow2(saturate(1.0 - clampWeight)) * -2.0;

        ColorAABB box = initAABB(currColorYCoCg, kernelWeight(texelPos, unjitterTexelPos, param));

        updateAABB(textureOffset(usam_main, screenPos, ivec2(-1, 0)).rgb, kernelWeight(texelPos, unjitterTexelPos + ivec2(-1.0, 0.0), param), box);
        updateAABB(textureOffset(usam_main, screenPos, ivec2(1, 0)).rgb, kernelWeight(texelPos, unjitterTexelPos + ivec2(1.0, 0.0), param), box);
        updateAABB(textureOffset(usam_main, screenPos, ivec2(0, -1)).rgb, kernelWeight(texelPos, unjitterTexelPos + ivec2(0.0, -1.0), param), box);
        updateAABB(textureOffset(usam_main, screenPos, ivec2(0, 1)).rgb, kernelWeight(texelPos, unjitterTexelPos + ivec2(0.0, 1.0), param), box);

        updateAABB(textureOffset(usam_main, screenPos, ivec2(-1, -1)).rgb, kernelWeight(texelPos, unjitterTexelPos + ivec2(-1.0, -1.0), param), box);
        updateAABB(textureOffset(usam_main, screenPos, ivec2(1, -1)).rgb, kernelWeight(texelPos, unjitterTexelPos + ivec2(1.0, -1.0), param), box);
        updateAABB(textureOffset(usam_main, screenPos, ivec2(-1, 1)).rgb, kernelWeight(texelPos, unjitterTexelPos + ivec2(-1.0, 1.0), param), box);
        updateAABB(textureOffset(usam_main, screenPos, ivec2(1, 1)).rgb, kernelWeight(texelPos, unjitterTexelPos + ivec2(1.0, 1.0), param), box);

        vec3 mean = box.moment1 / box.weightSum;
        vec3 mean2 = box.moment2 / box.weightSum;
        vec3 variance = mean2 - mean * mean;
        vec3 stddev = sqrt(abs(variance));

        vec3 prevColorYCoCg = colors_SRGBToYCoCg(prevColor);
        vec3 varianceAABBMin = mean - stddev * 1.0;
        vec3 varianceAABBMax = mean + stddev * 1.0;
        varianceAABBMin = clamp(varianceAABBMin, box.minVal, currColorYCoCg);
        varianceAABBMax = clamp(varianceAABBMax, currColorYCoCg, box.maxVal);

        const float clippingEps = FLT_MIN;
        vec3 delta = prevColorYCoCg - mean;
        delta /= max(1.0, length(delta / stddev));

        vec3 prevColorYCoCgAABBClamped = clamp(prevColorYCoCg, box.minVal, box.maxVal);
        vec3 prevColorYCoCgVarianceAABBClamped = clamp(prevColorYCoCgAABBClamped, varianceAABBMin, varianceAABBMax);
        vec3 prevColorYCoCgEllipsoid = clamp(mean + delta, box.minVal, box.maxVal);
        prevColorYCoCgEllipsoid = clamp(prevColorYCoCgEllipsoid, varianceAABBMin, varianceAABBMax);

        #ifdef SETTING_SCREENSHOT_MODE
        clampWeight *= extraReset;
        #endif
        clampWeight = pow3(clampWeight);

        vec3 prevColorYCoCgClamped = mix(prevColorYCoCgEllipsoid, prevColorYCoCgVarianceAABBClamped, linearStep(0.0, 0.5, clampWeight));
        prevColorYCoCgClamped = mix(prevColorYCoCgClamped, prevColorYCoCgAABBClamped, linearStep(0.5, 1.0, clampWeight));

        #ifdef SETTING_SCREENSHOT_MODE
        prevColor = colors_YCoCgToSRGB(mix(prevColorYCoCgClamped, prevColorYCoCg, extraReset));
        #else
        prevColor = colors_YCoCgToSRGB(prevColorYCoCgClamped);
        #endif
    }

    const float FRAME_RESET_FACTOR = 4.0;
    float frameReset = FRAME_RESET_FACTOR / (FRAME_RESET_FACTOR + speedSum);
    newFrameAccum *= frameReset;
    #ifdef SETTING_SCREENSHOT_MODE
    float MIN_ACCUM_FRAMES = 1.0;
    float MAX_ACCUM_FRAMES = 1024.0;
    #else
    float MIN_ACCUM_FRAMES = 2.0;
    float MAX_ACCUM_FRAMES = 100.0;
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

    float ditherNoise = rand_stbnVec1(texelPos, frameCounter + 3);
    outputData = dither_fp16(outputData, ditherNoise);
    transient_taaOutput_store(texelPos, outputData);
}