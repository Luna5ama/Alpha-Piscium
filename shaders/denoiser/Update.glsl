#include "Common.glsl"
#include "/util/Colors.glsl"

void gi_update(
vec3 currColor,
vec3 prevColor, vec3 prevFastColor, vec2 prevMoments, float prevHLen,
out vec3 newColor, out vec3 newFastColor, out vec2 newMoments, out float newHLen
) {
    vec2 currMoments;
    currMoments.r = min(colors_srgbLuma(currColor), 256.0);
    currMoments.g = currMoments.r * currMoments.r;

    if (prevHLen == 0.0) {
        newColor = currColor;
        newFastColor = currColor;
        newMoments = currMoments;
        newHLen = 1.0;
    } else {
        newHLen = min(prevHLen + 1.0, 1024.0);
        float alpha = 1.0 / pow(min(newHLen, SETTING_DENOISER_MAX_ACCUM), SETTING_DENOISER_ACCUM_DECAY);
        float alphaFast = 1.0 / pow(min(newHLen, SETTING_DENOISER_MAX_FAST_ACCUM), SETTING_DENOISER_ACCUM_DECAY);
        newColor = mix(prevColor, currColor, alpha);
        newFastColor = mix(prevFastColor, currColor, alphaFast);

        vec2 blurredMoments;
        blurredMoments.r = min(colors_srgbLuma(newColor), 256.0);
        blurredMoments.g = blurredMoments.r * blurredMoments.r;
        currMoments = mix(currMoments, blurredMoments, 0.0);

        newMoments = mix(prevMoments, currMoments, alpha);
    }
}