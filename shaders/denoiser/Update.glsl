#include "Common.glsl"
#include "/util/Colors.glsl"
#include "/util/Math.glsl"

void gi_update(
vec3 currColor,
vec3 prevColor, vec3 prevFastColor, vec2 prevMoments, float prevHLen,
out vec3 newColor, out vec3 newFastColor, out vec2 newMoments, out float newHLen
) {
    vec2 currMoments;
    currMoments.r = min(colors_srgbLuma(currColor), 256.0);
    currMoments.g = currMoments.r * currMoments.r;

    vec2 alpha = vec2(1.0);
    newHLen = 1.0;

    if (prevHLen != 0.0) {
        newHLen = min(prevHLen + 1.0, 1024.0);
        alpha = rcp(pow(min(vec2(newHLen), vec2(SETTING_DENOISER_MAX_ACCUM, SETTING_DENOISER_MAX_FAST_ACCUM)), vec2(SETTING_DENOISER_ACCUM_DECAY)));
    }
    newColor = mix(prevColor, currColor, alpha.x);
    newFastColor = mix(prevFastColor, currColor, alpha.y);
    newMoments = mix(prevMoments, currMoments, alpha.x);
    newMoments.x = min(colors_srgbLuma(newColor), 256.0);
}