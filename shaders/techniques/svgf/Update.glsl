#include "Common.glsl"
#include "/util/Colors.glsl"
#include "/util/Colors2.glsl"
#include "/util/Math.glsl"

const vec2 SVGF_MAX_ACCUM = vec2(SETTING_DENOISER_MAX_ACCUM, SETTING_DENOISER_MAX_FAST_ACCUM);

void svgf_updateHistory(
vec3 currColor, vec3 currFastColor,
vec3 prevColor, vec3 prevFastColor, vec2 prevMoments, float prevHLen,
out vec3 newColor, out vec3 newFastColor, out vec2 newMoments, out float newHLen
) {
    vec2 currMoments;
    currMoments.r = min(colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, currColor), 256.0);
    currMoments.g = currMoments.r * currMoments.r;

    vec2 alpha = vec2(1.0);
    newHLen = 1.0;
    prevHLen *= global_historyResetFactor;

    if (prevHLen != 0.0) {
        newHLen = min(prevHLen + 1.0, 1024.0);
        alpha = rcp(pow(min(vec2(newHLen), SVGF_MAX_ACCUM), vec2(SETTING_DENOISER_ACCUM_DECAY)));
    }
    newColor = mix(prevColor, currColor, alpha.x);
    newFastColor = mix(prevFastColor, currFastColor, alpha.y);
    newMoments = mix(prevMoments, currMoments, alpha.x);
    newMoments.x = min(colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, newColor), 256.0);
}