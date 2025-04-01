#include "Common.glsl"
#include "/util/Colors.glsl"

void gi_update(vec3 currColor, vec3 prevColor, vec2 prevMoments, float prevHLen, out float newHLen, out vec2 newMoments, out vec4 filterInput) {
    vec2 currMoments;
    currMoments.r = min(colors_srgbLuma(currColor), 256.0);
    currMoments.g = currMoments.r * currMoments.r;

    if (prevHLen == 0.0) {
        newHLen = 1.0;
        newMoments = currMoments;
        filterInput.rgb = currColor;
    } else {
        newHLen = min(prevHLen + 1.0, 1024.0);
        float alpha = 1.0 / pow(min(newHLen, SETTING_DENOISER_MAX_ACCUM), SETTING_DENOISER_ACCUM_DECAY);
        filterInput.rgb = mix(prevColor, currColor, alpha);

        vec2 blurredMoments;
        blurredMoments.r = min(colors_srgbLuma(filterInput.rgb), 256.0);
        blurredMoments.g = blurredMoments.r * blurredMoments.r;
        currMoments = mix(currMoments, blurredMoments, 0.0);

        newMoments = mix(prevMoments, currMoments, alpha);
    }

    float variance = max(newMoments.g - newMoments.r * newMoments.r, 0.0);
    filterInput.a = variance;
}