#include "Common.glsl"

void svgf_update(vec3 currColor, vec4 prevColorHLen, vec2 prevMoments, out vec4 newColorHLen, out vec2 newMoments, out vec4 filterInput) {
    vec2 currMoments;
    currMoments.r = colors_srgbLuma(currColor);
    currMoments.g = currMoments.r * currMoments.r;

    if (prevColorHLen.a == 0.0) {
        newColorHLen = vec4(currColor, 1.0);
        newMoments = currMoments;
    } else {
        newColorHLen.a = min(prevColorHLen.a + 1.0, 32.0);
        float alpha = 1.0 / prevColorHLen.a;
        newColorHLen.rgb = mix(prevColorHLen.rgb, currColor, alpha);
        newMoments = mix(prevMoments, currMoments, alpha);
    }

    float variance = max(newMoments.g - newMoments.r * newMoments.r, 0.0);

    filterInput.rgb = newColorHLen.rgb;
    filterInput.a = variance;
}