#include "Common.glsl"
#include "/util/Colors.glsl"

void gi_update(vec3 currColor, vec4 prevColorHLen, out float newHLen, out vec4 filterInput) {
    if (prevColorHLen.a == 0.0) {
        newHLen = 1.0;
        filterInput.rgb = currColor;
    } else {
        newHLen = min(prevColorHLen.a + 1.0, SETTING_DENOISER_MAX_ACCUM);
        float alpha = 1.0 / pow(newHLen, SETTING_DENOISER_ACCUM_DECAY);
        filterInput.rgb = mix(prevColorHLen.rgb, currColor, alpha);
    }

    filterInput.a = 0.0;
}