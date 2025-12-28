#version 460 compatibility

#include "/techniques/ffx/fsr1/RCAS.glsl"
#include "/util/AgxInvertible.glsl"
#include "/techniques/debug/DebugOutput.glsl"

const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict writeonly image2D uimg_main;
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;

vec4 rcas_loadInput(ivec2 texelPos) {
    return texelFetch(usam_temp1, clamp(texelPos, ivec2(0), uval_mainImageSizeI - ivec2(1)), 0);
}

void rcas_storeOutput(ivec2 texelPos, vec4 color) {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        #if SETTING_DEBUG_OUTPUT == 2
        debugOutput(texelPos, color);
        #endif
        color.a = 1.0;
        imageStore(uimg_main, texelPos, color);
    }
}

float rcas_sharpness() {
    return SETTING_TAA_CAS_SHARPNESS;
}