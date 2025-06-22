#version 460 compatibility

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "/util/FullScreenComp.glsl"
#include "/general/DebugOutput.glsl"

layout(rgba16f) restrict uniform image2D uimg_main;

#define FFXCAS_SHARPENESS SETTING_TAA_CAS_SHARPNESS
#include "/post/FFXCas.glsl"

vec3 ffxcas_load(ivec2 texelPos) {
    return imageLoad(uimg_main, texelPos).rgb;
}

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);
        outputColor.rgb = ffxcas_pass(texelPos);
        #if SETTING_DEBUG_OUTPUT == 3
        debugOutput(outputColor);
        #endif
        #ifdef SETTING_DOF_SHOW_FOCUS_PLANE
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        float alpha = float(viewZ < -global_focusDistance);
        outputColor.rgb = mix(outputColor.rgb, vec3(1.0, 0.0, 1.0), alpha * 0.25);
        #endif
        imageStore(uimg_main, texelPos, outputColor);
    }
}