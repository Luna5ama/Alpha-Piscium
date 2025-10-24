#version 460 compatibility

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "/techniques/textile/CSRGBA16F.glsl"
#include "/util/FullScreenComp.glsl"
#if SETTING_DEBUG_OUTPUT == 3
#include "/techniques/DebugOutput.glsl"
#endif

layout(rgba16f) restrict uniform writeonly image2D uimg_csrgba16f;

#define FFXCAS_SHARPENESS SETTING_TAA_CAS_SHARPNESS
#include "/techniques/ffx/FFXCas.glsl"

vec3 ffxcas_load(ivec2 texelPos) {
    return texelFetch(usam_main, texelPos, 0).rgb;
}

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_main, texelPos, 0);
//        outputColor.rgb = ffxcas_pass(texelPos);
        #if SETTING_DEBUG_OUTPUT == 3
        debugOutput(texelPos, outputColor);
        #endif
        #ifdef SETTING_DOF_SHOW_FOCUS_PLANE
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        float alpha = float(viewZ < -global_focusDistance);
        outputColor.rgb = mix(outputColor.rgb, vec3(1.0, 0.0, 1.0), alpha * 0.25);
        #endif
        imageStore(uimg_csrgba16f, csrgba16f_temp1_texelToTexel(texelPos), outputColor);
    }
}