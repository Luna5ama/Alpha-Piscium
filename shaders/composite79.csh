#version 460 compatibility

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "/util/FullScreenComp.glsl"
#include "/techniques/debug/DebugOutput.glsl"
#include "/techniques/debug/DebugFinalOutput.glsl"

layout(rgba16f) restrict uniform writeonly image2D uimg_main;

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_main, texelPos, 0);

        vec4 basicColor = texelFetch(usam_overlays, texelPos, 0);
        outputColor.rgb = mix(outputColor.rgb, basicColor.rgb, basicColor.a);

        #ifdef SETTING_DOF_SHOW_FOCUS_PLANE
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        float alpha = float(viewZ < -global_focusDistance);
        outputColor.rgb = mix(outputColor.rgb, vec3(1.0, 0.0, 1.0), alpha * 0.25);
        #endif

        #if SETTING_DEBUG_OUTPUT == 4
        debugOutput(texelPos, outputColor);
        #endif

        debugFinalOutput(texelPos, outputColor);

        imageStore(uimg_main, texelPos, outputColor);
    }
}