#include "/Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(POST_PROCESS_SCALE_FACTOR, POST_PROCESS_SCALE_FACTOR);

#include "/techniques/debug/DebugOutput.glsl"
#include "/techniques/debug/DebugFinalOutput.glsl"
#include "/util/Colors2.glsl"
#include "/util/Coords.glsl"

layout(rgba16f) restrict uniform image2D uimg_main;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, POST_PROCESS_IMAGE_SIZE_I))) {
        ivec2 mainTexelPos = renderScale_postToMainTexel(texelPos);
        vec4 outputColor = texelFetch(usam_main, texelPos, 0);

        vec4 basicColor = texelFetch(usam_overlays, mainTexelPos, 0);
        outputColor.rgb = mix(outputColor.rgb, basicColor.rgb, basicColor.a);

        #ifdef SETTING_DOF_SHOW_FOCUS_PLANE
        float viewZ = texelFetch(usam_gbufferSolidViewZ, mainTexelPos, 0).r;
        float alpha = float(viewZ < -global_focusDistance);
        outputColor.rgb = mix(outputColor.rgb, vec3(1.0, 0.0, 1.0), alpha * 0.25);
        #endif

        #if SETTING_DEBUG_OUTPUT == 4
        debugOutput(mainTexelPos, outputColor);
        #endif
        #ifdef SETTING_DEBUG_TEXT_OUTPUT
        debugFinalOutput(mainTexelPos, outputColor);
        #endif

        imageStore(uimg_main, texelPos, outputColor);
    }
}
