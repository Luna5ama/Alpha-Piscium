#define GLOBAL_DATA_MODIFIER restrict buffer
#include "/Base.glsl"

#define SETTING_RCAS_SHARPNESS SETTING_AA_SHARPNESS

#include "/techniques/ffx/fsr1/RCAS.glsl"

#if SETTING_AA_MODE == 2
#define FSR3_BIND_RCAS
#include "/techniques/ffx/fsr3upscaler/Integration.glsl"
#endif

#include "/util/AgxInvertible.glsl"
#include "/techniques/debug/DebugOutput.glsl"

const vec2 workGroupsRender = vec2(POST_PROCESS_SCALE_FACTOR, POST_PROCESS_SCALE_FACTOR);

layout(rgba16f) uniform restrict writeonly image2D uimg_main;

vec4 rcas_loadInput(ivec2 texelPos, bool center) {
    #if SETTING_AA_MODE == 2
    vec4 color = LoadRCAS_Input(texelPos);
    color.rgb *= exp2(global_aeData.expValues.z);
    color.rgb = agxInvertible_forward(color.rgb);
    return color;
    #else
    return transient_fxaaOutput_fetch(texelPos);
    #endif
}

layout(local_size_x = 16, local_size_y = 16) in;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, POST_PROCESS_IMAGE_SIZE_I))) {
        vec4 color = fsr1_rcas(texelPos);
        #if SETTING_DEBUG_OUTPUT == 2
        debugOutput(renderScale_postToMainTexel(texelPos), color);
        #endif
        color.rgb = agxInvertible_inverse(color.rgb);
        color.a = 1.0;
        imageStore(uimg_main, texelPos, color);
    }
    #if SETTING_AA_MODE == 2
    if (all(equal(gl_GlobalInvocationID, uvec3(0)))) {
        global_fsr3FrameInfo.w = float(frameCounter);
    }
    #endif
}
