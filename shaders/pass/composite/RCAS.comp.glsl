#include "/techniques/ffx/fsr1/RCAS.glsl"
#include "/util/AgxInvertible.glsl"
#include "/techniques/debug/DebugOutput.glsl"

const vec2 workGroupsRender = vec2(RENDER_SCALE_FACTOR, RENDER_SCALE_FACTOR);

layout(rgba16f) uniform restrict writeonly image2D uimg_main;

vec4 rcas_loadInput(ivec2 texelPos, bool center) {
    return transient_fxaaOutput_fetch(texelPos);
}

layout(local_size_x = 16, local_size_y = 16) in;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 color = fsr1_rcas(texelPos);
        #if SETTING_DEBUG_OUTPUT == 2
        debugOutput(texelPos, color);
        #endif
        color.a = 1.0;
        color.rgb = agxInvertible_inverse(color.rgb);
        imageStore(uimg_main, texelPos, color);
    }
}