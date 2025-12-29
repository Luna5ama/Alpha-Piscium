#version 460 compatibility

#include "/techniques/debug/DebugOutput.glsl"
#include "/util/Colors.glsl"
#include "/util/AgxInvertible.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_main;
layout(rgba16f) uniform restrict writeonly image2D uimg_rgba16f;
#include "/techniques/DOF.glsl"

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);
        #ifdef SETTING_DOF
        outputColor.rgb = dof_sample(texelPos);
        #endif
        #if SETTING_DEBUG_OUTPUT == 1
        debugOutput(texelPos, outputColor);
        #endif
        outputColor.rgb *= exp2(global_aeData.expValues.z);
        outputColor.rgb = agxInvertible_forward(outputColor.rgb);
        imageStore(uimg_main, texelPos, outputColor);
        transient_bloom_store(texelPos, vec4(0.0));
    }
}