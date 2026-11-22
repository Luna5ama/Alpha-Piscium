#define GLOBAL_DATA_MODIFIER buffer
layout(local_size_x = 16, local_size_y = 16) in;
#include "/techniques/debug/DebugOutput.glsl"
#include "/util/Colors.glsl"
#include "/util/AgxInvertible.glsl"

const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_main;
#include "/techniques/DOF.glsl"

#include "/techniques/displaytransform/Exposure.glsl"

void main() {
    meterBins[gl_LocalInvocationIndex] = 0u;
    barrier();
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_main, texelPos, 0);
        #ifdef SETTING_DOF
        outputColor.rgb = dof_sample(texelPos);
        #endif
        meterScene(texelPos, outputColor.rgb);
        #if SETTING_DEBUG_OUTPUT == 1
        debugOutput(texelPos, outputColor);
        #endif
        #ifdef SETTING_TAA
        outputColor.rgb = agxInvertible_forward(outputColor.rgb);
        #endif
        imageStore(uimg_main, texelPos, outputColor);
    }
    barrier();
    uint weight = meterBins[gl_LocalInvocationIndex];
    if (weight != 0u) atomicAdd(global_aeData.histogram[gl_LocalInvocationIndex], weight);
}
