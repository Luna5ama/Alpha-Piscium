#version 460 compatibility

#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "util/FullScreenComp.glsl"

layout(rgba16f) restrict uniform image2D uimg_main;

#include "general/DebugOutput.glsl"
#include "post/ToneMapping.glsl"

#define BLOOM_UP_SAMPLE 1
#define BLOOM_PASS 1
#define BLOOM_NON_STANDALONE a
#include "post/Bloom.comp"

void main() {
    toneMapping_init();
    bloom_init();
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);
        outputColor += bloom_main(texelPos);

        #if SETTING_DEBUG_OUTPUT == 1
        debugOutput(outputColor);
        #endif

        toneMapping_apply(outputColor);

        #if SETTING_DEBUG_OUTPUT == 2
        debugOutput(outputColor);
        #endif

        imageStore(uimg_main, texelPos, outputColor);
    }
}