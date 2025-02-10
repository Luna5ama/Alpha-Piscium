#version 460 compatibility

#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "util/FullScreenComp.glsl"

layout(rgba16f) restrict uniform image2D uimg_main;

#include "general/DebugOutput.glsl"
#include "post/ToneMapping.glsl"

#define BLOOM_UP_SAMPLE 1
#define BLOOM_PASS 1
#define BLOOM_NON_STANDALONE a
#if SETTING_DEBUG_TEMP_TEX == 3
#define BLOOM_NO_SAMPLER a
#endif
#include "post/Bloom.comp"

void main() {
    toneMapping_init();
    #ifdef SETTING_BLOOM
    bloom_init();
    #endif
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);

        #ifdef SETTING_BLOOM
        outputColor += bloom_main(texelPos);
        #endif

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