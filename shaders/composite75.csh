#version 460 compatibility

#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "/general/DebugOutput.glsl"
#include "/post/ToneMapping.glsl"
#include "/util/FullScreenComp.glsl"
#include "/rtwsm/ShadowAABB.glsl"

layout(rgba16f) restrict uniform image2D uimg_main;

#define BLOOM_UP_SAMPLE 1
#define BLOOM_PASS 1
#define BLOOM_NON_STANDALONE a
#if SETTING_DEBUG_TEMP_TEX == 3
#define BLOOM_NO_SAMPLER a
#endif
#include "/post/Bloom.comp.glsl"

void main() {
    toneMapping_init();
    #ifdef SETTING_BLOOM
    bloom_init();
    #endif
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);

        #ifdef SETTING_BLOOM
        outputColor += bloom_mainOutput(texelPos);
        #endif

        #if SETTING_DEBUG_OUTPUT == 1
        debugOutput(outputColor);
        #endif

        vec2 screenPos = coords_texelToScreen(texelPos, global_mainImageSizeRcp);
        vec2 ndcPos = screenPos * 2.0 - 1.0;
        float centerFactor = pow(saturate(1.0 - length(ndcPos)), SETTING_EXPOSURE_CENTER_WEIGHTING_CURVE);
        outputColor.a *= 1.0 + centerFactor * SETTING_EXPOSURE_CENTER_WEIGHTING;
        toneMapping_apply(outputColor);

        #if SETTING_DEBUG_OUTPUT == 2
        debugOutput(outputColor);
        #endif

        imageStore(uimg_main, texelPos, outputColor);
    }

    rtwsm_shadowAABB();
}