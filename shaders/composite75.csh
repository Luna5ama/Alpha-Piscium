#version 460 compatibility

#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "/util/Coords.glsl"
#if SETTING_DEBUG_OUTPUT == 1 || SETTING_DEBUG_OUTPUT == 2
#include "/general/DebugOutput.glsl"
#endif
#include "/post/ToneMapping.glsl"
#include "/util/FullScreenComp.glsl"

#if SETTING_DEBUG_TEMP_TEX != 6
#endif

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
        vec3 hdrColor = outputColor.rgb;

        #if SETTING_DEBUG_OUTPUT == 1
        debugOutput(outputColor);
        #endif

        #ifdef SETTING_PURKINJE_EFFECT
        // https://www.desmos.com/calculator/dvpjm8jrmx
        const vec3 ROD_RESPONSE = vec3(0.05, 0.562, 0.604);
        const vec3 SCOPTIC_BASE_COLOR = vec3(SETTING_PURKINJE_EFFECT_CR, SETTING_PURKINJE_EFFECT_CG, SETTING_PURKINJE_EFFECT_CB);
        const float EPSILON = 0.00000000001;

        float luminance = colors_sRGB_luma(hdrColor);
        float rodLuminance = dot(hdrColor, ROD_RESPONSE);
        vec3 scopticColor = SCOPTIC_BASE_COLOR * rodLuminance;
        float scopticLuminance = colors_sRGB_luma(scopticColor);
        float mesopicFactor = log2(luminance * 1000.0);
        mesopicFactor = linearStep(SETTING_PURKINJE_EFFECT_MIN_LUM, SETTING_PURKINJE_EFFECT_MAX_LUM, mesopicFactor);
        scopticColor *= luminance / max(scopticLuminance, EPSILON);
        outputColor.rgb = mix(scopticColor, outputColor.rgb, saturate(mesopicFactor + float(scopticLuminance <= EPSILON)));
        #endif

        vec2 screenPos = coords_texelToUV(texelPos, global_mainImageSizeRcp);
        vec2 ndcPos = screenPos * 2.0 - 1.0;
        float centerFactor = pow(saturate(1.0 - length(ndcPos)), SETTING_EXPOSURE_CENTER_WEIGHTING_CURVE);
        outputColor.a *= 1.0 + centerFactor * SETTING_EXPOSURE_CENTER_WEIGHTING;
        outputColor.a = abs(outputColor.a);
        toneMapping_apply(outputColor);

        vec4 basicColor = texelFetch(usam_temp6, texelPos, 0);
        outputColor.rgb = mix(outputColor.rgb, basicColor.rgb, basicColor.a);

        #if SETTING_DEBUG_OUTPUT == 2
        debugOutput(outputColor);
        #endif

        imageStore(uimg_main, texelPos, outputColor);
    }
}