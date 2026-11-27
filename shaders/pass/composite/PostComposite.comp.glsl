#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#define GLOBAL_DATA_MODIFIER buffer

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "/techniques/debug/DebugOutput.glsl"
#include "/techniques/displaytransform/DisplayTransform.glsl"
#include "/util/Colors.glsl"
#include "/util/Coords.glsl"
#include "/util/FullScreenComp.glsl"

layout(rgba16f) restrict uniform image2D uimg_main;

#ifdef SETTING_BLOOM
/*const*/
#define BLOOM_PASS 1
/*const*/
#include "/techniques/Bloom.glsl"
#endif

void main() {
    displaytransform_init();

    bool valid = all(lessThan(texelPos, uval_mainImageSizeI));
    vec4 outputColor = vec4(0.0);

    if (valid) {
        outputColor = texelFetch(usam_main, texelPos, 0);

        #ifdef SETTING_BLOOM
        // The eight levels are averaged. Keep the former default 8/128 blend.
        float bloomIntensity = SETTING_BLOOM_INTENSITY - 4.0;
        if (isEyeInWater == 1) {
            bloomIntensity += SETTING_BLOOM_UNDERWATER_BOOST;
        }
        vec3 glow = bloom_filterPixel(texelPos);
        #if SETTING_BLOOM_SHAPE != 0
        vec2 shapeSize = vec2(global_mipmapTileCeilPadded[1].zw);
        vec2 shapeTexel = clamp((vec2(texelPos) + 0.5) * 0.5, vec2(0.5), shapeSize - 0.5);
        glow += transient_bloomShape_sample(shapeTexel * uval_mainImageSizeRcp).rgb;
        #endif
        outputColor.rgb += glow * exp2(bloomIntensity);
        #endif

        #if SETTING_DEBUG_OUTPUT == 3
        debugOutput(texelPos, outputColor);
        #endif

        #ifdef SETTING_PURKINJE_EFFECT
        vec3 hdrColor = outputColor.rgb;
        // https://www.desmos.com/calculator/dvpjm8jrmx
        const vec3 ROD_RESPONSE = vec3(0.05, 0.562, 0.604);
        const vec3 SCOPTIC_BASE_COLOR = vec3(SETTING_PURKINJE_EFFECT_CR, SETTING_PURKINJE_EFFECT_CG, SETTING_PURKINJE_EFFECT_CB);
        const float EPSILON = 0.00000000001;

        float luminance = colors_colorspaces_luma(COLORS_WORKING_COLORSPACE, hdrColor);
        float rodLuminance = dot(hdrColor, ROD_RESPONSE);
        vec3 scopticColor = SCOPTIC_BASE_COLOR * rodLuminance;
        float scopticLuminance = colors_colorspaces_luma(COLORS_WORKING_COLORSPACE, scopticColor);
        float mesopicFactor = log2(luminance * 1000.0);
        mesopicFactor = linearStep(SETTING_PURKINJE_EFFECT_MIN_LUM, SETTING_PURKINJE_EFFECT_MAX_LUM, mesopicFactor);
        scopticColor *= luminance / max(scopticLuminance, EPSILON);
        outputColor.rgb = mix(scopticColor, outputColor.rgb, saturate(mesopicFactor + float(scopticLuminance <= EPSILON)));
        #endif

        vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
        vec2 ndcPos = screenPos * 2.0 - 1.0;
        float centerFactor = pow(saturate(1.0 - length(ndcPos)), SETTING_EXPOSURE_CENTER_WEIGHTING_CURVE);
        outputColor.a *= 1.0 + centerFactor * SETTING_EXPOSURE_CENTER_WEIGHTING;
        outputColor.a = abs(outputColor.a);
    }

    displaytransform_apply(valid, outputColor);

    if (valid) {
        imageStore(uimg_main, texelPos, outputColor);
    }
}
