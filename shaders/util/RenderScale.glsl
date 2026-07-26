#ifndef INCLUDE_util_RenderScale_glsl
#define INCLUDE_util_RenderScale_glsl a

#if SETTING_RENDER_SCALE == 0
#define RENDER_SCALE_FACTOR 0.5
#define RENDER_SCALE_HALF 0.25
#define RENDER_SCALE_QUARTER 0.125
#define RENDER_SCALE_TRIPLE 1.5
#define RENDER_SCALE_SIXTEENTH 0.03125
#elif SETTING_RENDER_SCALE == 1
#define RENDER_SCALE_FACTOR 0.55
#define RENDER_SCALE_HALF 0.275
#define RENDER_SCALE_QUARTER 0.1375
#define RENDER_SCALE_TRIPLE 1.65
#define RENDER_SCALE_SIXTEENTH 0.034375
#elif SETTING_RENDER_SCALE == 2
#define RENDER_SCALE_FACTOR 0.6
#define RENDER_SCALE_HALF 0.3
#define RENDER_SCALE_QUARTER 0.15
#define RENDER_SCALE_TRIPLE 1.8
#define RENDER_SCALE_SIXTEENTH 0.0375
#elif SETTING_RENDER_SCALE == 3
#define RENDER_SCALE_FACTOR 0.65
#define RENDER_SCALE_HALF 0.325
#define RENDER_SCALE_QUARTER 0.1625
#define RENDER_SCALE_TRIPLE 1.95
#define RENDER_SCALE_SIXTEENTH 0.040625
#elif SETTING_RENDER_SCALE == 4
#define RENDER_SCALE_FACTOR 0.7
#define RENDER_SCALE_HALF 0.35
#define RENDER_SCALE_QUARTER 0.175
#define RENDER_SCALE_TRIPLE 2.1
#define RENDER_SCALE_SIXTEENTH 0.04375
#elif SETTING_RENDER_SCALE == 5
#define RENDER_SCALE_FACTOR 0.75
#define RENDER_SCALE_HALF 0.375
#define RENDER_SCALE_QUARTER 0.1875
#define RENDER_SCALE_TRIPLE 2.25
#define RENDER_SCALE_SIXTEENTH 0.046875
#elif SETTING_RENDER_SCALE == 6
#define RENDER_SCALE_FACTOR 0.8
#define RENDER_SCALE_HALF 0.4
#define RENDER_SCALE_QUARTER 0.2
#define RENDER_SCALE_TRIPLE 2.4
#define RENDER_SCALE_SIXTEENTH 0.05
#elif SETTING_RENDER_SCALE == 7
#define RENDER_SCALE_FACTOR 0.85
#define RENDER_SCALE_HALF 0.425
#define RENDER_SCALE_QUARTER 0.2125
#define RENDER_SCALE_TRIPLE 2.55
#define RENDER_SCALE_SIXTEENTH 0.053125
#elif SETTING_RENDER_SCALE == 8
#define RENDER_SCALE_FACTOR 0.9
#define RENDER_SCALE_HALF 0.45
#define RENDER_SCALE_QUARTER 0.225
#define RENDER_SCALE_TRIPLE 2.7
#define RENDER_SCALE_SIXTEENTH 0.05625
#elif SETTING_RENDER_SCALE == 9
#define RENDER_SCALE_FACTOR 0.95
#define RENDER_SCALE_HALF 0.475
#define RENDER_SCALE_QUARTER 0.2375
#define RENDER_SCALE_TRIPLE 2.85
#define RENDER_SCALE_SIXTEENTH 0.059375
#else
#define RENDER_SCALE_FACTOR 1.0
#define RENDER_SCALE_HALF 0.5
#define RENDER_SCALE_QUARTER 0.25
#define RENDER_SCALE_TRIPLE 3.0
#define RENDER_SCALE_SIXTEENTH 0.0625
#endif

#ifdef SETTING_FSR3
#define POST_PROCESS_SCALE_FACTOR 1.0
#define POST_PROCESS_SCALE_HALF 0.5
#define POST_PROCESS_SCALE_QUARTER 0.25
#define POST_PROCESS_IMAGE_SIZE uval_viewImageSize
#define POST_PROCESS_IMAGE_SIZE_I ivec2(uval_viewImageSize)
#define POST_PROCESS_IMAGE_SIZE_RCP uval_viewImageSizeRcp
#else
#define POST_PROCESS_SCALE_FACTOR RENDER_SCALE_FACTOR
#define POST_PROCESS_SCALE_HALF RENDER_SCALE_HALF
#define POST_PROCESS_SCALE_QUARTER RENDER_SCALE_QUARTER
#define POST_PROCESS_IMAGE_SIZE uval_mainImageSize
#define POST_PROCESS_IMAGE_SIZE_I uval_mainImageSizeI
#define POST_PROCESS_IMAGE_SIZE_RCP uval_mainImageSizeRcp
#endif

#ifndef SKIP_UNIFORMS
ivec2 renderScale_postToMainTexel(ivec2 postTexel) {
#ifdef SETTING_FSR3
    vec2 viewUV = (vec2(postTexel) + 0.5) * uval_viewImageSizeRcp;
    return clamp(ivec2(viewUV * uval_mainImageSize), ivec2(0), uval_mainImageSizeI - 1);
#else
    return postTexel;
#endif
}

void renderScale_applyGBufferScale(inout vec4 position) {
#if SETTING_RENDER_SCALE < 10
    position.xy = position.xy * uval_mainImageScale + (uval_mainImageScale - 1.0) * position.w;
#endif
}

bool renderScale_isOutsideMainViewport(vec2 fragCoord) {
#if SETTING_RENDER_SCALE < 10
    return any(greaterThanEqual(fragCoord, uval_mainImageSize));
#else
    return false;
#endif
}

#if SETTING_RENDER_SCALE < 10
#undef _shadesmith_RGBA16F_ATLAS_SIZE_RCP
#undef _shadesmith_R32F_ATLAS_SIZE_RCP
#undef _shadesmith_RGB10_A2_ATLAS_SIZE_RCP
#undef _shadesmith_RGBA8_ATLAS_SIZE_RCP
#undef _shadesmith_RGBA32UI_ATLAS_SIZE_RCP
#undef _shadesmith_RG32UI_ATLAS_SIZE_RCP
#define _shadesmith_RGBA16F_ATLAS_SIZE_RCP (vec2(1.0) / vec2(textureSize(usam_rgba16f, 0)))
#define _shadesmith_R32F_ATLAS_SIZE_RCP (vec2(1.0) / vec2(textureSize(usam_r32f, 0)))
#define _shadesmith_RGB10_A2_ATLAS_SIZE_RCP (vec2(1.0) / vec2(textureSize(usam_rgb10_a2, 0)))
#define _shadesmith_RGBA8_ATLAS_SIZE_RCP (vec2(1.0) / vec2(textureSize(usam_rgba8, 0)))
#define _shadesmith_RGBA32UI_ATLAS_SIZE_RCP (vec2(1.0) / vec2(textureSize(usam_rgba32ui, 0)))
#define _shadesmith_RG32UI_ATLAS_SIZE_RCP (vec2(1.0) / vec2(textureSize(usam_rg32ui, 0)))
#endif

#endif

#endif
