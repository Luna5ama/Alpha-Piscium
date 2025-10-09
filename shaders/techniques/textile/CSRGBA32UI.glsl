#ifndef INCLUDE_textile_CSRGBA32UI_glsl
#define INCLUDE_textile_CSRGBA32UI_glsl a

#include "Common.glsl"

#define _CSRGBA32UI_TEXTURE_SIZE (uval_mainImageSizeI * ivec2(2, 3))
#define _CSRGBA32UI_TEXTURE_SIZE_F (uval_mainImageSize * vec2(2.0, 3.0))
#define _CSRGBA32UI_TEXTURE_SIZE_RCP rcp(_CSRGBA32UI_TEXTURE_SIZE_F)

#define _CSRGBA32UI_DIFFUSE_HISTORY_OFFSET ivec2(0)
#define _CSRGBA32UI_DIFFUSE_HISTORY_OFFSET_F vec2(0.0)
#define _CSRGBA32UI_DIFFUSE_HISTORY_SIZE uval_mainImageSizeI
#define _CSRGBA32UI_DIFFUSE_HISTORY_SIZE_F uval_mainImageSize

vec2 gi_diffuseHistory_texelToGatherUV(vec2 texelPos) {
    return textile_texelToGatherUV(
        texelPos,
        _CSRGBA32UI_DIFFUSE_HISTORY_OFFSET_F,
        _CSRGBA32UI_DIFFUSE_HISTORY_SIZE_F,
        _CSRGBA32UI_TEXTURE_SIZE_RCP
    );
}

ivec2 gi_diffuseHistory_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRGBA32UI_DIFFUSE_HISTORY_OFFSET,
        _CSRGBA32UI_DIFFUSE_HISTORY_SIZE
    );
}

#define _CSRGBA32UI_CLOUDS_SS_HISTORY_OFFSET ivec2(0, uval_mainImageSizeI.y)
#define _CSRGBA32UI_CLOUDS_SS_HISTORY_OFFSET_F vec2(0.0, uval_mainImageSize.y)
#define _CSRGBA32UI_CLOUDS_SS_HISTORY_SIZE uval_mainImageSizeI
#define _CSRGBA32UI_CLOUDS_SS_HISTORY_SIZE_F uval_mainImageSize

ivec2 clouds_ss_history_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRGBA32UI_CLOUDS_SS_HISTORY_OFFSET,
        _CSRGBA32UI_CLOUDS_SS_HISTORY_SIZE
    );
}

#define _CSRGBA32UI_TEMP1_OFFSET ivec2(uval_mainImageSizeI.x, 0)
#define _CSRGBA32UI_TEMP1_OFFSET_F vec2(uval_mainImageSize.x, 0.0)
#define _CSRGBA32UI_TEMP1_SIZE uval_mainImageSizeI
#define _CSRGBA32UI_TEMP1_SIZE_F uval_mainImageSize

ivec2 csrgba32ui_temp1_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRGBA32UI_TEMP1_OFFSET,
        _CSRGBA32UI_TEMP1_SIZE
    );
}

#define _CSRGBA32UI_TEMP2_OFFSET ivec2(uval_mainImageSizeI.xy)
#define _CSRGBA32UI_TEMP2_OFFSET_F vec2uval_mainImageSize.xy)
#define _CSRGBA32UI_TEMP2_SIZE uval_mainImageSizeI
#define _CSRGBA32UI_TEMP2_SIZE_F uval_mainImageSize

ivec2 csrgba32ui_temp2_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRGBA32UI_TEMP2_OFFSET,
        _CSRGBA32UI_TEMP2_SIZE
    );
}

#define _CSRGBA32UI_TEMP3_OFFSET ivec2(0, uval_mainImageSizeI.y * 2)
#define _CSRGBA32UI_TEMP3_OFFSET_F vec2(0.0, uval_mainImageSize.y * 2.0)
#define _CSRGBA32UI_TEMP3_SIZE uval_mainImageSizeI
#define _CSRGBA32UI_TEMP3_SIZE_F uval_mainImageSize

ivec2 csrgba32ui_temp3_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRGBA32UI_TEMP3_OFFSET,
        _CSRGBA32UI_TEMP3_SIZE
    );
}

#define _CSRGBA32UI_TEMP4_OFFSET ivec2(uval_mainImageSizeI.x, uval_mainImageSizeI.y * 2)
#define _CSRGBA32UI_TEMP4_OFFSET_F vec2(uval_mainImageSize.x, uval_mainImageSize.y * 2.0)
#define _CSRGBA32UI_TEMP4_SIZE uval_mainImageSizeI
#define _CSRGBA32UI_TEMP4_SIZE_F uval_mainImageSize

ivec2 csrgba32ui_temp4_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRGBA32UI_TEMP4_OFFSET,
        _CSRGBA32UI_TEMP4_SIZE
    );
}

#endif