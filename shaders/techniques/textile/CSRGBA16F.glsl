#ifndef INCLUDE_textile_CSRGBA16F_glsl
#define INCLUDE_textile_CSRGBA16F_glsl a

#include "Common.glsl"

#define _CSRGBA16F_TEXTURE_SIZE (global_mainImageSizeI * ivec2(2, 2))
#define _CSRGBA16F_TEXTURE_SIZE_F (global_mainImageSize * vec2(2.0, 2.0))
#define _CSRGBA16F_TEXTURE_SIZE_RCP rcp(_CSRGBA16F_TEXTURE_SIZE_F)

#define _CSRGBA16F_TEMP1_OFFSET ivec2(0)
#define _CSRGBA16F_TEMP1_OFFSET_F vec2(0.0)
#define _CSRGBA16F_TEMP1_SIZE global_mainImageSizeI
#define _CSRGBA16F_TEMP1_SIZE_F global_mainImageSize

vec2 csrgba16f_temp1_texelToGatherUV(vec2 texelPos) {
    return textile_texelToGatherUV(
        texelPos,
        _CSRGBA16F_TEMP1_OFFSET_F,
        _CSRGBA16F_TEMP1_SIZE_F,
        _CSRGBA16F_TEXTURE_SIZE_RCP
    );
}

ivec2 csrgba16f_temp1_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRGBA16F_TEMP1_OFFSET,
        _CSRGBA16F_TEMP1_SIZE
    );
}

vec2 csrgba16f_temp1_texelToUV(ivec2 texelPos) {
    return textile_texelToUV(
        texelPos,
        _CSRGBA16F_TEMP1_OFFSET,
        _CSRGBA16F_TEMP1_SIZE,
        _CSRGBA16F_TEXTURE_SIZE_RCP
    );
}

vec2 csrgba16f_temp1_texelToUV(vec2 texelPos) {
    return textile_texelToUV(
        texelPos,
        _CSRGBA16F_TEMP1_OFFSET_F,
        _CSRGBA16F_TEMP1_SIZE_F,
        _CSRGBA16F_TEXTURE_SIZE_RCP
    );
}


#define _CSRGBA16F_TEMP2_OFFSET ivec2(0, global_mainImageSizeI.y)
#define _CSRGBA16F_TEMP2_OFFSET_F vec2(0.0, global_mainImageSize.y)
#define _CSRGBA16F_TEMP2_SIZE global_mainImageSizeI
#define _CSRGBA16F_TEMP2_SIZE_F global_mainImageSize

ivec2 csrgba16f_temp2_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRGBA16F_TEMP2_OFFSET,
        _CSRGBA16F_TEMP2_SIZE
    );
}

#define _CSRGBA16F_TEMP3_OFFSET ivec2(global_mainImageSizeI.x, 0)
#define _CSRGBA16F_TEMP3_OFFSET_F vec2(global_mainImageSize.x, 0.0)
#define _CSRGBA16F_TEMP3_SIZE global_mainImageSizeI
#define _CSRGBA16F_TEMP3_SIZE_F global_mainImageSize

ivec2 csrgba16f_temp3_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRGBA16F_TEMP3_OFFSET,
        _CSRGBA16F_TEMP3_SIZE
    );
}

vec2 csrgba16f_temp3_uvToUV(vec2 uv) {
    return textile_uvToUV(
        uv,
        _CSRGBA16F_TEMP3_OFFSET_F,
        _CSRGBA16F_TEMP3_SIZE_F,
        _CSRGBA16F_TEXTURE_SIZE_RCP
    );
}

#define _CSRGBA16F_TEMP4_OFFSET ivec2(global_mainImageSizeI.xy)
#define _CSRGBA16F_TEMP4_OFFSET_F vec2global_mainImageSize.xy)
#define _CSRGBA16F_TEMP4_SIZE global_mainImageSizeI
#define _CSRGBA16F_TEMP4_SIZE_F global_mainImageSize

ivec2 csrgba16f_temp4_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRGBA16F_TEMP4_OFFSET,
        _CSRGBA16F_TEMP4_SIZE
    );
}

#endif