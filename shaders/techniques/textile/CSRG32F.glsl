#ifndef INCLUDE_textile_CSRG32F_glsl
#define INCLUDE_textile_CSRG32F_glsl a

#include "Common.glsl"

#define _CSRG32F_TEXTURE_SIZE (uval_mainImageSizeI * ivec2(1, 3))
#define _CSRG32F_TEXTURE_SIZE_F (uval_mainImageSize * vec2(1.0, 3.0))
#define _CSRG32F_TEXTURE_SIZE_RCP rcp(_CSRG32F_TEXTURE_SIZE_F)

#define _CSRG32F_TILE1_OFFSET ivec2(0)
#define _CSRG32F_TILE1_OFFSET_F vec2(0.0)
#define _CSRG32F_TILE1_SIZE uval_mainImageSizeI
#define _CSRG32F_TILE1_SIZE_F uval_mainImageSize

ivec2 csrg32f_tile1_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRG32F_TILE1_OFFSET,
        _CSRG32F_TILE1_SIZE
    );
}

vec2 csrg32f_tile1_uvToUV(vec2 uv) {
    return textile_uvToUV(
        uv,
        _CSRG32F_TILE1_OFFSET_F,
        _CSRG32F_TILE1_SIZE_F,
        _CSRG32F_TEXTURE_SIZE_RCP
    );
}


#define _CSRG32F_TILE2_OFFSET ivec2(0, uval_mainImageSizeI.y)
#define _CSRG32F_TILE2_OFFSET_F vec2(0.0, uval_mainImageSize.y)
#define _CSRG32F_TILE2_SIZE uval_mainImageSizeI
#define _CSRG32F_TILE2_SIZE_F uval_mainImageSize

ivec2 csrg32f_tile2_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRG32F_TILE2_OFFSET,
        _CSRG32F_TILE2_SIZE
    );
}

vec2 csrg32f_tile2_uvToUV(vec2 uv) {
    return textile_uvToUV(
        uv,
        _CSRG32F_TILE2_OFFSET_F,
        _CSRG32F_TILE2_SIZE_F,
        _CSRG32F_TEXTURE_SIZE_RCP
    );
}

#define _CSRG32F_TILE3_OFFSET ivec2(0, uval_mainImageSizeI.y * 2)
#define _CSRG32F_TILE3_OFFSET_F vec2(0.0, uval_mainImageSize.y * 2.0)
#define _CSRG32F_TILE3_SIZE uval_mainImageSizeI
#define _CSRG32F_TILE3_SIZE_F uval_mainImageSize

ivec2 csrg32f_tile3_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSRG32F_TILE3_OFFSET,
        _CSRG32F_TILE3_SIZE
    );
}

vec2 csrg32f_tile3_uvToUV(vec2 uv) {
    return textile_uvToUV(
        uv,
        _CSRG32F_TILE3_OFFSET_F,
        _CSRG32F_TILE3_SIZE_F,
        _CSRG32F_TEXTURE_SIZE_RCP
    );
}

#endif