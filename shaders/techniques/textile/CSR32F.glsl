#ifndef INCLUDE_textile_CSR32F_glsl
#define INCLUDE_textile_CSR32F_glsl a

#include "Common.glsl"

#define _CSR32F_TEXTURE_SIZE (uval_mainImageSizeI * ivec2(2, 3))
#define _CSR32F_TEXTURE_SIZE_F (uval_mainImageSize * vec2(2.0, 3.0))
#define _CSR32F_TEXTURE_SIZE_RCP rcp(_CSR32F_TEXTURE_SIZE_F)

#define _CSR32F_TILE1_OFFSET ivec2(0)
#define _CSR32F_TILE1_OFFSET_F vec2(0.0)
#define _CSR32F_TILE1_SIZE uval_mainImageSizeI
#define _CSR32F_TILE1_SIZE_F uval_mainImageSize

ivec2 csr32f_tile1_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSR32F_TILE1_OFFSET,
        _CSR32F_TILE1_SIZE
    );
}

vec2 csr32f_tile1_uvToUV(vec2 uv) {
    return textile_uvToUV(
        uv,
        _CSR32F_TILE1_OFFSET_F,
        _CSR32F_TILE1_SIZE_F,
        _CSR32F_TEXTURE_SIZE_RCP
    );
}


#define _CSR32F_TILE2_OFFSET ivec2(uval_mainImageSizeI.x, 0)
#define _CSR32F_TILE2_OFFSET_F vec2(uval_mainImageSize.x, 0.0)
#define _CSR32F_TILE2_SIZE uval_mainImageSizeI
#define _CSR32F_TILE2_SIZE_F uval_mainImageSize

ivec2 csr32f_tile2_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSR32F_TILE2_OFFSET,
        _CSR32F_TILE2_SIZE
    );
}

vec2 csr32f_tile2_uvToUV(vec2 uv) {
    return textile_uvToUV(
        uv,
        _CSR32F_TILE2_OFFSET_F,
        _CSR32F_TILE2_SIZE_F,
        _CSR32F_TEXTURE_SIZE_RCP
    );
}

#define _CSR32F_TILE3_OFFSET ivec2(0, uval_mainImageSizeI.y)
#define _CSR32F_TILE3_OFFSET_F vec2(0.0, uval_mainImageSize.y)
#define _CSR32F_TILE3_SIZE uval_mainImageSizeI
#define _CSR32F_TILE3_SIZE_F uval_mainImageSize

ivec2 csr32f_tile3_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSR32F_TILE3_OFFSET,
        _CSR32F_TILE3_SIZE
    );
}

vec2 csr32f_tile3_uvToUV(vec2 uv) {
    return textile_uvToUV(
        uv,
        _CSR32F_TILE3_OFFSET_F,
        _CSR32F_TILE3_SIZE_F,
        _CSR32F_TEXTURE_SIZE_RCP
    );
}

#define _CSR32F_TILE4_OFFSET ivec2(uval_mainImageSizeI.xy)
#define _CSR32F_TILE4_OFFSET_F vec2(uval_mainImageSize.xy)
#define _CSR32F_TILE4_SIZE uval_mainImageSizeI
#define _CSR32F_TILE4_SIZE_F uval_mainImageSize

ivec2 csr32f_tile4_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSR32F_TILE4_OFFSET,
        _CSR32F_TILE4_SIZE
    );
}

vec2 csr32f_tile4_uvToUV(vec2 uv) {
    return textile_uvToUV(
        uv,
        _CSR32F_TILE4_OFFSET_F,
        _CSR32F_TILE4_SIZE_F,
        _CSR32F_TEXTURE_SIZE_RCP
    );
}

#define _CSR32F_TILE5_OFFSET ivec2(0, uval_mainImageSizeI.y * 2)
#define _CSR32F_TILE5_OFFSET_F vec2(0.0, uval_mainImageSize.y * 2.0)
#define _CSR32F_TILE5_SIZE uval_mainImageSizeI
#define _CSR32F_TILE5_SIZE_F uval_mainImageSize

ivec2 csr32f_tile5_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSR32F_TILE5_OFFSET,
        _CSR32F_TILE5_SIZE
    );
}

vec2 csr32f_tile5_uvToUV(vec2 uv) {
    return textile_uvToUV(
        uv,
        _CSR32F_TILE5_OFFSET_F,
        _CSR32F_TILE5_SIZE_F,
        _CSR32F_TEXTURE_SIZE_RCP
    );
}

#define _CSR32F_TILE6_OFFSET ivec2(uval_mainImageSizeI.x, uval_mainImageSizeI.y * 2)
#define _CSR32F_TILE6_OFFSET_F vec2(uval_mainImageSize.x, uval_mainImageSize.y * 2.0)
#define _CSR32F_TILE6_SIZE uval_mainImageSizeI
#define _CSR32F_TILE6_SIZE_F uval_mainImageSize

ivec2 csr32f_tile6_texelToTexel(ivec2 texelPos) {
    return textile_texelToTexel(
        texelPos,
        _CSR32F_TILE6_OFFSET,
        _CSR32F_TILE6_SIZE
    );
}

vec2 csr32f_tile6_uvToUV(vec2 uv) {
    return textile_uvToUV(
        uv,
        _CSR32F_TILE6_OFFSET_F,
        _CSR32F_TILE6_SIZE_F,
        _CSR32F_TEXTURE_SIZE_RCP
    );
}

#endif