#ifndef INCLUDE_textile_CSRGBA32UI_glsl
#define INCLUDE_textile_CSRGBA32UI_glsl a

#include "Common.glsl"

#define _CSRGBA32UI_TEXTURE_SIZE (global_mainImageSizeI * ivec2(1, 2))
#define _CSRGBA32UI_TEXTURE_SIZE_F (global_mainImageSize * vec2(1.0, 2.0))
#define _CSRGBA32UI_TEXTURE_SIZE_RCP rcp(_CSRGBA32UI_TEXTURE_SIZE_F)

#define _CSRGBA32UI_DIFFUSE_HISTORY_OFFSET ivec2(0)
#define _CSRGBA32UI_DIFFUSE_HISTORY_OFFSET_F vec2(0.0)
#define _CSRGBA32UI_DIFFUSE_HISTORY_SIZE global_mainImageSizeI
#define _CSRGBA32UI_DIFFUSE_HISTORY_SIZE_F global_mainImageSize

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

#endif