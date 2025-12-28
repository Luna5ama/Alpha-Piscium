#ifndef INCLUDE_techniques_debug_Common_glsl
#define INCLUDE_techniques_debug_Common_glsl a

#include "/util/Math.glsl"

ivec2 _debug_texelPos = ivec2(0);

bool inViewPort(ivec4 originSize, out vec2 texCoord) {
    originSize = ivec4(vec4(originSize) * SETTING_DEBUG_SCALE);
    ivec2 min = originSize.xy;
    ivec2 max = originSize.xy + originSize.zw;
    texCoord = saturate((vec2(_debug_texelPos - min) + 0.5) / vec2(originSize.zw));
    if (all(greaterThanEqual(_debug_texelPos.xy, min)) && all(lessThan(_debug_texelPos.xy, max))) {
        return true;
    }
    return false;
}

#endif