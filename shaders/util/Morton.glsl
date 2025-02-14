#ifndef INCLUDE_util_Morton_glsl
#define INCLUDE_util_Morton_glsl a
#include "/_Base.glsl"

uvec2 morton_8bDecode(uint code) {
    uvec2 result = uvec2(code, code >> 1) & 0x55u;
    result = (result | (result >> 1)) & 0x33u;
    result = (result | (result >> 2)) & 0x0fu;
    return result;
}

#endif