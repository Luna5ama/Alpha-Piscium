#ifndef INCLUDE_util_Morton_glsl
#define INCLUDE_util_Morton_glsl a
#include "/_Base.glsl"

// Adapted from https://gist.github.com/JarkkoPFC/0e4e599320b0cc7ea92df45fb416d79a
uvec2 morton_8bDecode(uint code) {
    uvec2 result = uvec2(code, code >> 1) & 0x55u;
    result = (result | (result >> 1)) & 0x33u;
    result = (result | (result >> 2)) & 0x0fu;
    return result;
}

uvec2 morton_16bDecode(uint code) {
    uvec2 result = uvec2(code, code >> 1) & 0x5555u;
    result = (result | (result >> 1)) & 0x3333u;
    result = (result | (result >> 2)) & 0x0f0fu;
    result = (result | (result >> 4)) & 0x00ffu;
    return result;
}

uvec2 morton_32bDecode(uint code) {
    uvec2 result = uvec2(code, code >> 1) & 0x55555555u;
    result = (result | (result >> 1)) & 0x33333333u;
    result = (result | (result >> 2)) & 0x0f0f0f0fu;
    result = (result | (result >> 4)) & 0x00ff00ffu;
    result = (result | (result >> 8)) & 0x0000ffffu;
    return result;
}

#endif