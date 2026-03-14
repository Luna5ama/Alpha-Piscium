#ifndef INCLUDE_util_Morton_glsl
#define INCLUDE_util_Morton_glsl a
#include "/Base.glsl"

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

uint morton_8bEncode(uvec2 coords) {
    uvec2 x = coords & 0x0Fu;
    x = (x | (x << 2)) & 0x33u;
    x = (x | (x << 1)) & 0x55u;
    return x.x | (x.y << 1);
}

uint morton_16bEncode(uvec2 coords) {
    uvec2 x = coords & 0xFFu;
    x = (x | (x << 4)) & 0x0F0Fu;
    x = (x | (x << 2)) & 0x3333u;
    x = (x | (x << 1)) & 0x5555u;
    return x.x | (x.y << 1);
}

uint morton_32bEncode(uvec2 coords) {
    uvec2 x = coords & 0xFFFFu;
    x = (x | (x << 8)) & 0x00FF00FFu;
    x = (x | (x << 4)) & 0x0F0F0F0Fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    return x.x | (x.y << 1);
}

// Adapted from https://github.com/liamdon/fast-morton
uint morton3D_30bEncode(uvec3 coords) {
    uvec3 x = coords;
    x &= 0x000003FFu;
    x = (x | (x << 16u)) & 0x000003FFu;
    x = (x | (x <<  8u)) & 0x0300F00Fu;
    x = (x | (x <<  4u)) & 0x030C30C3u;
    x = (x | (x <<  2u)) & 0x09249249u;
    return x.x | (x.y << 1u) | (x.z << 2u);
}

uvec3 morton3D_30bDecode(uint x) {
    uvec3 coords = uvec3(x, x >> 1u, x >> 2u);
    coords &= 0x09249249u;
    coords = (coords | (coords >>  2u)) & 0x030C30C3u;
    coords = (coords | (coords >>  4u)) & 0x0300F00Fu;
    coords = (coords | (coords >>  8u)) & 0x030000FFu;
    coords = (coords | (coords >> 16u)) & 0x000003FFu;
    return coords;
}

#endif