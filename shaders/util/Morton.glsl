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
// Encode functions use multiply instead of shift+OR for bit-spreading:
// since bit groups never overlap, (x | (x << N)) == x * (1 + 2^N).
// Integer multiply routes to the FMAHeavy pipe, relieving the ALU pipe.
uint morton3D_6bEncodeFMAHeavy(uvec3 coords) {
    uvec3 x = coords & 0x03u;
    x = (x * 5u) & 0x09u;         // x*(1+2^2) == x|(x<<2), pairs isolated
    return x.x + x.y * 2u + x.z * 4u;
}

uint morton3D_6bEncode(uvec3 coords) {
    uvec3 x = coords & 0x03u;
    x = (x | (x << 2u)) & 0x09u;
    return x.x | (x.y << 1u) | (x.z << 2u);
}

uvec3 morton3D_6bDecode(uint x) {
    uvec3 coords = uvec3(x, x >> 1u, x >> 2u);
    coords &= 0x09u;
    coords = (coords | (coords >> 2u)) & 0x03u;
    return coords;
}

uint morton3D_12bEncodeFMAHeavy(uvec3 coords) {
    uvec3 x = coords & 0x0Fu;
    x = (x * 17u) & 0xC3u;        // x*(1+2^4) == x|(x<<4), nibbles isolated
    x = (x *  5u) & 0x249u;       // x*(1+2^2) == x|(x<<2), pairs isolated
    return x.x + x.y * 2u + x.z * 4u;
}

uint morton3D_12bEncode(uvec3 coords) {
    uvec3 x = coords & 0x0Fu;
    x = (x | (x << 4u)) & 0xC3u;
    x = (x | (x << 2u)) & 0x249u;
    return x.x | (x.y << 1u) | (x.z << 2u);
}

uvec3 morton3D_12bDecode(uint x) {
    uvec3 coords = uvec3(x, x >> 1u, x >> 2u);
    coords &= 0x249u;
    coords = (coords | (coords >> 2u)) & 0xC3u;
    coords = (coords | (coords >> 4u)) & 0x0Fu;
    return coords;
}

uint morton3D_24bEncodeFMAHeavy(uvec3 coords) {
    uvec3 x = coords & 0xFFu;
    x = (x * 257u) & 0x00F00Fu;   // x*(1+2^8) == x|(x<<8) when x<256
    x = (x *  17u) & 0x0C30C3u;   // x*(1+2^4) == x|(x<<4), nibbles isolated
    x = (x *   5u) & 0x249249u;   // x*(1+2^2) == x|(x<<2), pairs isolated
    return x.x + x.y * 2u + x.z * 4u;
}

uint morton3D_24bEncode(uvec3 coords) {
    uvec3 x = coords & 0xFFu;
    x = (x | (x << 8u)) & 0x00F00Fu;
    x = (x | (x << 4u)) & 0x0C30C3u;
    x = (x | (x << 2u)) & 0x249249u;
    return x.x | (x.y << 1u) | (x.z << 2u);
}

uvec3 morton3D_24bDecode(uint x) {
    uvec3 coords = uvec3(x, x >> 1u, x >> 2u);
    coords &= 0x249249u;
    coords = (coords | (coords >> 2u)) & 0x0C30C3u;
    coords = (coords | (coords >> 4u)) & 0x00F00Fu;
    coords = (coords | (coords >> 8u)) & 0x0000FFu;
    return coords;
}

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

// ---------------------------------------------------------------------------
// 2-D Hilbert curve
// ---------------------------------------------------------------------------
// Maps a pos in [0, 2^order)² to a Hilbert index in [0, 4^order).
// Use order=11 for a 22-bit index (2048×2048 grid), order=12 for 24 bits.
//
// Overflow check (order=12): max s=2048, s*s=4,194,304, 3*s*s=12,582,912,
// max d = 4096*4096-1 = 16,777,215 = 0xFFFFFF — fits in 32-bit uint.
uint hilbert2D_encode(uvec2 pos, uint order) {
    uint d = 0u;
    uint n = (1u << order) - 1u;
    for (uint s = 1u << (order - 1u); s > 0u; s >>= 1u) {
        uint rx = (pos.x & s) > 0u ? 1u : 0u;
        uint ry = (pos.y & s) > 0u ? 1u : 0u;
        d += s * s * ((3u * rx) ^ ry);
        // Rotate quadrant into canonical orientation
        if (ry == 0u) {
            if (rx == 1u) {
                pos = n - pos.yx;   // reflect + swap
            } else {
                pos = pos.yx;       // swap only
            }
        }
    }
    return d;
}

#endif