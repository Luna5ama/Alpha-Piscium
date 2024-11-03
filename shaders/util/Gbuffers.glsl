#ifndef INCLUDE_GBuffers.glsl
#define INCLUDE_GBuffers.glsl
#include "../_Base.glsl"
#include "BitPacking.glsl"

void gbuffer_pack(out uvec4 value, vec4 colorMul, vec2 texCoord, vec2 lmCoord, vec3 viewNormal, uint blockID) {
    value.r = packUnorm4x8(colorMul);

    value.g = packS11(viewNormal.x);
    value.g |= packS11(viewNormal.y) << 11;
    value.g |= packS10(viewNormal.z) << 22;

    value.b = packUnorm2x16(texCoord);

    value.a = packUnorm4x8(vec4(lmCoord, 0.0, 0.0)) & 0x0000FFFFu;
    value.a |= (blockID & 0xFFFFu) << 16;
}

void gbuffer_unpack(in uvec4 value, out vec4 colorMul, out vec2 texCoord, out vec2 lmCoord, out vec3 viewNormal, out uint blockID) {
    colorMul = unpackUnorm4x8(value.r);

    viewNormal.x = unpackS11(value.g & 0x7FFu);
    viewNormal.y = unpackS11((value.g >> 11) & 0x7FFu);
    viewNormal.z = unpackS10((value.g >> 22) & 0x3FFu);

    texCoord = unpackUnorm2x16(value.b);

    lmCoord = unpackUnorm4x8(value.a).xy;
    blockID = (value.a >> 16) & 0xFFFFu;
}

#endif