#include "/_Base.glsl"

void svgf_pack(out uvec4 packedData, vec4 colorHLen, vec2 moments) {
    packedData.x = packHalf2x16(colorHLen.rg);
    packedData.y = packHalf2x16(colorHLen.ba);
    packedData.z = packHalf2x16(moments);
    packedData.w = 0u;
}

void svgf_unpack(uvec4 packedData, out vec4 colorHLen, out vec2 moments) {
    vec2 temp1 = unpackHalf2x16(packedData.x);
    vec2 temp2 = unpackHalf2x16(packedData.y);
    vec2 temp3 = unpackHalf2x16(packedData.z);
//    vec2 temp4 = unpackHalf2x16(packedData.w); // unused

    colorHLen.rg = temp1;
    colorHLen.ba = temp2;
    moments = temp3;
}