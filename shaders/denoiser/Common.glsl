#include "/Base.glsl"
#include "/util/Colors.glsl"
#include "/textile/CSRGBA32UI.glsl"

void svgf_pack(out uvec4 packedData, vec3 color, vec3 fastColor, vec2 moments, float hLen) {
    packedData.x = packUnorm4x8(colors_sRGBToLogLuv32(color));
    packedData.y = packUnorm4x8(colors_sRGBToLogLuv32(fastColor));
    packedData.z = packHalf2x16(moments);
    packedData.w = floatBitsToUint(hLen);
}

void svgf_unpack(uvec4 packedData, out vec3 color, out vec3 fastColor, out vec2 moments, out float hLen) {
    color = colors_LogLuv32ToSRGB(unpackUnorm4x8(packedData.x));
    fastColor = colors_LogLuv32ToSRGB(unpackUnorm4x8(packedData.y));
    moments = unpackHalf2x16(packedData.z);
    hLen = uintBitsToFloat(packedData.w);
}