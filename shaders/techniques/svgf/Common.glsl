#include "/Base.glsl"
#include "/util/Colors.glsl"
#include "/util/Colors2.glsl"
#include "/techniques/textile/CSRGBA32UI.glsl"

void svgf_pack(out uvec4 packedData, vec3 color, vec3 fastColor, vec2 moments, float hLen) {
    color = clamp(color, 0.0, FP16_MAX);
    fastColor = clamp(fastColor, 0.0, FP16_MAX);
    moments = clamp(moments, 0.0, FP16_MAX);
    hLen = clamp(hLen, 0.0, FP16_MAX);
    packedData.x = packHalf2x16(color.xy);
    packedData.y = packHalf2x16(vec2(color.z, moments.y));
    packedData.z = packHalf2x16(fastColor.xy);
    packedData.w = packHalf2x16(vec2(fastColor.z, hLen));
}

void svgf_unpack(uvec4 packedData, out vec3 color, out vec3 fastColor, out vec2 moments, out float hLen) {
    vec2 temp1 = unpackHalf2x16(packedData.x);
    vec2 temp2 = unpackHalf2x16(packedData.y);
    vec2 temp3 = unpackHalf2x16(packedData.z);
    vec2 temp4 = unpackHalf2x16(packedData.w);
    color = vec3(temp1.xy, temp2.x);
    moments = vec2(min(colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, color), 256.0), temp2.y);
    fastColor = vec3(temp3.xy, temp4.x);
    hLen = temp4.y;
}