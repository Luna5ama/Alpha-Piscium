#version 460 compatibility

#include "/util/FullScreenComp.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(r32i) uniform iimage2D uimg_causticsPhoton;

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        ivec2 readPos = texelPos;
        int area = imageLoad(uimg_causticsPhoton, readPos).r;
        area = clamp(area, 0, 256 * 16 * 4);
        float v = float(area) / 256.0 / 16.0;
        imageStore(uimg_causticsPhoton, texelPos, ivec4(floatBitsToInt(v)));
    }
}