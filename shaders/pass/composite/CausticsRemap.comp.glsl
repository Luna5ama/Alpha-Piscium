#include "/util/FullScreenComp.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(RENDER_SCALE_FACTOR, RENDER_SCALE_FACTOR);

layout(r32i) uniform iimage2D uimg_r32f;

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        int area = transient_caustics_input_load(texelPos).r;
        area = clamp(area, 0, 256 * 32 * 16);
        float v = float(area) / 256.0 / 16.0;
        transient_caustics_final_store(texelPos, ivec4(floatBitsToInt(v)));
    }
}