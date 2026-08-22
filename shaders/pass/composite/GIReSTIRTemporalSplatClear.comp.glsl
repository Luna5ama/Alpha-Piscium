#include "/util/FullScreenComp.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(r32ui) uniform restrict writeonly uimage2D uimg_r32ui;

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        transient_restir_primary_store(texelPos, uvec4(0u));
    }
}
