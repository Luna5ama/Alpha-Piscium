#version 460 compatibility

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#define SSVBIL_SAMPLE_STEPS SETTING_SSVBIL_STEPS
#define SSVBIL_SAMPLE_SLICES SETTING_SSVBIL_SLICES
#include "/util/FullScreenComp.glsl"
#include "/post/gtvbgi/GTVBGI.glsl"

layout(rgba16f) uniform writeonly image2D uimg_ssvbil;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 giResult = gtvbgi();
        imageStore(uimg_ssvbil, texelPos, giResult);
    }
}