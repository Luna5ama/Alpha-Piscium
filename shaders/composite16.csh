#version 460 compatibility

#define DENOISER_KERNEL_RADIUS SETTING_DENOISER_FILTER_RADIUS
#define DENOISER_BOX 1
#define DENOISER_HORIZONTAL 1
const vec2 workGroupsRender = vec2(1.0, 1.0);
#include "/denoiser/Denoiser.comp.glsl"

layout(r32f) uniform readonly image2D uimg_gbufferViewZ;
layout(rgba16f) uniform readonly image2D uimg_temp1;

layout(rgba16f) uniform readonly image2D uimg_temp4;
layout(rgba16f) uniform writeonly image2D uimg_temp3;

ivec2 denoiser_getImageSize() {
    return global_mainImageSizeI;
}

void denoiser_input(ivec2 coord, out vec4 data, out vec3 normal, out float viewZ) {
    data = vec4(imageLoad(uimg_temp4, coord).rgb, 0.0);
    data.a = colors_srgbLuma(data.rgb);
    normal = imageLoad(uimg_temp1, coord).rgb;
    viewZ = imageLoad(uimg_gbufferViewZ, coord).r;
}

void denoiser_output(ivec2 coord, vec4 data) {
    imageStore(uimg_temp3, coord, data);
}
