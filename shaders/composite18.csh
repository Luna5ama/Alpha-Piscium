#version 460 compatibility

#define DENOISER_KERNEL_RADIUS 16
#define DENOISER_BOX 1
#define DENOISER_HORIZONTAL 1
const vec2 workGroupsRender = vec2(1.0, 1.0);
#include "general/Denoiser.comp"

layout(r32f) uniform readonly image2D uimg_gbufferViewZ;
layout(rgba16f) uniform readonly image2D uimg_temp1;

layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba16f) uniform readonly image2D uimg_ssvbil;

ivec2 denoiser_getImageSize() {
    return imageSize(uimg_ssvbil);
}

void denoiser_input(ivec2 coord, out vec4 data, out vec3 normal, out float viewZ) {
    data = vec4(imageLoad(uimg_ssvbil, coord).rgb, 0.0);
    normal = imageLoad(uimg_temp1, coord).rgb;
    viewZ = imageLoad(uimg_gbufferViewZ, coord).r;
}

void denoiser_output(ivec2 coord, vec4 data) {
    imageStore(uimg_temp3, coord, vec4(data.rgb, 0.0));
}
