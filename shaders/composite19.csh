#version 460 compatibility

#define DENOISER_KERNEL_RADIUS (SETTING_DENOISER_FILTER_RADIUS * 2)
#define DENOISER_BOX 1
#define DENOISER_VERTICAL 1
const vec2 workGroupsRender = vec2(1.0, 1.0);
#include "/denoiser/Denoiser.comp.glsl"

layout(r32f) uniform readonly image2D uimg_gbufferViewZ;
layout(rgba16f) uniform readonly image2D uimg_temp1;

layout(rgba16f) uniform readonly image2D uimg_temp3;
layout(rgba16f) uniform restrict image2D uimg_ssvbil;

layout(rgba16f) uniform writeonly image2D uimg_giHistoryColor;

ivec2 denoiser_getImageSize() {
    return global_mainImageSizeI;
}

void denoiser_input(ivec2 coord, out vec4 data, out vec3 normal, out float viewZ) {
    data = vec4(imageLoad(uimg_temp3, coord).rgb, 0.0);
    data.a = colors_srgbLuma(data.rgb);
    normal = imageLoad(uimg_temp1, coord).rgb;
    viewZ = imageLoad(uimg_gbufferViewZ, coord).r;
}

void denoiser_output(ivec2 coord, vec4 data) {
    float ao = imageLoad(uimg_ssvbil, coord).a;
    vec3 gi = data.rgb;
    imageStore(uimg_ssvbil, coord, vec4(gi, ao));
    float hLen = texelFetch(usam_temp6, coord, 0).r * 255.0 + 1.0;
    imageStore(uimg_giHistoryColor, coord, vec4(data.rgb, hLen));
}