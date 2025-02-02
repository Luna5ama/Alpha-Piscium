#version 460 compatibility

#define DENOISER_KERNEL_RADIUS 2
#define DENOISER_BOX 1
#define DENOISER_VERTICAL 1
const vec2 workGroupsRender = vec2(1.0, 1.0);
#include "general/Denoiser.comp"

layout(r32f) uniform readonly image2D uimg_gbufferViewZ;
layout(rgba16f) uniform readonly image2D uimg_temp1;

uniform sampler2D usam_temp5;
uniform sampler2D usam_projReject;

layout(rgba16f) uniform readonly image2D uimg_temp3;
layout(rgba16f) uniform writeonly image2D uimg_temp4;
layout(rgba16f) uniform writeonly image2D uimg_svgfHistoryColor;

ivec2 denoiser_getImageSize() {
    return global_mainImageSizeI;
}

void denoiser_input(ivec2 coord, out vec4 data, out vec3 normal, out float viewZ) {
    data = vec4(imageLoad(uimg_temp3, coord).rgb, 0.0);
    normal = imageLoad(uimg_temp1, coord).rgb;
    viewZ = imageLoad(uimg_gbufferViewZ, coord).r;
}

void denoiser_output(ivec2 coord, vec4 data) {
    imageStore(uimg_temp4, coord, data);


    vec2 projReject = texelFetch(usam_projReject, coord, 0).rg;
    projReject = max(projReject, texelFetchOffset(usam_projReject, coord, 0, ivec2(-1, 0)).rg);
    projReject = max(projReject, texelFetchOffset(usam_projReject, coord, 0, ivec2(1, 0)).rg);
    projReject = max(projReject, texelFetchOffset(usam_projReject, coord, 0, ivec2(0, -1)).rg);
    projReject = max(projReject, texelFetchOffset(usam_projReject, coord, 0, ivec2(0, 1)).rg);

    float frustumTest = float(projReject.x > 0.0);
    float newPixel = float(projReject.y > 0.0);

    float hLen = texelFetch(usam_temp5, coord, 0).r * 255.0 + 1.0;
    hLen *= saturate(1.0 - frustumTest * 0.5);
    hLen *= saturate(1.0 - newPixel * 0.5);

    imageStore(uimg_svgfHistoryColor, coord, vec4(data.rgb, hLen));
}