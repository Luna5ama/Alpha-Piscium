#version 460 compatibility

#include "rtwsm/RTWSM.glsl"

layout(rgba16f) uniform readonly image2D uimg_skyLUT;
layout(rgba16f) uniform writeonly image2D uimg_main;

#define GAUSSIAN_BLUR_INPUT uimg_skyLUT
#define GAUSSIAN_BLUR_OUTPUT uimg_main
#define GAUSSIAN_BLUR_CHANNELS 4
#define GAUSSIAN_BLUR_KERNEL_RADIUS 16
#define GAUSSIAN_BLUR_HORIZONTAL
const ivec3 workGroups = ivec3(1, 128, 1);
#include "general/GaussianBlur.comp"