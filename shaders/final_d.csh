#version 460 compatibility

#include "rtwsm/RTWSM.glsl"

layout(r32f) uniform readonly image2D uimg_rtwsm_imap2D;
layout(r32f) uniform writeonly image2D uimg_rtwsm_imap2DSwap;

#define GAUSSIAN_BLUR_INPUT uimg_rtwsm_imap2D
#define GAUSSIAN_BLUR_OUTPUT uimg_rtwsm_imap2DSwap
#define GAUSSIAN_BLUR_CHANNELS 1
#define GAUSSIAN_BLUR_KERNEL_RADIUS 16
#define GAUSSIAN_BLUR_VERTICAL
const ivec3 workGroups = ivec3(WORKGROUP128_COUNT, SETTING_RTWSM_IMAP_SIZE, 1);
#include "pass/GaussianBlur.comp"