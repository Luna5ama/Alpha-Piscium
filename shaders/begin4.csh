#version 460 compatibility
#include "atmosphere/Common.glsl"

const ivec3 workGroups = ivec3(EPIPOLAR_SLICE_D16, SLICE_SAMPLE_D16, 1);

layout(rgba16f) uniform writeonly image2D uimg_epipolarTransmittance;
#define CLEAR_IMAGE1 uimg_epipolarTransmittance
const ivec4 CLEAR_IMAGE_BOUND = ivec4(0, 0, SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);
const vec4 CLEAR_COLOR1 = vec4(0.0);

#include "general/Clear1.comp"