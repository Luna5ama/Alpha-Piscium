#version 460 compatibility
#include "atmosphere/Common.glsl"

const ivec3 workGroups = ivec3(EPIPOLAR_SLICE_D16, SLICE_SAMPLE_D16, 1);

layout(rgba32ui) uniform writeonly uimage2D uimg_epipolarData;
#define CLEAR_IMAGE1 uimg_epipolarData
const ivec4 CLEAR_IMAGE_BOUND = ivec4(0, 0, SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);
const uvec4 CLEAR_COLOR1 = uvec4(0u);

#include "general/Clear.comp"