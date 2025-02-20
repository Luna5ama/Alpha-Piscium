#version 460 compatibility

const bool colortex1MipmapEnabled = true;
const bool colortex2MipmapEnabled = true;
const bool colortex9MipmapEnabled = true;

#define SSVBIL_SAMPLE_STEPS SETTING_SSVBIL_STEPS
#define SSVBIL_SAMPLE_SLICES SETTING_SSVBIL_SLICES
#include "/post/gtvbgi/GTVBGI.glsl"