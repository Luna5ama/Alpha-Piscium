#version 460 compatibility

const bool colortex1MipmapEnabled = true;
const bool colortex2MipmapEnabled = true;
const bool colortex9MipmapEnabled = true;

#define SSVBIL_SAMPLE_STEPS 32
#define SSVBIL_SAMPLE_SLICES 2
#include "post/SSVBIL.frag"