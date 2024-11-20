#version 460 compatibility

const bool colortex9MipmapEnabled = true;

#define SSVBIL_SAMPLE_STEPS 32
#define SSVBIL_SAMPLE_SLICES 2
#include "post/SSVBIL.frag"