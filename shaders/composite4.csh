#version 460 compatibility

#if SETTING_SSVBIL_LOD_MUL != 0
const bool colortex1MipmapEnabled = true;
const bool colortex2MipmapEnabled = true;
const bool colortex9MipmapEnabled = true;
#endif

#include "atmosphere/EpipolarScattering.comp"