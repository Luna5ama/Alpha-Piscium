#ifndef INCLUDE_Settings.glsl
#define INCLUDE_Settings.glsl

#define RTWSM_IMAP_SIZE 1024 // RTWSM importance map resolution [128 256 512 1024 2048]
const int shadowMapResolution = 2048; // [128 256 512 1024 2048 4096]
#define RTWSM_BACKWARD_IMPORTANCE 8.0 // [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]

//#define RTWSM_DEBUG
#define RTWSM_SURFACE_NORMAL 4.0 // [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0]
#define RTWSM_SHADOW_EDGE 0.05 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]

const vec4 SHADOW_MAP_SIZE = vec4(vec2(shadowMapResolution), 1.0 / vec2(shadowMapResolution));
#endif