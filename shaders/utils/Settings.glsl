#ifndef SETTINGS_GLSL_
#define SETTINGS_GLSL_

#define RTWSM_IMAP_SIZE 1024 // RTWSM importance map resolution [128 256 512 1024 2048]
const int shadowMapResolution = 1024; // [128 256 512 1024 2048 4096]
#define RTWSM_BACKWARD_IMPORTANCE_MULTIPLIER 5.0 // [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]

//#define RTWSM_DEBUG

const vec4 SHADOW_MAP_SIZE = vec4(vec2(shadowMapResolution), 1.0 / vec2(shadowMapResolution));
#endif