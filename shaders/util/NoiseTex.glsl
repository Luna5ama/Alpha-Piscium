#ifndef INCLUDE_util_NoiseTex_glsl
#define INCLUDE_util_NoiseTex_glsl a

#include "Sampling.glsl"

vec4 noisetex_blueNoise3D(ivec3 pos) {
    return texelFetch(usam_blueNoise3D, pos & ivec3(63u), 0);
}

vec4 noisetex_whiteNoise3D(ivec3 pos) {
    return texelFetch(usam_whiteNoise3D, pos & ivec3(63u), 0);
}

#endif