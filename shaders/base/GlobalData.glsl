#ifndef GLOBAL_DATA_MODIFIER
#define GLOBAL_DATA_MODIFIER readonly
#endif

layout(std430, binding = 0) GLOBAL_DATA_MODIFIER buffer GlobalData {
    mat4 global_shadowRotationMatrix;
    mat4 global_taaJitterMat;
    vec4 global_sunRadiance;
    vec2 global_taaJitter;
};