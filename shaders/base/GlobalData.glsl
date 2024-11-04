#ifndef GLOBAL_DATA_MODIFIER
#define GLOBAL_DATA_MODIFIER readonly
#endif

layout(std430, binding = 0) GLOBAL_DATA_MODIFIER buffer GlobalData {
    mat4 shadowRotationMatrix;
    mat4 taaJitterMat;
    vec2 taaJitter;
} ssbo_globalData;