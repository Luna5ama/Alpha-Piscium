#ifndef GLOBAL_DATA_MODIFIER
#define GLOBAL_DATA_MODIFIER \

#endif

layout(std430, binding = 0) GLOBAL_DATA_MODIFIER buffer GlobalData {
    uint global_lumHistogram[257];
    mat4 global_shadowRotationMatrix;
    mat4 global_taaJitterMat;
    vec4 global_sunRadiance;
    vec2 global_taaJitter;
    vec4 global_exposure;
};