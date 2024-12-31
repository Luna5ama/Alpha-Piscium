#ifndef GLOBAL_DATA_MODIFIER
#define GLOBAL_DATA_MODIFIER \

#endif

layout(std430, binding = 0) GLOBAL_DATA_MODIFIER buffer GlobalData {
    mat4 gbufferPreviousProjectionInverse;
    mat4 gbufferPreviousModelViewInverse;
    mat4 global_shadowRotationMatrix;
    mat4 global_taaJitterMat;
    vec4 global_sunRadiance;
    vec4 global_exposure;
    vec2 global_taaJitter;
    ivec2 global_mainImageSizeI;
    vec2 global_mainImageSize;
    vec2 global_mainImageSizeRcp;
    uint global_lumHistogram[257];
};

const vec2 SHADOW_MAP_SIZE = vec2(float(shadowMapResolution), 1.0 / float(shadowMapResolution));
const vec3 MOON_RADIANCE_MUL = 0.02 * 0.12 * vec3(0.8, 0.9, 1.0);