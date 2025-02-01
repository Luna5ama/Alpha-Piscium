#ifndef GLOBAL_DATA_MODIFIER
#define GLOBAL_DATA_MODIFIER \

#endif

mat4 gbufferPrevProjection = gbufferPreviousProjection;
mat4 gbufferPrevModelView = gbufferPreviousModelView;

layout(std430, binding = 0) GLOBAL_DATA_MODIFIER buffer GlobalData {
    mat4 gbufferPrevProjectionInverse;
    mat4 gbufferPrevModelViewInverse;
    mat4 global_shadowRotationMatrix;
    mat4 global_taaJitterMat;
    mat4 gbufferProjectionJitter;
    mat4 gbufferProjectionJitterInverse;
    mat4 gbufferPrevProjectionJitter;
    mat4 gbufferPrevProjectionJitterInverse;
    vec4 global_sunRadiance;
    vec4 global_exposure;
    vec3 global_prevCameraDelta;
    vec2 global_taaJitter;
    ivec2 global_mainImageSizeI;
    vec2 global_mainImageSize;
    vec2 global_mainImageSizeRcp;
    uint global_lumHistogram[257];
};

const vec2 SHADOW_MAP_SIZE = vec2(float(shadowMapResolution), 1.0 / float(shadowMapResolution));
const vec3 MOON_RADIANCE_MUL = 0.002 * vec3(0.8, 0.9, 1.0);