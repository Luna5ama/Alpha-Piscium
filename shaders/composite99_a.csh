#version 460 compatibility

#define GLOBAL_DATA_MODIFIER restrict
#include "/_Base.glsl"

#ifdef SETTING_DEBUG_AE
layout(local_size_x = 256) in;
#else
layout(local_size_x = 1) in;
#endif
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    if (gl_LocalInvocationIndex == 0) {
        gbufferPrevProjectionInverse = gbufferProjectionInverse;
        gbufferPrevModelViewInverse = gbufferModelViewInverse;
        global_prevCameraDelta = cameraPosition - previousCameraPosition;

        gbufferPrevProjectionJitter = gbufferProjectionJitter;
        gbufferPrevProjectionJitterInverse = gbufferProjectionJitterInverse;
    }
    #ifdef SETTING_DEBUG_AE
    global_aeData.lumHistogram[gl_LocalInvocationIndex] = 0u;
    #endif
}