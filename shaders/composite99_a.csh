#version 460 compatibility

#define GLOBAL_DATA_MODIFIER restrict
#include "/Base.glsl"

#ifdef SETTING_DEBUG_AE
layout(local_size_x = 256) in;
#else
layout(local_size_x = 1) in;
#endif
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    if (gl_LocalInvocationIndex == 0) {
        gbufferPrevModelViewInverse = gbufferModelViewInverse;
        global_prevCameraDelta = uval_cameraDelta;

        global_shadowView = shadowModelView;
        global_shadowViewInverse = shadowModelViewInverse;
    }
    #ifdef SETTING_DEBUG_AE
    global_aeData.lumHistogram[gl_LocalInvocationIndex] = 0u;
    #endif
}