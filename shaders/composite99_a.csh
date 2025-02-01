#version 460 compatibility

#define GLOBAL_DATA_MODIFIER restrict
#include "_Util.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    gbufferPrevProjectionInverse = gbufferProjectionInverse;
    gbufferPrevModelViewInverse = gbufferModelViewInverse;
    global_prevCameraDelta = cameraPosition - previousCameraPosition;

    gbufferPrevProjectionJitter = gbufferProjectionJitter;
    gbufferPrevProjectionJitterInverse = gbufferProjectionJitterInverse;
}