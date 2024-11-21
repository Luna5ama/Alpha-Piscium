#version 460 compatibility

#define GLOBAL_DATA_MODIFIER writeonly
#include "_Util.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);


void main() {
    gbufferPreviousProjectionInverse = gbufferProjectionInverse;
    gbufferPreviousModelViewInverse = gbufferModelViewInverse;
}