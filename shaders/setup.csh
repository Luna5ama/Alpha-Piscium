#version 460 compatibility

#define GLOBAL_DATA_MODIFIER writeonly
#include "/_Base.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    global_exposure = vec4((SETTING_EXPOSURE_MIN_EV + SETTING_EXPOSURE_MAX_EV) * 0.5);
}