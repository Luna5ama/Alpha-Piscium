#define GLOBAL_DATA_MODIFIER writeonly
#include "/Base.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    global_aeData.expValues = vec3((SETTING_EXPOSURE_MIN_EV + SETTING_EXPOSURE_MAX_EV) * 0.5);
    global_shadowAABBMin = ivec3(0);
    global_shadowAABBMax = ivec3(0);
    global_shadowAABBMinNew = ivec3(0);
    global_shadowAABBMaxNew = ivec3(0);
}