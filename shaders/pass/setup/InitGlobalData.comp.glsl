#define GLOBAL_DATA_MODIFIER writeonly
#include "/util/Math.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    global_aeData.expValues = vec3((SETTING_EXPOSURE_MIN_EV + SETTING_EXPOSURE_MAX_EV) * 0.5);
    global_shadowAABBMin = ivec3(INT32_MAX);
    global_shadowAABBMax = ivec3(INT32_MIN);
    global_shadowAABBMinHistory = vec3(FLT_MAX);
    global_shadowAABBMaxHistory = vec3(-FLT_MAX);
    global_shadowAABBMinNew = ivec3(INT32_MAX);
    global_shadowAABBMaxNew = ivec3(INT32_MIN);
}