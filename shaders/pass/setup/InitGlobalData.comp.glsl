#define GLOBAL_DATA_MODIFIER writeonly buffer
#include "/util/Math.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    global_giWideState = 0;
    #ifdef VOXY
    global_giModelOrigin = ivec3(0);
    global_giModelEpoch = 1u;
    global_giLightEpoch = 0u;
    for (int i = 0; i < 512; ++i) global_giLightGroupEpoch[i] = 0u;
    for (int i = 0; i < 32768; ++i) global_giObservedCells[i] = uvec2(0);
    #endif
    global_aeData.expValues = vec4(SETTING_EXPOSURE_MIN_EV, SETTING_EXPOSURE_MIN_EV, SETTING_EXPOSURE_MIN_EV, 0.0);
    global_shadowAABBMin = ivec3(INT32_MAX);
    global_shadowAABBMax = ivec3(INT32_MIN);
    global_shadowAABBMinHistory = vec3(FLT_MAX);
    global_shadowAABBMaxHistory = vec3(-FLT_MAX);
    global_shadowAABBMinNew = ivec3(INT32_MAX);
    global_shadowAABBMaxNew = ivec3(INT32_MIN);
    global_historyResetFactor = 1.0;
    global_motionFactor = vec4(1.0);
    global_taaResetFactor = vec4(1.0);
}