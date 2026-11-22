#define GLOBAL_DATA_MODIFIER writeonly buffer
#include "/util/Math.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    voxel_tintLastFrame = -1;
    for (int i = 0; i < voxel_tintHistory.length(); ++i) {
        voxel_tintHistory[i] = vec4(1.0, 1.0, 1.0, 0.0);
        for (int j = 0; j < voxel_tintSamples[i].length(); ++j) voxel_tintSamples[i][j] = 0u;
    }
    #ifdef SETTING_EXPOSURE_MANUAL
    global_aeData.cameraEV = float(SETTING_EXPOSURE_MANUAL_EV_COARSE) + SETTING_EXPOSURE_MANUAL_EV_FINE;
    #else
    global_aeData.cameraEV = SETTING_EXPOSURE_MIN_EV;
    #endif
    for (int i = 0; i < 256; ++i) global_aeData.histogram[i] = 0u;
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