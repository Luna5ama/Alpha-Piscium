#define GLOBAL_DATA_MODIFIER \

#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/gi/Common.glsl"
#include "/techniques/HiZCheck.glsl"

#define SPD_CHANNELS 4
#define SPD_OP 2
#include "/techniques/ffx/spd/SPD.comp.glsl"

layout(rgba16f) uniform restrict image2D uimg_rgba16f;
const vec2 workGroupsRender = vec2(0.25, 0.25);

vec4 spd_loadInput(ivec2 texelPos) {
    ivec2 clampedTexelPos = clamp(texelPos, ivec2(0), uval_mainImageSizeI - 1);
    float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(gl_WorkGroupID.xy, 4, clampedTexelPos);
    vec4 result = vec4(0.0);

    if (viewZ > -65536.0) {
        GIHistoryData historyData = gi_historyData_init();
        gi_historyData_unpack1(historyData, transient_gi1Reprojected_fetch(texelPos));
        gi_historyData_unpack2(historyData, transient_gi2Reprojected_fetch(texelPos));
        gi_historyData_unpack3(historyData, transient_gi3Reprojected_fetch(texelPos));
        gi_historyData_unpack4(historyData, transient_gi4Reprojected_fetch(texelPos));
        gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));

        #if MIP_TYPE == 0
        result = vec4(historyData.diffuseColor, historyData.diffuseHitDistance);
        #elif MIP_TYPE == 1
        result = vec4(historyData.specularColor, historyData.specularHitDistance);
        #endif
    }

    return result;
}

vec4 spd_loadOutput(ivec2 texelPos, uint level) {
    return vec4(0.0);
}

void spd_storeOutput(ivec2 texelPos, uint level, vec4 value) {
    ivec2 mipSize = global_mipmapSizesI[level];
    ivec2 mipOffset = ivec2(global_mipmapSizePrefixes[level - 1].x - uval_mainImageSizeI.x, 0);
    ivec2 storePos = mipOffset + texelPos;
    if (all(lessThanEqual(texelPos, mipSize))) {
        #if MIP_TYPE == 0
        transient_gi_diffMip_store(storePos, value);
        #elif MIP_TYPE == 1
        transient_gi_specMip_store(storePos, value);
        #endif
    }
}

uint spd_mipCount() {
    return 4u;
}