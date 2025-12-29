#define GLOBAL_DATA_MODIFIER buffer

#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/gi/Common.glsl"
#include "/techniques/HiZCheck.glsl"

#define SPD_CHANNELS 4
#define SPD_OP 2
#include "/techniques/ffx/spd/SPD.comp.glsl"

layout(rgba16f) uniform restrict writeonly image2D uimg_rgba16f;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;

vec4 spd_loadInput(ivec2 texelPos, uint slice) {
    texelPos = clamp(texelPos, ivec2(0), uval_mainImageSizeI - 1);
    vec4 result = vec4(0.0);
    if (gl_WorkGroupID.z == 0) {
        result.xyz = transient_viewNormal_fetch(texelPos).xyz * 2.0 - 1.0;
    } else {
        GIHistoryData historyData = gi_historyData_init();

        if (gl_WorkGroupID.z == 1) {
            result = transient_gi1Reprojected_fetch(texelPos);
        } else {
            result = transient_gi3Reprojected_fetch(texelPos);
        }
    }

    return result;
}

vec4 spd_loadOutput(ivec2 texelPos, uint level, uint slice) {
    return vec4(0.0);
}

shared ivec4 shared_mipTiles[16];

void spd_storeOutput(ivec2 texelPos, uint level, uint slice, vec4 value) {
    ivec4 mipTile = shared_mipTiles[level];
    ivec2 storePos = mipTile.xy + texelPos;
    if (all(lessThan(texelPos, mipTile.zw))) {
        if (gl_WorkGroupID.z == 0) {
            transient_viewNormalMip_store(storePos, vec4(value.xyz * 0.5 + 0.5, 0.0));
        } else if (gl_WorkGroupID.z == 1) {
            transient_gi_diffMip_store(storePos, value);
        } else if (gl_WorkGroupID.z == 2) {
            transient_gi_specMip_store(storePos, value);
        }
    }
}

uint spd_mipCount() {
    return 6u;
}
void spd_init() {
    if (gl_LocalInvocationIndex < 16u) {
        ivec4 mipTile = global_mipmapTileCeil[gl_LocalInvocationIndex];
        shared_mipTiles[gl_LocalInvocationIndex] = mipTile;
    }
    barrier();
}