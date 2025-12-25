#define GLOBAL_DATA_MODIFIER \

#extension GL_KHR_shader_subgroup_ballot : enable

#include "/util/GBufferData.glsl"
#include "/techniques/HiZCheck.glsl"

#define SPD_CHANNELS 3
#define SPD_OP 2
#include "/techniques/ffx/spd/SPD.comp.glsl"

layout(rgb10_a2) uniform restrict image2D uimg_rgb10_a2;
const vec2 workGroupsRender = vec2(0.25, 0.25);

vec3 spd_loadInput(ivec2 texelPos) {
    return transient_geomViewNormal_fetch(texelPos).xyz * 2.0 - 1.0;
}

vec3 spd_loadOutput(ivec2 texelPos, uint level) {
    return vec3(0.0);
}

void spd_storeOutput(ivec2 texelPos, uint level, vec3 value) {
    ivec2 mipSize = global_mipmapSizesI[level];
    ivec2 mipOffset = ivec2(global_mipmapSizePrefixes[level - 1].x - uval_mainImageSizeI.x, 0);
    ivec2 storePos = mipOffset + texelPos;
    if (all(lessThanEqual(texelPos, mipSize))) {
        transient_geomNormalMip_store(storePos, vec4(value * 0.5 + 0.5, 0.0));
    }
}

uint spd_mipCount() {
    return 4u;
}