#define GLOBAL_DATA_MODIFIER buffer

#extension GL_KHR_shader_subgroup_ballot : enable

#include "/util/Colors2.glsl"
#include "/util/Math.glsl"

#define SPD_CHANNELS 4
#define SPD_OP 3
#include "/techniques/ffx/spd/SPD.comp.glsl"

layout(rgba32ui) uniform coherent uimage2D uimg_rgba32ui;
const vec2 workGroupsRender = vec2(0.25, 0.25);

shared ivec2 shared_mipTile6;

vec4 spd_loadInput(ivec2 texelPos, uint slice) {
    vec4 result = vec4(0.0);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 weightData = transient_exposureWeights_fetch(texelPos);
        float weight = weightData.x * weightData.y;
        vec3 color = transient_taaOutput_fetch(texelPos).rgb;
        result = vec4(color * weight, weight);
    }
    return result;
}

vec4 spd_loadOutput(ivec2 texelPos, uint level, uint slice) {
    vec4 result = vec4(0.0);
    if (all(lessThan(texelPos, shared_mipTile6))) {
        result = uintBitsToFloat(transient_mainMipTemp_load(texelPos));
    }
    return result;
}

void spd_storeOutput(ivec2 texelPos, uint level, uint slice, vec4 value) {
    if (level == 6u) {
        if (all(lessThan(texelPos, shared_mipTile6))) {
            transient_mainMipTemp_store(texelPos, floatBitsToUint(value));
        }
    }
    if (level == 12u) {
        vec3 finalColor = value.rgb * safeRcp(value.a);
        float luma = colors2_colorspaces_luma(COLORS2_OUTPUT_COLORSPACE, finalColor);
        global_aeData.screenAvgLum = vec4(finalColor, luma);
    }
}

uint spd_mipCount() {
    return 12u;
}

void spd_init() {
    if (gl_LocalInvocationIndex < 1u) {
        shared_mipTile6 = global_mipmapTileCeil[6u].zw;
    }
    barrier();
}

