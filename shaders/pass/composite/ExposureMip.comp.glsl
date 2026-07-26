#define GLOBAL_DATA_MODIFIER buffer

#extension GL_KHR_shader_subgroup_ballot : enable

#include "/util/Colors2.glsl"
#include "/util/Math.glsl"

#define SPD_CHANNELS 4
#define SPD_OP 3
#include "/techniques/ffx/spd/SPD.comp.glsl"

layout(rgba32ui) uniform coherent uimage2D uimg_rgba32ui;
const vec2 workGroupsRender = vec2(POST_PROCESS_SCALE_QUARTER, POST_PROCESS_SCALE_QUARTER);

shared ivec2 shared_mipTile6;

vec4 spd_loadInput(ivec2 texelPos, uint slice) {
    vec4 result = vec4(0.0);
    if (all(lessThan(texelPos, POST_PROCESS_IMAGE_SIZE_I))) {
        vec4 weightData = transient_exposureWeights_fetch(renderScale_postToMainTexel(texelPos));
        float weight = weightData.x * weightData.y;
        vec3 color = texelFetch(usam_main, texelPos, 0).rgb;
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
        shared_mipTile6 = max(ivec2(ceil(ldexp(POST_PROCESS_IMAGE_SIZE, ivec2(-6)))), ivec2(1));
    }
    barrier();
}
