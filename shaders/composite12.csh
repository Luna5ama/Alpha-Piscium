#version 460 compatibility

#include "svgf/Common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp4;
uniform sampler2D usam_temp5;

layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba16f) uniform writeonly image2D uimg_svgfHistoryColor;

#define ATROUS_STEP_SIZE 1

vec4 svgf_atrous(sampler2D filterInput, ivec2 texelPos) {
    vec4 colorSum = texelFetch(filterInput, texelPos, 0);
    float weightSum = 1.0;

    colorSum /= weightSum;
    return colorSum;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        float hLen = texelFetch(usam_temp5, texelPos, 0).r * 255.0 + 1.0;
        vec4 filterOutput = svgf_atrous(usam_temp4, texelPos);
        imageStore(uimg_temp3, texelPos, filterOutput);
        imageStore(uimg_svgfHistoryColor, texelPos, vec4(filterOutput.rgb, hLen));
    }
}