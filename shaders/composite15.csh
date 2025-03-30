#version 460 compatibility

#include "/denoiser/Atrous.glsl"

layout(local_size_x = 1, local_size_y = 128) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp1;
uniform sampler2D usam_temp6;
uniform usampler2D usam_packedZN;
layout(rgba16f) uniform writeonly image2D uimg_temp2;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        float sigmaL = SETTING_DENOISER_FILTER_COLOR_STRICTNESS * texelFetch(usam_temp6, texelPos, 0).r;
        vec4 outputColor = svgf_atrous(usam_temp1, usam_packedZN, texelPos, ivec2(0, 2), sigmaL);
        imageStore(uimg_temp2, texelPos, outputColor);
    }
}