#version 460 compatibility

#include "/denoiser/Atrous.glsl"

layout(local_size_x = 1, local_size_y = 128) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);

uniform sampler2D usam_temp6;
uniform sampler2D usam_temp3;
uniform usampler2D usam_packedNZ;
layout(rgba16f) uniform writeonly image2D uimg_temp4;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, global_mipmapSizesI[1]))) {
        float hLen = texelFetch(usam_temp6, texelPos, 0).r;
        float sigmaL = SETTING_DENOISER_FILTER_COLOR_STRICTNESS * linearStep(0.0, 0.2, hLen);
        vec4 outputColor = svgf_atrous(usam_temp3, usam_packedNZ, texelPos, ivec2(0, 8), sigmaL);
        imageStore(uimg_temp4, texelPos, outputColor);
    }
}