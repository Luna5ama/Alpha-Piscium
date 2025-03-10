#version 460 compatibility

#include "/denoiser/Atrous.glsl"

layout(local_size_x = 128, local_size_y = 1) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp6;
uniform sampler2D usam_temp4;
uniform usampler2D usam_prevNZ;
layout(rgba16f) uniform writeonly image2D uimg_temp3;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        float hLen = texelFetch(usam_temp6, texelPos, 0).r;
        float sigmaL = SETTING_DENOISER_FILTER_COLOR_STRICTNESS * pow4(linearStep(0.0, 0.25, hLen));
        vec4 outputColor = svgf_atrous(usam_temp4, usam_prevNZ, texelPos, ivec2(2, 0), sigmaL);
        imageStore(uimg_temp3, texelPos, outputColor);
    }
}