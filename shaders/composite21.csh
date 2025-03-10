#version 460 compatibility

#include "/denoiser/Atrous.glsl"

layout(local_size_x = 1, local_size_y = 128) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);

uniform sampler2D usam_temp6;
uniform sampler2D usam_temp3;
uniform usampler2D usam_packedNZ;

layout(rgba16f) uniform restrict image2D uimg_ssvbil;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        float hLen = texelFetch(usam_temp6, texelPos, 0).r;
        float sigmaL = SETTING_DENOISER_FILTER_COLOR_STRICTNESS * pow4(linearStep(0.0, 0.25, hLen));
        vec4 outputColor = svgf_atrous(usam_temp3, usam_packedNZ, texelPos, ivec2(0, 16), sigmaL);
        imageStore(uimg_ssvbil, texelPos, vec4(outputColor.rgb, 1.0));
    }
}