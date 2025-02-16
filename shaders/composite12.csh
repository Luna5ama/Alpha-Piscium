#version 460 compatibility

#include "/svgf/Common.glsl"
#include "/util/Math.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp4;
uniform sampler2D usam_temp6;
uniform sampler2D usam_projReject;
uniform sampler2D usam_gbufferViewZ;

layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba16f) uniform writeonly image2D uimg_svgfHistoryColor;
layout(rgba16f) uniform restrict image2D uimg_ssvbil;

layout(rgba16f) uniform writeonly image2D uimg_temp2;

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
        vec4 filterOutput = svgf_atrous(usam_temp4, texelPos);
        vec2 projReject = texelFetch(usam_projReject, texelPos, 0).rg;

        float frustumTest = float(projReject.x > 0.0);
        float newPixel = float(projReject.y > 0.0);

        float hLen = texelFetch(usam_temp6, texelPos, 0).r * 255.0 + 1.0;
        hLen *= saturate(1.0 - frustumTest * 0.5);
        
        imageStore(uimg_temp3, texelPos, filterOutput);
        imageStore(uimg_svgfHistoryColor, texelPos, vec4(filterOutput.rgb, hLen));

        float ao = imageLoad(uimg_ssvbil, texelPos).a;
        vec3 gi = filterOutput.rgb;
        imageStore(uimg_ssvbil, texelPos, vec4(gi, ao));

        float debug = saturate((hLen - 1.0));
        if (texelFetch(usam_gbufferViewZ, texelPos, 0).r == -65536.0) {
            debug = 1.0;
        }
        imageStore(uimg_temp2, texelPos, vec4(debug));
    }
}