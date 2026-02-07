#version 460 compatibility
#define COMP 1

#include "/techniques/textile/CSR32F.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_main;
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_main, texelPos, 0);
        float viewZ = texelFetch(usam_gbufferSolidViewZ, texelPos, 0).r;

        ivec2 waterNearDepthTexelPos = csr32f_tile1_texelToTexel(texelPos);
        ivec2 translucentNearDepthTexelPos = csr32f_tile3_texelToTexel(texelPos);
        float waterStartViewZ = -texelFetch(usam_csr32f, waterNearDepthTexelPos, 0).r;
        float translucentStartViewZ = -texelFetch(usam_csr32f, translucentNearDepthTexelPos, 0).r;
        float startViewZ = max(translucentStartViewZ, waterStartViewZ);

        viewZ = max(startViewZ, viewZ);
        outputColor.a = abs(viewZ);
        transient_dofInput_store(texelPos, outputColor);
    }
}