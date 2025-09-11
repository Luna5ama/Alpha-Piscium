#version 460 compatibility

#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"
#include "/util/Colors.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_main;
layout(rgba16f) uniform writeonly image2D uimg_csrgba16f;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);

        vec4 translucentColor = texelFetch(usam_translucentColor, texelPos, 0);
        vec4 translucentData = texelFetch(usam_translucentData, texelPos, 0);


        vec3 tScatteringCoeff = mix(vec3(0.0), translucentColor.rgb / translucentColor.a, translucentColor.a > 0.0);

        vec3 tTransmittanceCoeff = saturate(translucentData.rgb);
        vec3 tAbsorptionCoeff = -log(tTransmittanceCoeff);
        vec3 tExtionctionCoeff = tAbsorptionCoeff + tScatteringCoeff;
        vec3 tOpticalDepth = tExtionctionCoeff;
        vec3 sampleTransmittance = exp(-tOpticalDepth);

        outputColor.rgb *= sampleTransmittance;

        ivec2 farDepthTexelPos = texelPos;
        ivec2 nearDepthTexelPos = texelPos;
        farDepthTexelPos.y += global_mainImageSizeI.y;
        nearDepthTexelPos += global_mainImageSizeI;

        float startViewZ = -texelFetch(usam_translucentDepthLayers, nearDepthTexelPos, 0).r;
        float endViewZ = -texelFetch(usam_translucentDepthLayers, farDepthTexelPos, 0).r;

        imageStore(uimg_main, texelPos, outputColor);

        #ifdef SETTING_DOF
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        outputColor.a = abs(viewZ);
        imageStore(uimg_csrgba16f, csrgba16f_temp1_texelToTexel(texelPos), outputColor);
        #endif
    }
}