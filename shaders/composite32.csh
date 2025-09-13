#version 460 compatibility

#include "/techniques/SST.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Coords.glsl"
#include "/util/Colors.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_main;
layout(rgba16f) uniform writeonly image2D uimg_csrgba16f;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_temp1, texelPos, 0);

            ivec2 farDepthTexelPos = texelPos;
            ivec2 nearDepthTexelPos = texelPos;
            farDepthTexelPos.y += global_mainImageSizeI.y;
            nearDepthTexelPos += global_mainImageSizeI;
//
            float startViewZ = -texelFetch(usam_translucentDepthLayers, nearDepthTexelPos, 0).r;
//            float endViewZ = -texelFetch(usam_translucentDepthLayers, farDepthTexelPos, 0).r;
//            float startViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        if (startViewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, global_mainImageSizeRcp);
            vec3 startViewPos = coords_toViewCoord(screenPos, startViewZ, global_camProjInverse);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

            const float RIOR = 1.0 / 1.5;
            vec3 viewDir = normalize(-startViewPos);
                        vec3 refractDir = refract(-viewDir, gData.normal, RIOR);
//            vec3 refractDir = reflect(-viewDir, gData.geomNormal);

            SSTResult result = sst_trace(startViewPos, refractDir);
            if (result.hit) {
                vec2 coord = result.hitScreenPos.xy;
                outputColor = texture(usam_temp1, result.hitScreenPos.xy);
            }
        }

        vec4 translucentColor = texelFetch(usam_translucentColor, texelPos, 0);

        outputColor.rgb *= translucentColor.rgb;

        imageStore(uimg_main, texelPos, outputColor);

        #ifdef SETTING_DOF
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        outputColor.a = abs(viewZ);
        imageStore(uimg_csrgba16f, csrgba16f_temp1_texelToTexel(texelPos), outputColor);
        #endif
    }
}