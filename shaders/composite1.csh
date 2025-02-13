#version 460 compatibility

#include "/general/NDPacking.glsl"
#include "/svgf/Reproject.glsl"
#include "/util/GBuffers.glsl"
#include "/util/Material.glsl"
#include "/util/FullScreenComp.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_gbufferViewZ;
uniform usampler2D usam_gbufferData;
uniform usampler2D usam_prevNZ;
uniform sampler2D usam_svgfHistoryColor;
uniform sampler2D usam_svgfHistoryMoments;

layout(rg8) uniform writeonly image2D uimg_projReject;
layout(rgba16f) uniform writeonly image2D uimg_temp1;
layout(rgba16f) uniform writeonly image2D uimg_temp2;
layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba16f) uniform writeonly image2D uimg_temp4;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        if (viewZ != -65536.0) {
            GBufferData gData;
            gbuffer_unpack(texelFetch(usam_gbufferData, texelPos, 0), gData);

            Material material = material_decode(gData);

            {
                vec4 temp1Out = vec4(0.0);
                temp1Out.rgb = gData.normal;
                temp1Out.a = float(any(greaterThan(material.emissive, vec3(0.0))));
                imageStore(uimg_temp1, texelPos, temp1Out);
            }

            {
                vec3 viewCoord = coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse);

                vec2 projRejectOut;
                ndpacking_updateProjReject(usam_prevNZ, texelPos, screenPos, gData.normal, viewCoord, projRejectOut);
                imageStore(uimg_projReject, texelPos, vec4(projRejectOut, 0.0, 0.0));
            }

            vec4 prevColorHLen;
            vec2 prevMoments;

            svgf_reproject(
                usam_svgfHistoryColor, usam_svgfHistoryMoments, usam_prevNZ,
                screenPos, viewZ, gData.normal, float(gData.isHand),
                prevColorHLen, prevMoments
            );

            imageStore(uimg_temp3, texelPos, prevColorHLen);
            imageStore(uimg_temp4, texelPos, vec4(prevMoments, 0.0, 0.0));

            {
                vec4 ssgiOut = vec4(0.0);
                if (gData.materialID == 65534u) {
                    ssgiOut = vec4(0.0, 0.0, 0.0, 0.0);
                } else {
                    float multiBounceV = SETTING_SSVBIL_GI_MB * 2.0 * RCP_PI;
                    ssgiOut.rgb = multiBounceV * max(prevColorHLen.rgb, 0.0) * material.albedo;
                }
                imageStore(uimg_temp2, texelPos, ssgiOut);
            }
        }
    }
}