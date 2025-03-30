#version 460 compatibility

#extension GL_KHR_shader_subgroup_basic : enable

#include "/denoiser/Reproject.glsl"
#include "/util/NZPacking.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Morton.glsl"
#include "/rtwsm/Backward.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferData8UN;
uniform sampler2D usam_gbufferViewZ;
uniform usampler2D usam_svgfHistory;

layout(rgba32ui) uniform writeonly uimage2D uimg_tempRGBA32UI;
layout(rg32ui) uniform writeonly uimage2D uimg_tempRG32UI;
layout(rgba8) uniform writeonly image2D uimg_temp6;

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 3;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos1x1 = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos1x1, global_mainImageSizeI))) {
        vec2 screenPos1x1 = (vec2(texelPos1x1) + 0.5) * global_mipmapSizesRcp[1];

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos1x1, 0).r;

        if (viewZ != -65536.0) {
            GBufferData gData;
            gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos1x1, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData8UN, texelPos1x1, 0), gData);

            Material material = material_decode(gData);

//            imageStore(uimg_temp6, texelPos1x1, vec4(gData.normal * 0.5 + 0.5, 0.0));

            vec3 prevDiffuse;
            vec2 prevMoments;
            float prevHLen;

            gi_reproject(
                usam_svgfHistory,
                screenPos1x1, viewZ, gData.normal, gData.isHand,
                prevDiffuse, prevMoments, prevHLen
            );

            uvec4 temp32UIOut = uvec4(
                packHalf2x16(prevDiffuse.rg),
                packHalf2x16(vec2(prevDiffuse.b, 0.0)),
                packHalf2x16(prevMoments),
                floatBitsToUint(prevHLen)
            );

            imageStore(uimg_tempRGBA32UI, texelPos1x1, temp32UIOut);

            if ((threadIdx & 3u) == 0u) {
                ivec2 texelPos2x2 = texelPos1x1 >> 1;

                {
                    uvec4 tempRG32UIOut = uvec4(0u);
                    nzpacking_pack(tempRG32UIOut.xy, gData.normal, viewZ);
                    imageStore(uimg_tempRG32UI, texelPos2x2, tempRG32UIOut);
                }

                {
                    vec4 ssgiOut = vec4(0.0);
                    ssgiOut.a = gData.lmCoord.y;
                    if (gData.materialID != 65534u) {
                        float multiBounceV = SETTING_VBGI_GI_MB * RCP_PI;
                        ssgiOut.rgb = multiBounceV * max(prevDiffuse, 0.0) * material.albedo;
                        ssgiOut.a *= mix(-1.0, 1.0, any(greaterThan(material.emissive, vec3(0.0))));
                    }
                    uvec4 tempRG32UIOut = uvec4(0u);
                    tempRG32UIOut.x = packHalf2x16(ssgiOut.rg);
                    tempRG32UIOut.y = packHalf2x16(ssgiOut.ba);
                    imageStore(uimg_tempRG32UI, texelPos2x2 + ivec2(0, global_mipmapSizesI[1].y), tempRG32UIOut);
                }
            }

            rtwsm_backward(texelPos1x1, viewZ, gData.geometryNormal);
        } else {
            uvec4 temp32UIOut = uvec4(0u);
            imageStore(uimg_tempRGBA32UI, texelPos1x1, temp32UIOut);

            if ((threadIdx & 3u) == 0u) {
                ivec2 texelPos2x2 = texelPos1x1 >> 1;
                {
                    uvec4 tempRG32UIOut = uvec4(0u);
                    tempRG32UIOut.y = floatBitsToUint(viewZ);
                    imageStore(uimg_tempRG32UI, texelPos2x2, tempRG32UIOut);
                }
                {
                    uvec4 tempRG32UIOut = uvec4(0u);
                    imageStore(uimg_tempRG32UI, texelPos2x2 + ivec2(0, global_mipmapSizesI[1].y), tempRG32UIOut);
                }
            }
        }
    }
}