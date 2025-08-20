#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#define HIZ_SUBGROUP_CHECK a

#include "/techniques/gtvbgi/Common.glsl"
#include "/util/NZPacking.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Morton.glsl"
#include "/techniques/HiZ.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "/techniques/svgf/Reproject.glsl"

layout(rgba32ui) uniform writeonly uimage2D uimg_tempRGBA32UI;
layout(rg32ui) uniform writeonly uimage2D uimg_packedZN;

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 3;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos1x1 = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos1x1, global_mainImageSizeI))) {
        ivec2 texelPos2x2 = texelPos1x1 >> 1;
        vec2 screenPos1x1 = (vec2(texelPos1x1) + 0.5) * global_mainImageSizeRcp;

        if (hiz_groupGroundCheckSubgroup(gl_WorkGroupID.xy, 3)) {
            float viewZ = texelFetch(usam_gbufferViewZ, texelPos1x1, 0).r;
            if (viewZ != -65536.0) {
                GBufferData gData = gbufferData_init();
                gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos1x1, 0), gData);
                gbufferData2_unpack(texelFetch(usam_gbufferData8UN, texelPos1x1, 0), gData);

                Material material = material_decode(gData);
                vec3 prevDiffuse;
                vec3 prevFastDiffuse;
                vec2 prevMoments;
                float prevHLen;

                gi_reproject(
                    screenPos1x1, viewZ, gData.normal, gData.geometryNormal, gData.isHand,
                    prevDiffuse, prevFastDiffuse, prevMoments, prevHLen
                );

                uvec4 temp32UIOut = uvec4(0u);
                svgf_pack(temp32UIOut, prevDiffuse, prevFastDiffuse, prevMoments, prevHLen);
                imageStore(uimg_tempRGBA32UI, texelPos1x1, temp32UIOut);

                uvec4 packedZNOut = uvec4(0u);
                nzpacking_pack(packedZNOut.xy, gData.normal, viewZ);

                uint ssgiOutWriteFlag = uint(vbgi_selectDownSampleInput(threadIdx));
                ssgiOutWriteFlag &= uint(all(lessThan(texelPos2x2, global_mipmapSizesI[1])));
                if (bool(ssgiOutWriteFlag)) {
                    imageStore(uimg_packedZN, texelPos2x2, packedZNOut);

                    {
                        vec4 ssgiOut = vec4(0.0);
                        ssgiOut.a = gData.lmCoord.y + 0.000001;
                        float multiBounceV = SETTING_VBGI_GI_MB;
                        float albedoLum = colors_sRGB_luma(material.albedo);
                        ssgiOut.rgb = multiBounceV * max(prevDiffuse, 0.0) * material.albedo;
                        ssgiOut.a *= mix(-1.0, 1.0, any(greaterThan(material.emissive, vec3(0.0))));
                        uvec4 tempRG32UIOut = uvec4(0u);
                        tempRG32UIOut.x = packHalf2x16(ssgiOut.rg);
                        tempRG32UIOut.y = packHalf2x16(ssgiOut.ba);
                        imageStore(uimg_packedZN, texelPos2x2 + ivec2(0, global_mipmapSizesI[1].y), tempRG32UIOut);
                    }
                }
                return;
            }
        }

        uvec4 temp32UIOut = uvec4(0u);
        imageStore(uimg_tempRGBA32UI, texelPos1x1, temp32UIOut);

        uvec4 packedZNOut = uvec4(0u);
        packedZNOut.y = floatBitsToUint(-65536.0);

        uint ssgiOutWriteFlag = uint(vbgi_selectDownSampleInput(threadIdx));
        ssgiOutWriteFlag &= uint(all(lessThan(texelPos2x2, global_mipmapSizesI[1])));
        if (bool(ssgiOutWriteFlag)) {
            imageStore(uimg_packedZN, texelPos2x2, packedZNOut);
            uvec4 tempRG32UIOut = uvec4(0u);
            imageStore(uimg_packedZN, texelPos2x2 + ivec2(0, global_mipmapSizesI[1].y), tempRG32UIOut);
        }
    }
}