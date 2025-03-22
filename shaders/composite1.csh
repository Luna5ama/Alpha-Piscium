#version 460 compatibility

#extension GL_KHR_shader_subgroup_basic : enable

#include "/denoiser/Reproject.glsl"
#include "/util/NZPacking.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);

uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferData8UN;
uniform sampler2D usam_gbufferViewZ;
uniform usampler2D usam_packedNZ;
uniform usampler2D usam_svgfHistory;
uniform sampler2D usam_temp7;

layout(rgba16f) uniform writeonly image2D uimg_temp1;
layout(rgba16f) uniform writeonly image2D uimg_temp2;

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 3;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos2x2 = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos2x2, global_mipmapSizesI[1]))) {
        ivec2 texelPos1x1 = texelPos2x2 << 1;
        vec2 screenPos = (vec2(texelPos2x2) + 0.5) * global_mipmapSizesRcp[1];

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos1x1, 0).r;

        if (viewZ != -65536.0) {
            GBufferData gData;
            gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos1x1, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData8UN, texelPos1x1, 0), gData);

            Material material = material_decode(gData);

            {
                vec4 temp1Out = vec4(0.0);
                temp1Out.rgb = gData.normal;
                temp1Out.a = gData.lmCoord.y;
                imageStore(uimg_temp1, texelPos2x2 + ivec2(global_mipmapSizesI[1].x, 0), temp1Out);
            }

            vec4 prevColorHLen;
            vec2 prevMoments;

            gi_reproject(
                usam_svgfHistory, usam_packedNZ,
                screenPos, viewZ, gData.normal, gData.isHand,
                prevColorHLen, prevMoments
            );

            imageStore(uimg_temp1, texelPos2x2 + ivec2(0, global_mipmapSizesI[1].y), prevColorHLen);
            imageStore(uimg_temp1, texelPos2x2 + global_mipmapSizesI[1], vec4(prevMoments, 0.0, 0.0));

            {
                vec4 ssgiOut = vec4(0.0);
                if (gData.materialID == 65534u) {
                    ssgiOut = vec4(0.0);
                } else {
                    float multiBounceV = SETTING_VBGI_GI_MB * RCP_PI;
                    ssgiOut.rgb = multiBounceV * max(prevColorHLen.rgb, 0.0) * material.albedo;
                    ssgiOut.a = float(any(greaterThan(material.emissive, vec3(0.0))));
                }
                imageStore(uimg_temp1, texelPos2x2, ssgiOut);
            }
        }
    }
}