#version 460 compatibility

#extension GL_KHR_shader_subgroup_clustered : enable

#include "../_Util.glsl"
#include "util/Morton.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_temp5;

layout(rgba32ui) uniform restrict uimage2D uimg_gbufferData;
layout(rgba8) uniform writeonly image2D uimg_temp7;

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    ivec2 clampedTexelPos = min(texelPos, global_mainImageSizeI - 1);
    vec3 albedo = texelFetch(usam_temp5, clampedTexelPos, 0).rgb;
    GBufferData gData;
    gbuffer_unpack(imageLoad(uimg_gbufferData, clampedTexelPos), gData);
    gData.albedo = albedo;
    float viewZ = texelFetch(usam_gbufferViewZ, clampedTexelPos, 0).r;

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        uvec4 packedData;
        gbuffer_pack(packedData, gData);
        imageStore(uimg_gbufferData, texelPos, packedData);

        vec4 normalViewZ;
        normalViewZ.xyz = viewZ == -65536.0 ? vec3(0.0) : gData.normal;
        normalViewZ.w = viewZ;
        vec4 avgNormalViewZ = subgroupClusteredAdd(normalViewZ, 4u) * 0.25;

        vec3 avgNormal = avgNormalViewZ.xyz;
        float normalWeight = pow2(dot(gData.normal, avgNormal));
        normalWeight = subgroupClusteredMul(normalWeight, 4u);

        float avgViewZ = avgNormalViewZ.w;
        float a = -0.01 * avgViewZ;
        float viewZWeight = a / (a + abs(viewZ - avgViewZ));
        viewZWeight = 1.0 - viewZWeight;
        viewZWeight = subgroupClusteredMul(viewZWeight, 4u);
        viewZWeight = 1.0 - viewZWeight;
        viewZWeight = pow2(viewZWeight);

        vec3 avgAlbedo = subgroupClusteredAdd(albedo, 4u) * 0.25;
        float albedoWeight = colors_srgbLuma(abs(avgAlbedo - albedo));
        const float albedoA = 0.2;
        albedoWeight = albedoA / (albedoA + albedoWeight);
        albedoWeight = subgroupClusteredMul(albedoWeight, 4u);

        float noPixelWeight = float(subgroupClusteredMin(viewZ, 4u) != -65536.0);

        vec4 vrsWeight2x2 = vec4(normalWeight, viewZWeight, albedoWeight, 1.0) * noPixelWeight;

        if ((threadIdx & 3u) == 0u) {
            imageStore(uimg_temp7, clampedTexelPos >> 1, vec4(vrsWeight2x2));
            vec4 vrsWeighr4x4 = subgroupClusteredMin(vrsWeight2x2, 16u);
            if ((threadIdx & 15u) == 0u) {
                ivec2 texelPos4x4 = clampedTexelPos >> 2;
                texelPos4x4.x += global_mipmapSizesI[1].x;
                imageStore(uimg_temp7, texelPos4x4, vrsWeighr4x4);
            }
        }
    }
}