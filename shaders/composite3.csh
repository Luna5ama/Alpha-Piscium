#version 460 compatibility

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable

#include "/atmosphere/Common.glsl"
#include "/general/Lighting.glsl"
#include "/atmosphere/SunMoon.glsl"
#include "/util/Morton.glsl"
#include "/util/FullScreenComp.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_gbufferViewZ;
uniform usampler2D usam_gbufferData;
uniform usampler2D usam_temp7;

layout(rgba8) uniform writeonly image2D uimg_temp5;

vec2 texel2Screen(ivec2 texelPos) {
    return (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
}

vec4 compShadow(ivec2 texelPos, float viewZ) {
    vec2 screenPos = texel2Screen(texelPos);
    gbuffer_unpack(texelFetch(usam_gbufferData, texelPos, 0), gData);
    Material material = material_decode(gData);
    lighting_init(coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse), texelPos);
    return vec4(calcShadow(material.sss), 1.0);
}

void imageStore2x2(ivec2 texelPos1x1, vec4 outputColor) {
    imageStore(uimg_temp5, texelPos1x1, outputColor);
    imageStore(uimg_temp5, texelPos1x1 + ivec2(0, 1), outputColor);
    imageStore(uimg_temp5, texelPos1x1 + ivec2(1, 0), outputColor);
    imageStore(uimg_temp5, texelPos1x1 + ivec2(1, 1), outputColor);
}

void vrs2x2(ivec2 texelPos2x2) {
    ivec2 texelPos1x1 = texelPos2x2 << 1;

    if (all(lessThan(texelPos1x1, global_mainImageSizeI))) {
        vec2 quadCenterScreenPos = vec2(texelPos1x1 + 1) * global_mainImageSizeRcp;

        vec4 vrsWeight2x2 = unpackUnorm4x8(texelFetch(usam_temp7, texelPos2x2, 0).r);
        float weight2x2 = dot(vrsWeight2x2, vec4(0.5, 0.5, 0.0, 0.0));

        vec4 viewZs = textureGather(usam_gbufferViewZ, quadCenterScreenPos, 0);
        uint bitFlag = uint(weight2x2 > 0.9);
        bitFlag &= uint(all(notEqual(viewZs, vec4(-65536.0))));
        bool bitFlagBool = bool(bitFlag);

        if (subgroupAll(bitFlagBool)) {
            ivec2 offset = ivec2(morton_8bDecode((gl_LocalInvocationIndex + frameCounter) & 3u));
            ivec2 shadingTexelPos = texelPos1x1 + offset;
            float viewZ = texelFetch(usam_gbufferViewZ, shadingTexelPos, 0).r;

            vec4 result = compShadow(shadingTexelPos, viewZ);
            float lum = colors_srgbLuma(result.rgb);
            uint lumFlag = uint(lum < 0.05) | uint(lum > 0.95);
            bool lumFlagBool = bool(lumFlag);
            if (subgroupAll(lumFlagBool)) {
                imageStore2x2(texelPos1x1, result);
                return;
            }
        }

        ivec2 shadingTexelPos;
        vec4 temp5Out;

        shadingTexelPos = texelPos1x1;
        temp5Out = compShadow(shadingTexelPos, viewZs.w);
        imageStore(uimg_temp5, shadingTexelPos, temp5Out);

        shadingTexelPos = texelPos1x1 + ivec2(1, 0);
        temp5Out = compShadow(shadingTexelPos, viewZs.z);
        imageStore(uimg_temp5, shadingTexelPos, temp5Out);

        shadingTexelPos = texelPos1x1 + ivec2(0, 1);
        temp5Out = compShadow(shadingTexelPos, viewZs.x);
        imageStore(uimg_temp5, shadingTexelPos, temp5Out);

        shadingTexelPos = texelPos1x1 + ivec2(1, 1);
        temp5Out = compShadow(shadingTexelPos, viewZs.y);
        imageStore(uimg_temp5, shadingTexelPos, temp5Out);
    }
}

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;

    ivec2 texelPos = ivec2(mortonGlobalPosU);
    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    vec4 outputColor = compShadow(texelPos, viewZ);
    imageStore(uimg_temp5, texelPos, outputColor);
}