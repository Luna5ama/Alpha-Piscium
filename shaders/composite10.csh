#version 460 compatibility

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#define SSVBIL_SAMPLE_STEPS SETTING_VBGI_STEPS
#define SSVBIL_SAMPLE_SLICES SETTING_VBGI_SLICES
#include "/post/gtvbgi/GTVBGI.glsl"
#include "/util/Morton.glsl"

uniform sampler2D usam_temp7;
layout(rgba16f) uniform writeonly image2D uimg_ssvbil;

//void imageStore2x2(ivec2 texelPos1x1, vec4 outputColor) {
//    imageStore(uimg_ssvbil, texelPos1x1, outputColor);
//    imageStore(uimg_ssvbil, texelPos1x1 + ivec2(0, 1), outputColor);
//    imageStore(uimg_ssvbil, texelPos1x1 + ivec2(1, 0), outputColor);
//    imageStore(uimg_ssvbil, texelPos1x1 + ivec2(1, 1), outputColor);
//}
//
//void vrs2x2(ivec2 texelPos2x2) {
//    ivec2 texelPos1x1 = texelPos2x2 << 1;
//
//    if (all(lessThan(texelPos1x1, global_mainImageSizeI))) {
//        vec2 quadCenterScreenPos = vec2(texelPos1x1 + 1) * global_mainImageSizeRcp;
//
//        vec4 vrsWeight2x2 = texelFetch(usam_temp7, texelPos2x2, 0);
//        float weight2x2 = dot(vrsWeight2x2, vec4(0.5, 0.5, 0.0, 0.0));
//
//        vec4 viewZs = textureGather(usam_gbufferViewZ, quadCenterScreenPos, 0);
//        uint bitFlag = uint(weight2x2 > 0.25);
//        bitFlag &= uint(all(notEqual(viewZs, vec4(-65536.0))));
//        bool bitFlagBool = bool(bitFlag);
//
//        if (bitFlagBool) {
//            ivec2 offset = ivec2(morton_8bDecode((gl_LocalInvocationIndex + frameCounter) & 3u));
//            ivec2 shadingTexelPos = texelPos1x1 + offset;
//
//            vec4 result = gtvbgi(shadingTexelPos);
//            imageStore(uimg_ssvbil, shadingTexelPos, result);
//            return;
//        }
//
//        ivec2 shadingTexelPos;
//        vec4 giOut;
//
//        shadingTexelPos = texelPos1x1;
//        giOut = gtvbgi(shadingTexelPos);
//        imageStore(uimg_ssvbil, shadingTexelPos, giOut);
//
//        shadingTexelPos = texelPos1x1 + ivec2(1, 0);
//        giOut = gtvbgi(shadingTexelPos);
//        imageStore(uimg_ssvbil, shadingTexelPos, giOut);
//
//        shadingTexelPos = texelPos1x1 + ivec2(0, 1);
//        giOut = gtvbgi(shadingTexelPos);
//        imageStore(uimg_ssvbil, shadingTexelPos, giOut);
//
//        shadingTexelPos = texelPos1x1 + ivec2(1, 1);
//        giOut = gtvbgi(shadingTexelPos);
//        imageStore(uimg_ssvbil, shadingTexelPos, giOut);
//    }
//}

void main() {
//    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
//    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
//    uvec2 mortonPos = morton_8bDecode(threadIdx);
//    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
//
//    ivec2 texelPos2x2 = ivec2(mortonGlobalPosU);
//    vrs2x2(texelPos2x2);

    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 giOut = gtvbgi(texelPos);
        imageStore(uimg_ssvbil, texelPos, giOut);
    }
}