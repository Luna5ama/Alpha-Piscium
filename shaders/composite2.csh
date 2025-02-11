#version 460 compatibility

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);

#include "util/FullScreenComp.glsl"

#include "_Util.glsl"
#include "util/Morton.glsl"
#include "atmosphere/Common.glsl"
#include "general/Lighting.glsl"
#include "atmosphere/SunMoon.glsl"

uniform sampler2D usam_gbufferViewZ;
uniform usampler2D usam_gbufferData;
uniform sampler2D usam_temp7;

layout(rgba8) uniform writeonly image2D uimg_temp5;

vec2 texel2Screen(ivec2 texelPos) {
    return (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
}

vec4 compShadow(ivec2 texelPos, float viewZ) {
    vec2 screenPos = texel2Screen(texelPos);
    gbuffer_unpack(texelFetch(usam_gbufferData, texelPos, 0), gData);
    Material material = material_decode(gData);
    lighting_init(coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse));
    return vec4(calcShadow(material.sss), 1.0);
}

void main() {
    ivec2 fullResTexelPos = texelPos << 1;
    vec2 quadCenterScreenPos = vec2(fullResTexelPos + 1) * global_mainImageSizeRcp;

    uint bitFlag = uint(all(lessThan(fullResTexelPos, global_mainImageSizeI)));
    vec4 viewZs = textureGather(usam_gbufferViewZ, quadCenterScreenPos, 0);
    bitFlag &= uint(any(notEqual(viewZs, vec4(-65536.0))));

    uint workGroupIndex = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    uint threadIndex = (workGroupIndex << 8) | gl_LocalInvocationIndex.x;

    if (bool(bitFlag)) {
        vec4 vrsWeights = texelFetch(usam_temp7, texelPos, 0);
        float weight = dot(vrsWeights, vec4(0.5, 0.5, 0.0, 0.0));

        if (weight > 0.9) {
            float viewZ = dot(viewZs, vec4(0.25));
            ivec2 offset = ivec2(morton_8bDecode((threadIndex + frameCounter) & 3u));
            ivec2 shadingTexelPos = fullResTexelPos + offset;

            vec4 temp5Out = compShadow(shadingTexelPos, viewZ);
            imageStore(uimg_temp5, fullResTexelPos, temp5Out);
            imageStore(uimg_temp5, fullResTexelPos + ivec2(0, 1), temp5Out);
            imageStore(uimg_temp5, fullResTexelPos + ivec2(1, 0), temp5Out);
            imageStore(uimg_temp5, fullResTexelPos + ivec2(1, 1), temp5Out);
        } else {
            ivec2 shadingTexelPos;
            vec4 temp5Out;
            shadingTexelPos = fullResTexelPos;

            temp5Out = compShadow(shadingTexelPos, viewZs.w);
            imageStore(uimg_temp5, shadingTexelPos, temp5Out);
            shadingTexelPos = fullResTexelPos + ivec2(1, 0);
            temp5Out = compShadow(shadingTexelPos, viewZs.z);
            imageStore(uimg_temp5, shadingTexelPos, temp5Out);
            shadingTexelPos = fullResTexelPos + ivec2(0, 1);
            temp5Out = compShadow(shadingTexelPos, viewZs.x);
            imageStore(uimg_temp5, shadingTexelPos, temp5Out);
            shadingTexelPos = fullResTexelPos + ivec2(1, 1);
            temp5Out = compShadow(shadingTexelPos, viewZs.y);
            imageStore(uimg_temp5, shadingTexelPos, temp5Out);
        }
    }
}