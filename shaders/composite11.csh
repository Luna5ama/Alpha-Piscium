#version 460 compatibility

#include "/denoiser/Update.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
#include "/util/Interpo.glsl"
#include "/util/Material.glsl"
#include "/util/Dither.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp1;
uniform usampler2D usam_tempRGBA32UI;
uniform usampler2D usam_packedZN;
uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferViewZ;

layout(rgba16f) writeonly uniform image2D uimg_temp2;
layout(rgba8) uniform writeonly image2D uimg_temp6;
layout(rgba32ui) uniform writeonly uimage2D uimg_svgfHistory;

float computeGeometryWeight(vec3 centerPos, vec3 centerNormal, float sampleViewZ, uint sampleNormal, vec2 sampleScreenPos, float a) {
    vec3 sampleViewPos = coords_toViewCoord(sampleScreenPos, sampleViewZ, gbufferProjectionInverse);
    vec3 sampleViewGeomNormal = coords_octDecode11(unpackSnorm2x16(sampleNormal));

    float normalWeight = pow4(dot(centerNormal, sampleViewGeomNormal));

    vec3 posDiff = centerPos - sampleViewPos;
    float planeDist1 = pow2(dot(posDiff, centerNormal));
    float planeDist2 = pow2(dot(posDiff, sampleViewGeomNormal));
    float maxPlaneDist = max(planeDist1, planeDist2);
    float planeWeight = a / (a + maxPlaneDist);

    return planeWeight;
}

void bilateralBilinearSample(
vec2 gatherTexelPos, vec4 baseWeights,
vec3 centerPos, vec3 centerNormal,
inout vec3 colorSum, inout float weightSum
) {
    vec2 originScreenPos = (gatherTexelPos * 2.0) * global_mainImageSizeRcp;
    vec2 gatherUV = gatherTexelPos * global_mainImageSizeRcp;
    vec2 gatherUV2Y = gatherUV * vec2(1.0, 0.5);

    vec4 bilateralWeights = vec4(1.0);
    uvec4 prevNs = textureGather(usam_packedZN, gatherUV2Y, 0);
    vec4 prevZs = uintBitsToFloat(textureGather(usam_packedZN, gatherUV2Y, 1));
    float a = 0.000001 * max(abs(centerPos.z), 0.1);
    bilateralWeights.x *= computeGeometryWeight(centerPos, centerNormal, prevZs.x, prevNs.x, global_mainImageSizeRcp * vec2(-1.0, 1.0) + originScreenPos, a);
    bilateralWeights.y *= computeGeometryWeight(centerPos, centerNormal, prevZs.y, prevNs.y, global_mainImageSizeRcp * vec2(1.0, 1.0) + originScreenPos, a);
    bilateralWeights.z *= computeGeometryWeight(centerPos, centerNormal, prevZs.z, prevNs.z, global_mainImageSizeRcp * vec2(1.0, -1.0) + originScreenPos, a);
    bilateralWeights.w *= computeGeometryWeight(centerPos, centerNormal, prevZs.w, prevNs.w, global_mainImageSizeRcp * vec2(-1.0, -1.0) + originScreenPos, a);

    vec4 interpoWeights = baseWeights * bilateralWeights;
    weightSum += interpoWeights.x + interpoWeights.y + interpoWeights.z + interpoWeights.w;

    vec4 colorRs = textureGather(usam_temp1, gatherUV, 0);
    colorSum.r += dot(interpoWeights, colorRs);
    vec4 colorGs = textureGather(usam_temp1, gatherUV, 1);
    colorSum.g += dot(interpoWeights, colorGs);
    vec4 colorBs = textureGather(usam_temp1, gatherUV, 2);
    colorSum.b += dot(interpoWeights, colorBs);
}

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 currColor = vec4(0.0);

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        if (viewZ != -65536.0) {
            vec2 texelPos2x2F = vec2(texelPos) * 0.5;

            vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse);

            GBufferData gData;
            gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos, 0), gData);

            vec3 colorSum = vec3(0.0);
            float weightSum = 0.0;

            vec2 centerPixel = texelPos2x2F - 0.5;
            vec2 centerPixelOrigin = floor(centerPixel);
            vec2 gatherTexelPos = centerPixelOrigin + 1.0;
            vec2 pixelPosFract = centerPixel - centerPixelOrigin;

            vec4 weightX = interpo_bSplineWeights(pixelPosFract.x);
            vec4 weightY = interpo_bSplineWeights(pixelPosFract.y);

            vec2 bilinearWeights2 = pixelPosFract;
            vec4 blinearWeights4;
            blinearWeights4.yz = bilinearWeights2.xx;
            blinearWeights4.xw = 1.0 - bilinearWeights2.xx;
            blinearWeights4.xy *= bilinearWeights2.yy;
            blinearWeights4.zw *= 1.0 - bilinearWeights2.yy;

            vec4 sampleGatherWeights = weightX.xyyx * weightY.wwzz;
            sampleGatherWeights.z += blinearWeights4.x;
            bilateralBilinearSample(
                texelPos2x2F + vec2(-1.0, 1.0), sampleGatherWeights,
                viewPos, gData.geometryNormal,
                colorSum, weightSum
            );

            sampleGatherWeights = weightX.zwwz * weightY.wwzz;
            sampleGatherWeights.w += blinearWeights4.y;
            bilateralBilinearSample(
                texelPos2x2F + vec2(1.0, 1.0), sampleGatherWeights,
                viewPos, gData.geometryNormal,
                colorSum, weightSum
            );

            sampleGatherWeights = weightX.zwwz * weightY.yyxx;
            sampleGatherWeights.x += blinearWeights4.z;
            bilateralBilinearSample(
                texelPos2x2F + vec2(1.0, -1.0), sampleGatherWeights,
                viewPos, gData.geometryNormal,
                colorSum, weightSum
            );

            sampleGatherWeights = weightX.xyyx * weightY.yyxx;
            sampleGatherWeights.y += blinearWeights4.w;
            bilateralBilinearSample(
                texelPos2x2F + vec2(-1.0, -1.0), sampleGatherWeights,
                viewPos, gData.geometryNormal,
                colorSum, weightSum
            );

            if (weightSum > 0.01) {
                colorSum /= weightSum;
                currColor.rgb = colorSum;
            }
        }

        uvec4 packedData = texelFetch(usam_tempRGBA32UI, texelPos, 0);
        vec3 prevColor;
        vec3 prevFastColor;
        vec2 prevMoments;
        float prevHLen;
        svgf_unpack(packedData, prevColor, prevFastColor, prevMoments, prevHLen);

        vec3 newColor;
        vec3 newFastColor;
        vec2 newMoments;
        float newHLen;
        gi_update(
            currColor.rgb,
            prevColor, prevFastColor, prevMoments, prevHLen,
            newColor, newFastColor, newMoments, newHLen
        );

        float variance = max(newMoments.g - newMoments.r * newMoments.r, 0.0);
        vec4 filterInput = vec4(newColor, variance);
        filterInput = dither_fp16(filterInput, rand_IGN(texelPos, frameCounter));
        imageStore(uimg_temp2, texelPos, filterInput);

        imageStore(uimg_temp6, texelPos, vec4(linearStep(1.0, 128.0, newHLen)));

        uvec4 packedOutData = uvec4(0u);
        #ifdef SETTING_DENOISER
        svgf_packNoColor(packedOutData, newFastColor, newMoments, newHLen);
        #else
        svgf_pack(packedOutData, newColor, newFastColor, newMoments, newHLen);
        #endif
        imageStore(uimg_svgfHistory, svgf_texelPos1(texelPos), packedOutData);
    }
}