#version 460 compatibility

#include "/denoiser/Update.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
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
vec2 sampleTexel, float baseWeight, vec3 centerPos, vec3 centerNormal,
inout vec3 colorSum, inout float weightSum
) {
    vec2 pixelPos = sampleTexel - 0.5;
    vec2 originPixelPos = floor(pixelPos);
    vec2 gatherTexelPos = originPixelPos + 1.0;
    vec2 originScreenPos = (gatherTexelPos * 2.0) * global_mainImageSizeRcp;
    vec2 gatherUV = gatherTexelPos * global_mainImageSizeRcp;
    vec2 gatherUV2Y = gatherUV * vec2(1.0, 0.5);
    vec2 bilinearWeightXY = pixelPos - originPixelPos;

    vec4 bilinearWeightGather;
    bilinearWeightGather.yz = bilinearWeightXY.xx;
    bilinearWeightGather.xw = 1.0 - bilinearWeightXY.xx;
    bilinearWeightGather.xy *= bilinearWeightXY.yy;
    bilinearWeightGather.zw *= 1.0 - bilinearWeightXY.yy;

    vec4 bilateralWeightGather = bilinearWeightGather * baseWeight;

    uvec4 prevNs = textureGather(usam_packedZN, gatherUV2Y, 0);
    vec4 prevZs = uintBitsToFloat(textureGather(usam_packedZN, gatherUV2Y, 1));
    float a = 0.000001 * max(abs(centerPos.z), 0.1);
    bilateralWeightGather.x *= computeGeometryWeight(centerPos, centerNormal, prevZs.x, prevNs.x, global_mainImageSizeRcp * vec2(-1.0, 1.0) + originScreenPos, a);
    bilateralWeightGather.y *= computeGeometryWeight(centerPos, centerNormal, prevZs.y, prevNs.y, global_mainImageSizeRcp * vec2(1.0, 1.0) + originScreenPos, a);
    bilateralWeightGather.z *= computeGeometryWeight(centerPos, centerNormal, prevZs.z, prevNs.z, global_mainImageSizeRcp * vec2(1.0, -1.0) + originScreenPos, a);
    bilateralWeightGather.w *= computeGeometryWeight(centerPos, centerNormal, prevZs.w, prevNs.w, global_mainImageSizeRcp * vec2(-1.0, -1.0) + originScreenPos, a);

    weightSum += bilateralWeightGather.x + bilateralWeightGather.y + bilateralWeightGather.z + bilateralWeightGather.w;

    vec4 colorRs = textureGather(usam_temp1, gatherUV, 0);
    colorSum.r += dot(bilateralWeightGather, colorRs);
    vec4 colorGs = textureGather(usam_temp1, gatherUV, 1);
    colorSum.g += dot(bilateralWeightGather, colorGs);
    vec4 colorBs = textureGather(usam_temp1, gatherUV, 2);
    colorSum.b += dot(bilateralWeightGather, colorBs);
}

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = vec4(0.0);

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        if (viewZ != -65536.0) {
            vec2 texelPos2x2F = vec2(texelPos) * 0.5;

            vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse);

            GBufferData gData;
            gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos, 0), gData);

            vec3 colorSum = vec3(0.0);
            float weightSum = 0.0;

            bilateralBilinearSample(texelPos2x2F + vec2(-1.0, 0.0), 1.0, viewPos, gData.geometryNormal, colorSum, weightSum);
            bilateralBilinearSample(texelPos2x2F + vec2(1.0, 0.0), 1.0, viewPos, gData.geometryNormal, colorSum, weightSum);
            bilateralBilinearSample(texelPos2x2F + vec2(0.0, -1.0), 1.0, viewPos, gData.geometryNormal, colorSum, weightSum);
            bilateralBilinearSample(texelPos2x2F + vec2(0.0, 1.0), 1.0, viewPos, gData.geometryNormal, colorSum, weightSum);

            if (weightSum > 0.01) {
                colorSum /= weightSum;
                outputColor.rgb = colorSum;
            }
        }

        uvec4 packedData = texelFetch(usam_tempRGBA32UI, texelPos, 0);

        vec4 prevColorHLen = vec4(unpackHalf2x16(packedData.x), unpackHalf2x16(packedData.y));
        vec2 prevMoments = unpackHalf2x16(packedData.z);

        float newHLen;
        vec2 newMoments;
        vec4 filterInput;
        gi_update(outputColor.rgb, prevColorHLen, prevMoments, newHLen, newMoments, filterInput);
        filterInput.rgb = dither_fp16(filterInput.rgb, rand_IGN(texelPos, frameCounter));
        imageStore(uimg_temp2, texelPos, filterInput);

        imageStore(uimg_temp6, texelPos, vec4(pow2(linearStep(0.0, 32.0, newHLen))));

        uvec4 packedOutData;
        svgf_pack(packedData, filterInput.rgb, vec3(0.0), newMoments, newHLen);
        imageStore(uimg_svgfHistory, svgf_texelPos1(texelPos), packedData);
    }
}