#version 460 compatibility

#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp3;
uniform usampler2D usam_packedNZ;
uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_translucentColor;

layout(rgba16f) writeonly uniform image2D uimg_temp4;

float normalWeight(vec3 centerNormal, uint packedNormal) {
    vec3 sampleNormal = coords_octDecode11(unpackSnorm2x16(packedNormal));
    float sdot = saturate(dot(centerNormal, sampleNormal));
    return pow(sdot, 64.0);
}

float viewZWeight(float centerViewZ, float sampleViewZ, float phiZ) {
    return phiZ / (phiZ + pow2(centerViewZ - sampleViewZ));
}

void bilateralBilinearSample(
vec2 sampleTexel, float baseWeight, float centerViewZ, vec3 centerWorldNormal,
inout vec3 colorSum, inout float weightSum
) {
    vec2 pixelPos = sampleTexel - 0.5;
    vec2 originPixelPos = floor(pixelPos);
    vec2 gatherUV = (originPixelPos + 1.0) * global_mainImageSizeRcp;
    vec2 gatherUVHalf = (originPixelPos + 1.0) * global_mipmapSizesRcp[1];
    vec2 bilinearWeightXY = pixelPos - originPixelPos;

    vec4 bilinearWeightGather;
    bilinearWeightGather.yz = bilinearWeightXY.xx;
    bilinearWeightGather.xw = 1.0 - bilinearWeightXY.xx;
    bilinearWeightGather.xy *= bilinearWeightXY.yy;
    bilinearWeightGather.zw *= 1.0 - bilinearWeightXY.yy;

    vec4 bilateralWeightGather = bilinearWeightGather * baseWeight;

    uvec4 prevNs = textureGather(usam_packedNZ, gatherUVHalf, 0);
    bilateralWeightGather.x *= normalWeight(centerWorldNormal, prevNs.x);
    bilateralWeightGather.y *= normalWeight(centerWorldNormal, prevNs.y);
    bilateralWeightGather.z *= normalWeight(centerWorldNormal, prevNs.z);
    bilateralWeightGather.w *= normalWeight(centerWorldNormal, prevNs.w);

    float phiZ = max(0.01 * pow2(centerViewZ), 0.5);
    vec4 prevViewZs = uintBitsToFloat(textureGather(usam_packedNZ, gatherUVHalf, 1));
    bilateralWeightGather.x *= viewZWeight(centerViewZ, prevViewZs.x, phiZ);
    bilateralWeightGather.y *= viewZWeight(centerViewZ, prevViewZs.y, phiZ);
    bilateralWeightGather.z *= viewZWeight(centerViewZ, prevViewZs.z, phiZ);
    bilateralWeightGather.w *= viewZWeight(centerViewZ, prevViewZs.w, phiZ);

    weightSum += bilateralWeightGather.x + bilateralWeightGather.y + bilateralWeightGather.z + bilateralWeightGather.w;

    vec4 colorRs = textureGather(usam_temp3, gatherUV, 0);
    colorSum.r += dot(bilateralWeightGather, colorRs);
    vec4 colorGs = textureGather(usam_temp3, gatherUV, 1);
    colorSum.g += dot(bilateralWeightGather, colorGs);
    vec4 colorBs = textureGather(usam_temp3, gatherUV, 2);
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
            vec3 scenePos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;

            GBufferData gData;
            gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos, 0), gData);
            vec3 worldNormal = mat3(gbufferModelViewInverse) * gData.normal;


            vec3 colorSum = vec3(0.0);
            float weightSum = 0.0;

            bilateralBilinearSample(texelPos2x2F, 8.0, viewZ, worldNormal, colorSum, weightSum);
            bilateralBilinearSample(texelPos2x2F + vec2(-0.5, -0.5), 1.0, viewZ, worldNormal, colorSum, weightSum);
            bilateralBilinearSample(texelPos2x2F + vec2(-0.5, 0.5), 1.0, viewZ, worldNormal, colorSum, weightSum);
            bilateralBilinearSample(texelPos2x2F + vec2(0.5, -0.5), 1.0, viewZ, worldNormal, colorSum, weightSum);
            bilateralBilinearSample(texelPos2x2F + vec2(0.5, 0.5), 1.0, viewZ, worldNormal, colorSum, weightSum);
            
            if (weightSum > 0.0001) {
                colorSum /= weightSum;
                outputColor.rgb = colorSum;
            }
        }

        imageStore(uimg_temp4, texelPos, outputColor);
    }
}