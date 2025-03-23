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

void bilateralSample(
vec2 sampleTexel, float baseWeight, float centerViewZ, vec3 centerWorldNormal,
inout vec3 colorSum, inout float weightSum
) {
    vec2 pixelPos = sampleTexel - 0.5;
    vec2 originPixelPos = floor(pixelPos);
    vec2 gatherUV = (originPixelPos + 1.0) * global_mainImageSizeRcp;
    vec2 gatherUVHalf = (originPixelPos + 1.0) * global_mipmapSizesRcp[1];
    vec2 bilinearWeights = pixelPos - originPixelPos;

    vec4 bilateralWeights;
    bilateralWeights.yz = bilinearWeights.xx;
    bilateralWeights.xw = 1.0 - bilinearWeights.xx;
    bilateralWeights.xy *= bilinearWeights.yy;
    bilateralWeights.zw *= 1.0 - bilinearWeights.yy;

    uvec4 prevNs = textureGather(usam_packedNZ, gatherUVHalf, 0);
    bilateralWeights.x *= normalWeight(centerWorldNormal, prevNs.x);
    bilateralWeights.y *= normalWeight(centerWorldNormal, prevNs.y);
    bilateralWeights.z *= normalWeight(centerWorldNormal, prevNs.z);
    bilateralWeights.w *= normalWeight(centerWorldNormal, prevNs.w);

    float phiZ = max(0.01 * pow2(centerViewZ), 0.5);
    vec4 prevViewZs = uintBitsToFloat(textureGather(usam_packedNZ, gatherUVHalf, 1));
    bilateralWeights.x *= viewZWeight(centerViewZ, prevViewZs.x, phiZ);
    bilateralWeights.y *= viewZWeight(centerViewZ, prevViewZs.y, phiZ);
    bilateralWeights.z *= viewZWeight(centerViewZ, prevViewZs.z, phiZ);
    bilateralWeights.w *= viewZWeight(centerViewZ, prevViewZs.w, phiZ);

    bilateralWeights *= baseWeight;
    weightSum += bilateralWeights.x + bilateralWeights.y + bilateralWeights.z + bilateralWeights.w;

    vec4 colorRs = textureGather(usam_temp3, gatherUV, 0);
    colorSum.r += dot(bilateralWeights, colorRs);
    vec4 colorGs = textureGather(usam_temp3, gatherUV, 1);
    colorSum.g += dot(bilateralWeights, colorGs);
    vec4 colorBs = textureGather(usam_temp3, gatherUV, 2);
    colorSum.b += dot(bilateralWeights, colorBs);
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

            {
                vec2 pixelPos = texelPos2x2F - 0.5;
                vec2 originPixelPos = floor(pixelPos);
                vec2 gatherUV = (originPixelPos + 1.0) * global_mainImageSizeRcp;
                vec2 bilinearWeights = pixelPos - originPixelPos;

                vec4 bilateralWeights;
                bilateralWeights.yz = bilinearWeights.xx;
                bilateralWeights.xw = 1.0 - bilinearWeights.xx;
                bilateralWeights.xy *= bilinearWeights.yy;
                bilateralWeights.zw *= 1.0 - bilinearWeights.yy;
                weightSum += bilateralWeights.x + bilateralWeights.y + bilateralWeights.z + bilateralWeights.w;

                vec4 colorRs = textureGather(usam_temp3, gatherUV, 0);
                colorSum.r += dot(bilateralWeights, colorRs);
                vec4 colorGs = textureGather(usam_temp3, gatherUV, 1);
                colorSum.g += dot(bilateralWeights, colorGs);
                vec4 colorBs = textureGather(usam_temp3, gatherUV, 2);
                colorSum.b += dot(bilateralWeights, colorBs);
            }

            bilateralSample(texelPos2x2F + vec2(-0.5, -0.5), 128.0, viewZ, worldNormal, colorSum, weightSum);
            bilateralSample(texelPos2x2F + vec2(-0.5, 0.5), 128.0, viewZ, worldNormal, colorSum, weightSum);
            bilateralSample(texelPos2x2F + vec2(0.5, -0.5), 128.0, viewZ, worldNormal, colorSum, weightSum);
            bilateralSample(texelPos2x2F + vec2(0.5, 0.5), 128.0, viewZ, worldNormal, colorSum, weightSum);
            colorSum /= weightSum;
            outputColor.rgb = colorSum;
        }

        imageStore(uimg_temp4, texelPos, outputColor);
    }
}