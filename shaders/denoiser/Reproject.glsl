#include "Common.glsl"
#include "/util/Coords.glsl"
#include "/util/Colors.glsl"
#include "/util/Interpo.glsl"
#include "/util/NZPacking.glsl"

vec3 cameraDelta = cameraPosition - previousCameraPosition;

float normalWeight(vec3 currWorldNormal, uint packedNormal) {
    vec3 prevWorldNormal = coords_octDecode11(unpackSnorm2x16(packedNormal));
    float sdot = saturate(dot(currWorldNormal, prevWorldNormal));
    return pow(sdot, float(SETTING_DENOISER_REPROJ_NORMAL_STRICTNESS));
}

float posWeight(vec3 currScene, vec2 curr2PrevScreen, uint prevViewZI, float a) {
    float prevViewZ = uintBitsToFloat(prevViewZI);
    vec3 prevView = coords_toViewCoord(curr2PrevScreen, prevViewZ, gbufferPrevProjectionInverse);
    vec4 prevScene = gbufferPrevModelViewInverse * vec4(prevView, 1.0);
    prevScene.xyz -= cameraDelta;

    vec3 diff = currScene.xyz - prevScene.xyz;
    float distSq = dot(diff, diff);
    return a / (a + distSq);
}

vec4 computeBilateralWeights(
usampler2D packedZN,
vec2 gatherTexelPos,
vec3 currScene, vec3 currToPrevViewNormal, float a
) {
    vec2 screenPos = gatherTexelPos * global_mainImageSizeRcp;
    vec2 gatherUV = nzpacking_fullResGatherUV(gatherTexelPos);
    vec4 result = vec4(1.0);

    uvec4 prevNs = textureGather(packedZN, gatherUV, 0);
    result.x *= normalWeight(currToPrevViewNormal, prevNs.x);
    result.y *= normalWeight(currToPrevViewNormal, prevNs.y);
    result.z *= normalWeight(currToPrevViewNormal, prevNs.z);
    result.w *= normalWeight(currToPrevViewNormal, prevNs.w);

    uvec4 prevViewZs = textureGather(packedZN, gatherUV, 1);
    result.x *= posWeight(currScene, screenPos + global_mainImageSizeRcp * vec2(-0.5, 0.5), prevViewZs.x, a);
    result.y *= posWeight(currScene, screenPos + global_mainImageSizeRcp * vec2(0.5, 0.5), prevViewZs.y, a);
    result.z *= posWeight(currScene, screenPos + global_mainImageSizeRcp * vec2(0.5, -0.5), prevViewZs.z, a);
    result.w *= posWeight(currScene, screenPos + global_mainImageSizeRcp * vec2(-0.5, -0.5), prevViewZs.w, a);

    return result;
}

void bilateralSample(
usampler2D svgfHistory, usampler2D packedZN,
vec2 gatherTexelPos, vec4 baseWeights,
vec3 currScene, float currViewZ, vec3 currToPrevViewNormal,
inout vec3 prevColor, inout vec3 prevFastColor, inout vec2 prevMoments, inout float prevHLen, inout float weightSum
) {
    vec2 gatherTexelPosClamped = clamp(gatherTexelPos, vec2(1.0), global_mainImageSize - 1);
    if (all(equal(gatherTexelPos, gatherTexelPosClamped))) {
        vec2 gatherUV1 = svgf_gatherUV1(gatherTexelPos);

        float a = abs(currViewZ) * 0.0001;
        vec4 bilateralWeights = computeBilateralWeights(packedZN, gatherTexelPos, currScene, currToPrevViewNormal, a);
        float bilateralWeightSum = bilateralWeights.x + bilateralWeights.y + bilateralWeights.z + bilateralWeights.w;

        vec4 interpoWeights = baseWeights * bilateralWeights;
        weightSum += interpoWeights.x + interpoWeights.y + interpoWeights.z + interpoWeights.w;

        {
            uvec4 prevColorData = textureGather(svgfHistory, gatherUV1, 0);
            vec3 prevColor1 = colors_LogLuvToSRGB(unpackUnorm4x8(prevColorData.x));
            vec3 prevColor2 = colors_LogLuvToSRGB(unpackUnorm4x8(prevColorData.y));
            vec3 prevColor3 = colors_LogLuvToSRGB(unpackUnorm4x8(prevColorData.z));
            vec3 prevColor4 = colors_LogLuvToSRGB(unpackUnorm4x8(prevColorData.w));

            vec4 prevColorR = vec4(prevColor1.r, prevColor2.r, prevColor3.r, prevColor4.r);
            prevColor.r += dot(interpoWeights, prevColorR);
            vec4 prevColorG = vec4(prevColor1.g, prevColor2.g, prevColor3.g, prevColor4.g);
            prevColor.g += dot(interpoWeights, prevColorG);
            vec4 prevColorB = vec4(prevColor1.b, prevColor2.b, prevColor3.b, prevColor4.b);
            prevColor.b += dot(interpoWeights, prevColorB);
        }

        {
            uvec4 prevFastColorData = textureGather(svgfHistory, gatherUV1, 1);
            vec3 prevFastColor1 = colors_LogLuvToSRGB(unpackUnorm4x8(prevFastColorData.x));
            vec3 prevFastColor2 = colors_LogLuvToSRGB(unpackUnorm4x8(prevFastColorData.y));
            vec3 prevFastColor3 = colors_LogLuvToSRGB(unpackUnorm4x8(prevFastColorData.z));
            vec3 prevFastColor4 = colors_LogLuvToSRGB(unpackUnorm4x8(prevFastColorData.w));

            vec4 prevFastColorR = vec4(prevFastColor1.r, prevFastColor2.r, prevFastColor3.r, prevFastColor4.r);
            prevFastColor.r += dot(interpoWeights, prevFastColorR);
            vec4 prevFastColorG = vec4(prevFastColor1.g, prevFastColor2.g, prevFastColor3.g, prevFastColor4.g);
            prevFastColor.g += dot(interpoWeights, prevFastColorG);
            vec4 prevFastColorB = vec4(prevFastColor1.b, prevFastColor2.b, prevFastColor3.b, prevFastColor4.b);
            prevFastColor.b += dot(interpoWeights, prevFastColorB);
        }

        {
            uvec4 prevMomentDatas = textureGather(svgfHistory, gatherUV1, 2);
            vec2 prevMoments1 = unpackHalf2x16(prevMomentDatas.x);
            vec2 prevMoments2 = unpackHalf2x16(prevMomentDatas.y);
            vec2 prevMoments3 = unpackHalf2x16(prevMomentDatas.z);
            vec2 prevMoments4 = unpackHalf2x16(prevMomentDatas.w);
            vec4 prevMomentsR = vec4(prevMoments1.x, prevMoments2.x, prevMoments3.x, prevMoments4.x);
            vec4 prevMomentsG = vec4(prevMoments1.y, prevMoments2.y, prevMoments3.y, prevMoments4.y);

            prevMoments.x += dot(interpoWeights, prevMomentsR);
            prevMoments.y += dot(interpoWeights, prevMomentsG);
        }

        {
            uvec4 prevHLenData = textureGather(svgfHistory, gatherUV1, 3);
            vec4 prevHLens = uintBitsToFloat(prevHLenData);

            prevHLen = max(prevHLen, dot(bilateralWeights, prevHLens) / bilateralWeightSum);
        }
    }
}

void gi_reproject(
usampler2D svgfHistory, usampler2D packedZN,
vec2 screenPos, float currViewZ, vec3 currViewNormal, bool isHand,
out vec3 prevColor, out vec3 prevFastColor, out vec2 prevMoments, out float prevHLen
) {
    prevColor = vec3(0.0);
    prevFastColor = vec3(0.0);
    prevMoments = vec2(0.0);
    prevHLen = 0.0;

    vec3 currView = coords_toViewCoord(screenPos, currViewZ, gbufferProjectionInverse);
    vec4 currScene = gbufferModelViewInverse * vec4(currView, 1.0);
    vec4 curr2PrevView = coord_viewCurrToPrev(vec4(currView, 1.0), isHand);
    vec4 curr2PrevClip = gbufferPrevProjection * curr2PrevView;
    uint clipFlag = uint(curr2PrevClip.z > 0.0);
    clipFlag &= uint(all(lessThan(abs(curr2PrevClip.xyz), curr2PrevClip.www)));
    if (!bool(clipFlag)) {
        return;
    }

    vec2 curr2PrevNDC = curr2PrevClip.xy / curr2PrevClip.w;
    vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;

    if (any(notEqual(curr2PrevScreen, saturate(curr2PrevScreen)))) {
        return;
    }
    vec2 curr2PrevTexel = curr2PrevScreen * global_mainImageSize;
    vec3 currWorldNormal = mat3(gbufferModelViewInverse) * currViewNormal;
    vec3 currToPrevViewNormal = mat3(gbufferPrevModelView) * currWorldNormal;

    vec2 textureSizeRcp = global_mainImageSizeRcp;

    vec2 centerPixel = curr2PrevTexel - 0.5;
    vec2 centerPixelOrigin = floor(centerPixel);
    vec2 gatherTexelPos = centerPixelOrigin + 1.0;
    vec2 pixelPosFract = centerPixel - centerPixelOrigin;

    float a = abs(currViewZ) * 0.002;
    vec4 centerWeights = computeBilateralWeights(packedZN, gatherTexelPos, currScene.xyz, currToPrevViewNormal, a);

    const float WEIGHT_EPSILON = 0.5;
    float weightSum = 0.0;
    uint flag = uint(any(lessThan(centerWeights, vec4(WEIGHT_EPSILON))));
    flag |= uint(any(lessThan(curr2PrevTexel, vec2(1.0))));
    flag |= uint(any(greaterThan(curr2PrevTexel, global_mainImageSize - 1.0)));
    if (bool(flag)) {
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
        bilateralSample(
            svgfHistory, packedZN,
            gatherTexelPos + vec2(-1.0, 1.0), sampleGatherWeights,
            currScene.xyz, currViewZ, currToPrevViewNormal,
            prevColor, prevFastColor, prevMoments, prevHLen, weightSum
        );

        sampleGatherWeights = weightX.zwwz * weightY.wwzz;
        sampleGatherWeights.w += blinearWeights4.y;
        bilateralSample(
            svgfHistory, packedZN,
            gatherTexelPos + vec2(1.0, 1.0), sampleGatherWeights,
            currScene.xyz, currViewZ, currToPrevViewNormal,
            prevColor, prevFastColor, prevMoments, prevHLen, weightSum
        );

        sampleGatherWeights = weightX.zwwz * weightY.yyxx;
        sampleGatherWeights.x += blinearWeights4.z;
        bilateralSample(
            svgfHistory, packedZN,
            gatherTexelPos + vec2(1.0, -1.0), sampleGatherWeights,
            currScene.xyz, currViewZ, currToPrevViewNormal,
            prevColor, prevFastColor, prevMoments, prevHLen, weightSum
        );

        sampleGatherWeights = weightX.xyyx * weightY.yyxx;
        sampleGatherWeights.y += blinearWeights4.w;
        bilateralSample(
            svgfHistory, packedZN,
            gatherTexelPos + vec2(-1.0, -1.0), sampleGatherWeights,
            currScene.xyz, currViewZ, currToPrevViewNormal,
            prevColor, prevFastColor, prevMoments, prevHLen, weightSum
        );
    } else {
        vec4 weightX = interpo_catmullRomWeights(pixelPosFract.x);
        vec4 weightY = interpo_catmullRomWeights(pixelPosFract.y);

        bilateralSample(
            svgfHistory, packedZN,
            gatherTexelPos + vec2(-1.0, 1.0), weightX.xyyx * weightY.wwzz,
            currScene.xyz, currViewZ, currToPrevViewNormal,
            prevColor, prevFastColor, prevMoments, prevHLen, weightSum
        );

        bilateralSample(
            svgfHistory, packedZN,
            gatherTexelPos + vec2(1.0, 1.0), weightX.zwwz * weightY.wwzz,
            currScene.xyz, currViewZ, currToPrevViewNormal,
            prevColor, prevFastColor, prevMoments, prevHLen, weightSum
        );

        bilateralSample(
            svgfHistory, packedZN,
            gatherTexelPos + vec2(1.0, -1.0), weightX.zwwz * weightY.yyxx,
            currScene.xyz, currViewZ, currToPrevViewNormal,
            prevColor, prevFastColor, prevMoments, prevHLen, weightSum
        );

        bilateralSample(
            svgfHistory, packedZN,
            gatherTexelPos + vec2(-1.0, -1.0), weightX.xyyx * weightY.yyxx,
            currScene.xyz, currViewZ, currToPrevViewNormal,
            prevColor, prevFastColor, prevMoments, prevHLen, weightSum
        );
    }


    const float WEIGHT_EPSILON_FINAL = 0.0001;
    if (weightSum < WEIGHT_EPSILON_FINAL) {
        prevColor = vec3(0.0);
        prevFastColor = vec3(0.0);
        prevMoments = vec2(0.0);
        prevHLen = 0.0;
    } else {
        float rcpWeightSum = 1.0 / weightSum;
        prevColor = max(prevColor * rcpWeightSum, 0.0);
        prevFastColor = max(prevFastColor * rcpWeightSum, 0.0);
        prevMoments = max(prevMoments * rcpWeightSum, 0.0);
        prevHLen = max(ceil(prevHLen), 0.0);
    }
}