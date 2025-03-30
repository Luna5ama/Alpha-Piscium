#include "Common.glsl"
#include "/util/Coords.glsl"
#include "/util/Interpo.glsl"

vec3 cameraDelta = cameraPosition - previousCameraPosition;

float normalWeight(vec3 currWorldNormal, uint packedNormal) {
    vec3 prevWorldNormal = coords_octDecode11(unpackSnorm2x16(packedNormal));
    float sdot = saturate(dot(currWorldNormal, prevWorldNormal));
    return pow(sdot, float(SETTING_DENOISER_REPROJ_NORMAL_STRICTNESS));
}

float posWeight(float currViewZ, vec3 currScene, vec2 curr2PrevScreen, uint prevViewZI) {
    float prevViewZ = uintBitsToFloat(prevViewZI);
    vec3 prevView = coords_toViewCoord(curr2PrevScreen, prevViewZ, gbufferPrevProjectionInverse);
    vec4 prevScene = gbufferPrevModelViewInverse * vec4(prevView, 1.0);
    prevScene.xyz -= cameraDelta;

    vec3 diff = currScene.xyz - prevScene.xyz;
    float distSq = dot(diff, diff);
    float a = max(pow2(currViewZ) * 0.001, 0.5);
    return a / (a + distSq);
}

vec4 computeBilateralWeights(
usampler2D svgfHistory, vec2 gatherUV,
vec3 currScene, float currViewZ, vec3 currWorldNormal
) {
    vec4 result = vec4(1.0);

    uvec4 prevNs = textureGather(svgfHistory, gatherUV, 0);
    result.x *= normalWeight(currWorldNormal, prevNs.x);
    result.y *= normalWeight(currWorldNormal, prevNs.y);
    result.z *= normalWeight(currWorldNormal, prevNs.z);
    result.w *= normalWeight(currWorldNormal, prevNs.w);

    uvec4 prevViewZs = textureGather(svgfHistory, gatherUV, 1);
    result.x *= posWeight(currViewZ, currScene, gatherUV + global_mainImageSizeRcp * vec2(-0.5, 0.5), prevViewZs.x);
    result.y *= posWeight(currViewZ, currScene, gatherUV + global_mainImageSizeRcp * vec2(0.5, 0.5), prevViewZs.y);
    result.z *= posWeight(currViewZ, currScene, gatherUV + global_mainImageSizeRcp * vec2(0.5, -0.5), prevViewZs.z);
    result.w *= posWeight(currViewZ, currScene, gatherUV + global_mainImageSizeRcp * vec2(-0.5, -0.5), prevViewZs.w);

    return result;
}

void bilateralSample(
usampler2D svgfHistory,
vec2 gatherTexelPos, vec4 baseWeights,
vec3 currScene, float currViewZ, vec3 currWorldNormal,
inout vec3 prevDiffuse, inout vec2 prevMoments, inout float prevHLen, inout float weightSum
) {
    vec2 gatherTexelPosClamped = clamp(gatherTexelPos, vec2(1.0), global_mainImageSize - 1);
    if (all(equal(gatherTexelPos, gatherTexelPosClamped))) {
        vec2 gatherUV1 = svgf_gatherUV1(gatherTexelPos);
        vec2 gatherUV2 = svgf_gatherUV2(gatherTexelPos);
        vec2 gatherUV3 = svgf_gatherUV3(gatherTexelPos);
        vec2 gatherUV4 = svgf_gatherUV4(gatherTexelPos);

        vec4 bilateralWeights = computeBilateralWeights(svgfHistory, gatherUV1, currScene, currViewZ, currWorldNormal);
        float bilateralWeightSum = bilateralWeights.x + bilateralWeights.y + bilateralWeights.z + bilateralWeights.w;

        vec4 interpoWeights = baseWeights * bilateralWeights;
        weightSum += interpoWeights.x + interpoWeights.y + interpoWeights.z + interpoWeights.w;

        {
            uvec4 prevDiffuseData1 = textureGather(svgfHistory, gatherUV3, 0);
            vec2 prevDiffuse11 = unpackHalf2x16(prevDiffuseData1.x);
            vec2 prevDiffuse12 = unpackHalf2x16(prevDiffuseData1.y);
            vec2 prevDiffuse13 = unpackHalf2x16(prevDiffuseData1.z);
            vec2 prevDiffuse14 = unpackHalf2x16(prevDiffuseData1.w);
            vec4 prevDiffuseR = vec4(prevDiffuse11.x, prevDiffuse12.x, prevDiffuse13.x, prevDiffuse14.x);
            vec4 prevDiffuseG = vec4(prevDiffuse11.y, prevDiffuse12.y, prevDiffuse13.y, prevDiffuse14.y);

            uvec4 prevDiffuseData2 = textureGather(svgfHistory, gatherUV3, 1);
            vec2 prevDiffuse21 = unpackHalf2x16(prevDiffuseData2.x);
            vec2 prevDiffuse22 = unpackHalf2x16(prevDiffuseData2.y);
            vec2 prevDiffuse23 = unpackHalf2x16(prevDiffuseData2.z);
            vec2 prevDiffuse24 = unpackHalf2x16(prevDiffuseData2.w);
            vec4 prevDiffuseB = vec4(prevDiffuse21.x, prevDiffuse22.x, prevDiffuse23.x, prevDiffuse24.x);

            prevDiffuse.r += dot(interpoWeights, prevDiffuseR);
            prevDiffuse.g += dot(interpoWeights, prevDiffuseG);
            prevDiffuse.b += dot(interpoWeights, prevDiffuseB);
        }

        {
            uvec4 prevMomentDatas = textureGather(svgfHistory, gatherUV2, 0);
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
            uvec4 prevHLenData = textureGather(svgfHistory, gatherUV2, 1);
            vec4 prevHLens = vec4(bitfieldExtract(prevHLenData, 0, 16));

            prevHLen += max(prevHLen, dot(bilateralWeights, prevHLens) / bilateralWeightSum);
        }
    }
}

void gi_reproject(
usampler2D svgfHistory,
vec2 screenPos, float currViewZ, vec3 currViewNormal, bool isHand,
out vec3 prevDiffuse, out vec2 prevMoments, out float prevHLen
) {
    prevDiffuse = vec3(0.0);
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
    vec2 curr2PrevTexel = curr2PrevScreen * global_mipmapSizes[1];
    vec3 currWorldNormal = mat3(gbufferModelViewInverse) * currViewNormal;

    vec2 textureSizeRcp = global_mipmapSizesRcp[1];

    vec2 centerPixel = curr2PrevTexel - 0.5;
    vec2 centerPixelOrigin = floor(centerPixel);
    vec2 gatherTexelPos = centerPixelOrigin + 1.0;
    vec2 pixelPosFract = centerPixel - centerPixelOrigin;

    vec4 centerWeights = computeBilateralWeights(svgfHistory, gatherTexelPos, currScene.xyz, currViewZ, currWorldNormal);

    const float WEIGHT_EPSILON = 0.5;
    float weightSum = 0.0;
    uint flag = uint(any(lessThan(centerWeights, vec4(WEIGHT_EPSILON))));
    flag |= uint(any(lessThan(curr2PrevTexel, vec2(1.0))));
    flag |= uint(any(greaterThan(curr2PrevTexel, global_mipmapSizes[1] - 2.0)));
    if (bool(flag)) {
        vec2 bilinearWeights = pixelPosFract;

        vec4 gatherWeights;
        gatherWeights.yz = bilinearWeights.xx;
        gatherWeights.xw = 1.0 - bilinearWeights.xx;
        gatherWeights.xy *= bilinearWeights.yy;
        gatherWeights.zw *= 1.0 - bilinearWeights.yy;

        bilateralSample(
            svgfHistory,
            gatherTexelPos + vec2(-1.0, 1.0), gatherWeights,
            currScene.xyz, currViewZ, currWorldNormal,
            prevDiffuse, prevMoments, prevHLen, weightSum
        );

        bilateralSample(
            svgfHistory,
            gatherTexelPos + vec2(1.0, 1.0), gatherWeights,
            currScene.xyz, currViewZ, currWorldNormal,
            prevDiffuse, prevMoments, prevHLen, weightSum
        );

        bilateralSample(
            svgfHistory,
            gatherTexelPos + vec2(1.0, -1.0), gatherWeights,
            currScene.xyz, currViewZ, currWorldNormal,
            prevDiffuse, prevMoments, prevHLen, weightSum
        );

        bilateralSample(
            svgfHistory,
            gatherTexelPos + vec2(-1.0, -1.0), gatherWeights,
            currScene.xyz, currViewZ, currWorldNormal,
            prevDiffuse, prevMoments, prevHLen, weightSum
        );
    } else {
        vec4 weightX = interpo_catmullRomWeights(pixelPosFract.x);
        vec4 weightY = interpo_catmullRomWeights(pixelPosFract.y);

        bilateralSample(
            svgfHistory,
            gatherTexelPos + vec2(-1.0, 1.0), weightX.xyyx * weightY.wwzz,
            currScene.xyz, currViewZ, currWorldNormal,
            prevDiffuse, prevMoments, prevHLen, weightSum
        );

        bilateralSample(
            svgfHistory,
            gatherTexelPos + vec2(1.0, 1.0), weightX.zwwz * weightY.wwzz,
            currScene.xyz, currViewZ, currWorldNormal,
            prevDiffuse, prevMoments, prevHLen, weightSum
        );

        bilateralSample(
            svgfHistory,
            gatherTexelPos + vec2(1.0, -1.0), weightX.zwwz * weightY.yyxx,
            currScene.xyz, currViewZ, currWorldNormal,
            prevDiffuse, prevMoments, prevHLen, weightSum
        );

        bilateralSample(
            svgfHistory,
            gatherTexelPos + vec2(-1.0, -1.0), weightX.xyyx * weightY.yyxx,
            currScene.xyz, currViewZ, currWorldNormal,
            prevDiffuse, prevMoments, prevHLen, weightSum
        );
    }


    const float WEIGHT_EPSILON_FINAL = 0.0001;
    if (weightSum < WEIGHT_EPSILON_FINAL) {
        prevDiffuse = vec3(0.0);
        prevMoments = vec2(0.0);
        prevHLen = 0.0;
    } else {
        float rcpWeightSum = 1.0 / weightSum;
        prevDiffuse = max(prevDiffuse * rcpWeightSum, 0.0);
        prevMoments = max(prevMoments * rcpWeightSum, 0.0);
        prevHLen = max(ceil(prevHLen), 1.0);
    }
}