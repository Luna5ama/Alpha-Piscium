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
usampler2D prevNZTex, vec2 gatherUV,
vec3 currScene, float currViewZ, vec3 currWorldNormal
) {
    vec4 result = vec4(1.0);

    uvec4 prevNs = textureGather(prevNZTex, gatherUV, 0);
    result.x *= normalWeight(currWorldNormal, prevNs.x);
    result.y *= normalWeight(currWorldNormal, prevNs.y);
    result.z *= normalWeight(currWorldNormal, prevNs.z);
    result.w *= normalWeight(currWorldNormal, prevNs.w);

    uvec4 prevViewZs = textureGather(prevNZTex, gatherUV, 1);
    result.x *= posWeight(currViewZ, currScene, gatherUV + global_mipmapSizesRcp[1] * vec2(-0.5, 0.5), prevViewZs.x);
    result.y *= posWeight(currViewZ, currScene, gatherUV + global_mipmapSizesRcp[1] * vec2(0.5, 0.5), prevViewZs.y);
    result.z *= posWeight(currViewZ, currScene, gatherUV + global_mipmapSizesRcp[1] * vec2(0.5, -0.5), prevViewZs.z);
    result.w *= posWeight(currViewZ, currScene, gatherUV + global_mipmapSizesRcp[1] * vec2(-0.5, -0.5), prevViewZs.w);

    return result;
}

void bilateralSample(
usampler2D svgfHistory, usampler2D prevNZTex,
vec2 gatherUVIn, vec4 baseWeights,
vec3 currScene, float currViewZ, vec3 currWorldNormal,
inout vec4 prevColorHLen, inout vec2 prevMoments, inout float weightSum
) {
    vec2 gatherUV = clamp(gatherUVIn, global_mipmapSizesRcp[1] * 1.5, 1.0 - global_mipmapSizesRcp[1] * 1.5);
    vec4 bilateralWeights = computeBilateralWeights(prevNZTex, gatherUV, currScene, currViewZ, currWorldNormal);

    float bilateralWeightSum = bilateralWeights.x + bilateralWeights.y + bilateralWeights.z + bilateralWeights.w;

    uvec4 prevDataG = textureGather(svgfHistory, gatherUV, 1);
    vec2 prevDataG1 = unpackHalf2x16(prevDataG.x);
    vec2 prevDataG2 = unpackHalf2x16(prevDataG.y);
    vec2 prevDataG3 = unpackHalf2x16(prevDataG.z);
    vec2 prevDataG4 = unpackHalf2x16(prevDataG.w);
    vec4 prevColorHLens = vec4(prevDataG1.y, prevDataG2.y, prevDataG3.y, prevDataG4.y);
    prevColorHLen.a = max(prevColorHLen.a, dot(bilateralWeights, prevColorHLens) / bilateralWeightSum);

    vec4 interpoWeights = baseWeights * bilateralWeights;
    weightSum += interpoWeights.x + interpoWeights.y + interpoWeights.z + interpoWeights.w;

    vec4 prevColorBs = vec4(prevDataG1.x, prevDataG2.x, prevDataG3.x, prevDataG4.x);
    prevColorHLen.b += dot(interpoWeights, prevColorBs);

    uvec4 prevDataR = textureGather(svgfHistory, gatherUV, 0);
    vec2 prevDataR1 = unpackHalf2x16(prevDataR.x);
    vec2 prevDataR2 = unpackHalf2x16(prevDataR.y);
    vec2 prevDataR3 = unpackHalf2x16(prevDataR.z);
    vec2 prevDataR4 = unpackHalf2x16(prevDataR.w);
    vec4 prevColorRs = vec4(prevDataR1.x, prevDataR2.x, prevDataR3.x, prevDataR4.x);
    vec4 prevColorGs = vec4(prevDataR1.y, prevDataR2.y, prevDataR3.y, prevDataR4.y);
    prevColorHLen.r += dot(interpoWeights, prevColorRs);
    prevColorHLen.g += dot(interpoWeights, prevColorGs);

    uvec4 prevDataB = textureGather(svgfHistory, gatherUV, 2);
    vec2 prevDataB1 = unpackHalf2x16(prevDataB.x);
    vec2 prevDataB2 = unpackHalf2x16(prevDataB.y);
    vec2 prevDataB3 = unpackHalf2x16(prevDataB.z);
    vec2 prevDataB4 = unpackHalf2x16(prevDataB.w);
    vec4 prevMomentsRs = vec4(prevDataB1.x, prevDataB2.x, prevDataB3.x, prevDataB4.x);
    vec4 prevMomentsGs = vec4(prevDataB1.y, prevDataB2.y, prevDataB3.y, prevDataB4.y);
    prevMoments.x += dot(interpoWeights, prevMomentsRs);
    prevMoments.y += dot(interpoWeights, prevMomentsGs);
}

void gi_reproject(
usampler2D svgfHistory, usampler2D prevNZTex,
vec2 screenPos, float currViewZ, vec3 currViewNormal, bool isHand,
out vec4 prevColorHLen, out vec2 prevMoments
) {
    prevColorHLen = vec4(0.0);
    prevMoments = vec2(0.0);

    vec3 currView = coords_toViewCoord(screenPos, currViewZ, gbufferProjectionInverse);
    vec4 currScene = gbufferModelViewInverse * vec4(currView, 1.0);
    currScene.xyz = isHand ? currScene.xyz + gbufferModelViewInverse[3].xyz : currScene.xyz;

    vec4 curr2PrevScene = coord_sceneCurrToPrev(currScene, isHand);
    curr2PrevScene.xyz = isHand ? curr2PrevScene.xyz - gbufferPrevModelViewInverse[3].xyz : curr2PrevScene.xyz;

    vec4 curr2PrevView = gbufferPrevModelView * curr2PrevScene;
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
    vec2 gatherUV = (centerPixelOrigin + 1.0) * textureSizeRcp;
    vec2 pixelPosFract = centerPixel - centerPixelOrigin;

    vec4 centerWeights = computeBilateralWeights(prevNZTex, gatherUV, currScene.xyz, currViewZ, currWorldNormal);

    const float WEIGHT_EPSILON = 0.5;
    float weightSum = 0.0;
    if (any(lessThan(centerWeights, vec4(WEIGHT_EPSILON)))) {
        vec2 bilinearWeights = pixelPosFract;

        vec4 gatherWeights;
        gatherWeights.yz = bilinearWeights.xx;
        gatherWeights.xw = 1.0 - bilinearWeights.xx;
        gatherWeights.xy *= bilinearWeights.yy;
        gatherWeights.zw *= 1.0 - bilinearWeights.yy;

        bilateralSample(
            svgfHistory, prevNZTex,
            gatherUV + vec2(-1.0, 1.0) * textureSizeRcp, gatherWeights,
            currScene.xyz, currViewZ, currWorldNormal,
            prevColorHLen, prevMoments, weightSum
        );

        bilateralSample(
            svgfHistory, prevNZTex,
            gatherUV + vec2(1.0, 1.0) * textureSizeRcp, gatherWeights,
            currScene.xyz, currViewZ, currWorldNormal,
            prevColorHLen, prevMoments, weightSum
        );

        bilateralSample(
            svgfHistory, prevNZTex,
            gatherUV + vec2(1.0, -1.0) * textureSizeRcp, gatherWeights,
            currScene.xyz, currViewZ, currWorldNormal,
            prevColorHLen, prevMoments, weightSum
        );

        bilateralSample(
            svgfHistory, prevNZTex,
            gatherUV + vec2(-1.0, -1.0) * textureSizeRcp, gatherWeights,
            currScene.xyz, currViewZ, currWorldNormal,
            prevColorHLen, prevMoments, weightSum
        );
    } else {
        vec4 weightX = interpo_catmullRomWeights(pixelPosFract.x);
        vec4 weightY = interpo_catmullRomWeights(pixelPosFract.y);

        bilateralSample(
            svgfHistory, prevNZTex,
            gatherUV + vec2(-1.0, 1.0) * textureSizeRcp, weightX.xyyx * weightY.wwzz,
            currScene.xyz, currViewZ, currWorldNormal,
            prevColorHLen, prevMoments, weightSum
        );

        bilateralSample(
            svgfHistory, prevNZTex,
            gatherUV + vec2(1.0, 1.0) * textureSizeRcp, weightX.zwwz * weightY.wwzz,
            currScene.xyz, currViewZ, currWorldNormal,
            prevColorHLen, prevMoments, weightSum
        );

        bilateralSample(
            svgfHistory, prevNZTex,
            gatherUV + vec2(1.0, -1.0) * textureSizeRcp, weightX.zwwz * weightY.yyxx,
            currScene.xyz, currViewZ, currWorldNormal,
            prevColorHLen, prevMoments, weightSum
        );

        bilateralSample(
            svgfHistory, prevNZTex,
            gatherUV + vec2(-1.0, -1.0) * textureSizeRcp, weightX.xyyx * weightY.yyxx,
            currScene.xyz, currViewZ, currWorldNormal,
            prevColorHLen, prevMoments, weightSum
        );
    }


    const float WEIGHT_EPSILON_FINAL = 0.0001;
    if (weightSum < WEIGHT_EPSILON_FINAL) {
        prevColorHLen = vec4(0.0);
        prevMoments = vec2(0.0);
    } else {
        float rcpWeightSum = 1.0 / weightSum;
        prevColorHLen.rgb = max(prevColorHLen.rgb * rcpWeightSum, 0.0);
        prevMoments = max(prevMoments * rcpWeightSum, 0.0);
        prevColorHLen.a = max(ceil(prevColorHLen.a), 1.0);
    }
}