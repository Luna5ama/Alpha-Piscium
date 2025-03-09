#include "Common.glsl"
#include "/util/Coords.glsl"

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
    float a = abs(currViewZ) * 0.00001;
    return a / (a + distSq);
}

void bilateralSample(
usampler2D svgfHistory, usampler2D prevNZTex,
vec2 sampleTexel, vec3 currScene, float currViewZ, vec3 currWorldNormal, float baseWeight,
inout vec4 prevColorHLen, inout vec2 prevMoments, inout float weightSum
) {
    vec2 pixelPos = sampleTexel - 0.5;
    vec2 originPixelPos = floor(pixelPos);
    vec2 gatherUV = (originPixelPos + 1.0) * global_mainImageSizeRcp;
    vec2 bilinearWeights = pixelPos - originPixelPos;

    vec4 bilateralWeights;
    bilateralWeights.yz = bilinearWeights.xx;
    bilateralWeights.xw = 1.0 - bilinearWeights.xx;
    bilateralWeights.xy *= bilinearWeights.yy;
    bilateralWeights.zw *= 1.0 - bilinearWeights.yy;

    uvec4 prevNs = textureGather(prevNZTex, gatherUV, 0);
    bilateralWeights.x *= normalWeight(currWorldNormal, prevNs.x);
    bilateralWeights.y *= normalWeight(currWorldNormal, prevNs.y);
    bilateralWeights.z *= normalWeight(currWorldNormal, prevNs.z);
    bilateralWeights.w *= normalWeight(currWorldNormal, prevNs.w);

    uvec4 prevViewZs = textureGather(prevNZTex, gatherUV, 1);
    bilateralWeights.x *= posWeight(currViewZ, currScene, gatherUV, prevViewZs.x);
    bilateralWeights.y *= posWeight(currViewZ, currScene, gatherUV, prevViewZs.y);
    bilateralWeights.z *= posWeight(currViewZ, currScene, gatherUV, prevViewZs.z);
    bilateralWeights.w *= posWeight(currViewZ, currScene, gatherUV, prevViewZs.w);

    bilateralWeights *= baseWeight;
    weightSum += bilateralWeights.x + bilateralWeights.y + bilateralWeights.z + bilateralWeights.w;

    uvec4 prevDataR = textureGather(svgfHistory, gatherUV, 0);
    vec2 prevDataR1 = unpackHalf2x16(prevDataR.x);
    vec2 prevDataR2 = unpackHalf2x16(prevDataR.y);
    vec2 prevDataR3 = unpackHalf2x16(prevDataR.z);
    vec2 prevDataR4 = unpackHalf2x16(prevDataR.w);
    vec4 prevColorRs = vec4(prevDataR1.x, prevDataR2.x, prevDataR3.x, prevDataR4.x);
    vec4 prevColorGs = vec4(prevDataR1.y, prevDataR2.y, prevDataR3.y, prevDataR4.y);
    prevColorHLen.r += dot(bilateralWeights, prevColorRs);
    prevColorHLen.g += dot(bilateralWeights, prevColorGs);

    uvec4 prevDataG = textureGather(svgfHistory, gatherUV, 1);
    vec2 prevDataG1 = unpackHalf2x16(prevDataG.x);
    vec2 prevDataG2 = unpackHalf2x16(prevDataG.y);
    vec2 prevDataG3 = unpackHalf2x16(prevDataG.z);
    vec2 prevDataG4 = unpackHalf2x16(prevDataG.w);
    vec4 prevColorBs = vec4(prevDataG1.x, prevDataG2.x, prevDataG3.x, prevDataG4.x);
    vec4 prevColorHLens = vec4(prevDataG1.y, prevDataG2.y, prevDataG3.y, prevDataG4.y);
    prevColorHLen.b += dot(bilateralWeights, prevColorBs);
    prevColorHLen.a += dot(bilateralWeights, prevColorHLens);

    uvec4 prevDataB = textureGather(svgfHistory, gatherUV, 2);
    vec2 prevDataB1 = unpackHalf2x16(prevDataB.x);
    vec2 prevDataB2 = unpackHalf2x16(prevDataB.y);
    vec2 prevDataB3 = unpackHalf2x16(prevDataB.z);
    vec2 prevDataB4 = unpackHalf2x16(prevDataB.w);
    vec4 prevMomentsRs = vec4(prevDataB1.x, prevDataB2.x, prevDataB3.x, prevDataB4.x);
    vec4 prevMomentsGs = vec4(prevDataB1.y, prevDataB2.y, prevDataB3.y, prevDataB4.y);
    prevMoments.x += dot(bilateralWeights, prevMomentsRs);
    prevMoments.y += dot(bilateralWeights, prevMomentsGs);
}

void gi_reproject(
usampler2D svgfHistory, usampler2D prevNZTex,
vec2 screenPos, float currViewZ, vec3 currViewNormal, float isHand,
out vec4 prevColorHLen, out vec2 prevMoments
) {
    prevColorHLen = vec4(0.0);
    prevMoments = vec2(0.0);

    vec3 currView = coords_toViewCoord(screenPos, currViewZ, gbufferProjectionInverse);
    vec4 currScene = gbufferModelViewInverse * vec4(currView, 1.0);

    vec4 curr2PrevScene = coord_sceneCurrToPrev(currScene);
    vec4 curr2PrevView = gbufferPrevModelView * curr2PrevScene;
    vec4 curr2PrevClip = gbufferPrevProjection * curr2PrevView;
    vec2 curr2PrevNDC = curr2PrevClip.xy / curr2PrevClip.w;
    vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;
    curr2PrevScreen = mix(curr2PrevScreen, screenPos, isHand);
    if (any(notEqual(curr2PrevScreen, saturate(curr2PrevScreen)))) {
        return;
    }
    vec2 curr2PrevTexel = curr2PrevScreen * global_mainImageSize;
    vec3 currWorldNormal = mat3(gbufferModelViewInverse) * currViewNormal;

    float weightSum = 0.0;

    bilateralSample(
        svgfHistory, prevNZTex,
        curr2PrevTexel, currScene.xyz, currViewZ, currWorldNormal, 1.0,
        prevColorHLen, prevMoments, weightSum
    );
    const float WEIGHT_EPSILON = 0.01;
    if (weightSum < WEIGHT_EPSILON) {
        bilateralSample(
            svgfHistory, prevNZTex,
            curr2PrevTexel + vec2(-1.0, 0.0), currScene.xyz, currViewZ, currWorldNormal, 0.5,
            prevColorHLen, prevMoments, weightSum
        );

        bilateralSample(
            svgfHistory, prevNZTex,
            curr2PrevTexel + vec2(1.0, 0.0), currScene.xyz, currViewZ, currWorldNormal, 0.5,
            prevColorHLen, prevMoments, weightSum
        );

        bilateralSample(
            svgfHistory, prevNZTex,
            curr2PrevTexel + vec2(0.0, -1.0), currScene.xyz, currViewZ, currWorldNormal, 0.5,
            prevColorHLen, prevMoments, weightSum
        );

        bilateralSample(
            svgfHistory, prevNZTex,
            curr2PrevTexel + vec2(0.0, 1.0), currScene.xyz, currViewZ, currWorldNormal, 0.5,
            prevColorHLen, prevMoments, weightSum
        );
    }

    const float WEIGHT_EPSILON_FINAL = 0.0001;
    if (weightSum < WEIGHT_EPSILON_FINAL) {
        prevColorHLen = vec4(0.0);
        prevMoments = vec2(0.0);
    } else {
        float rcpWeightSum = 1.0 / weightSum;
        prevColorHLen *= rcpWeightSum;
        prevMoments *= rcpWeightSum;
        prevColorHLen.a = max(ceil(prevColorHLen.a), 1.0);
    }
}