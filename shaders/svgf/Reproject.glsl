#include "Common.glsl"

vec3 cameraDelta = cameraPosition - previousCameraPosition;

float normalWeight(vec3 currWorldNormal, uint packedNormal) {
    vec3 prevViewNormal = coords_octDecode11(unpackSnorm2x16(packedNormal));
    vec3 prevWorldNormal = mat3(gbufferPrevModelViewInverse) * prevViewNormal;
    float sdot = saturate(dot(currWorldNormal, prevWorldNormal));
    return sdot * sdot * sdot * sdot;
}

float posWeight(float currViewZ, vec3 currScene, vec2 curr2PrevScreen, uint prevViewZI) {
    float prevViewZ = uintBitsToFloat(prevViewZI);
    vec3 prevView = coords_toViewCoord(curr2PrevScreen, prevViewZ, gbufferPrevProjectionInverse);
    vec4 prevScene = gbufferPrevModelViewInverse * vec4(prevView, 1.0);
    prevScene.xyz -= cameraDelta;

    vec3 diff = currScene.xyz - prevScene.xyz;
    float distSq = dot(diff, diff);
    const float a = -currViewZ * 0.00001;
    return a / (a + distSq);
}

void svgf_reproject(
sampler2D svgfHistoryColor, sampler2D svgfHistoryMoments, usampler2D prevNZTex,
vec2 screenPos, float viewZ, vec3 currViewNormal, vec2 projReject, float isHand,
out vec4 prevColorHLen, out vec2 prevMoments
) {
    vec3 currView = coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse);
    vec4 currScene = gbufferModelViewInverse * vec4(currView, 1.0);

    vec4 curr2PrevScene = coord_sceneCurrToPrev(currScene);
    vec4 curr2PrevView = gbufferPrevModelView * curr2PrevScene;
    vec4 curr2PrevClip = gbufferPrevProjection * curr2PrevView;
    vec2 curr2PrevNDC = curr2PrevClip.xy / curr2PrevClip.w;
    vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;
    curr2PrevScreen = mix(curr2PrevScreen, screenPos, isHand);

    vec2 pixelPos = curr2PrevScreen * global_mainImageSize - 0.5;
    vec2 originPixelPos = floor(pixelPos);
    vec2 gatherUV = (originPixelPos + 1.0) * global_mainImageSizeRcp;
    vec2 bilinearWeights = pixelPos - originPixelPos;

    vec3 currWorldNormal = mat3(gbufferModelViewInverse) * currViewNormal;

    vec4 bilateralWeights;
    bilateralWeights.yz = bilinearWeights.xx;
    bilateralWeights.xw = 1.0 - bilinearWeights.xx;
    bilateralWeights.xy *= bilinearWeights.yy;
    bilateralWeights.zw *= 1.0 - bilinearWeights.yy;
    bilateralWeights += 0.01;

    uvec4 prevNs = textureGather(prevNZTex, gatherUV, 0);
    bilateralWeights.x *= normalWeight(currWorldNormal, prevNs.x);
    bilateralWeights.y *= normalWeight(currWorldNormal, prevNs.y);
    bilateralWeights.z *= normalWeight(currWorldNormal, prevNs.z);
    bilateralWeights.w *= normalWeight(currWorldNormal, prevNs.w);

    uvec4 prevViewZs = textureGather(prevNZTex, gatherUV, 1);
    bilateralWeights.x *= posWeight(viewZ, currScene.xyz, gatherUV, prevViewZs.x);
    bilateralWeights.y *= posWeight(viewZ, currScene.xyz, gatherUV, prevViewZs.y);
    bilateralWeights.z *= posWeight(viewZ, currScene.xyz, gatherUV, prevViewZs.z);
    bilateralWeights.w *= posWeight(viewZ, currScene.xyz, gatherUV, prevViewZs.w);

    float weightSum = bilateralWeights.x + bilateralWeights.y + bilateralWeights.z + bilateralWeights.w;
    const float WEIGHT_EPSILON = 0.0001;

    if (weightSum < WEIGHT_EPSILON) {
        prevColorHLen = vec4(0.0, 0.0, 0.0, 1.0);
        prevMoments = vec2(0.0);
    } else {
        float rcpWeightSum = 1.0 / weightSum;

        vec4 prevColorRs = textureGather(svgfHistoryColor, gatherUV, 0);
        vec4 prevColorGs = textureGather(svgfHistoryColor, gatherUV, 1);
        vec4 prevColorBs = textureGather(svgfHistoryColor, gatherUV, 2);
        vec4 prevColorHLens = textureGather(svgfHistoryColor, gatherUV, 3);

        prevColorHLen.r = dot(bilateralWeights, prevColorRs);
        prevColorHLen.g = dot(bilateralWeights, prevColorGs);
        prevColorHLen.b = dot(bilateralWeights, prevColorBs);
        prevColorHLen.a = dot(bilateralWeights, prevColorHLens);
        prevColorHLen *= rcpWeightSum;
        prevColorHLen *= saturate(1.0 - projReject.x * 0.1);
        prevColorHLen.a = max(floor(prevColorHLen.a), 0.0);

        vec4 prevMomentXs = textureGather(svgfHistoryMoments, gatherUV, 0);
        vec4 prevMomentYs = textureGather(svgfHistoryMoments, gatherUV, 1);

        prevMoments.x = dot(bilateralWeights, prevMomentXs);
        prevMoments.y = dot(bilateralWeights, prevMomentYs);
        prevMoments *= rcpWeightSum;
    }
}