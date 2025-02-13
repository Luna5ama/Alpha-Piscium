#include "Common.glsl"
#include "/util/Coords.glsl"

vec3 cameraDelta = cameraPosition - previousCameraPosition;

float normalWeight(vec3 currWorldNormal, uint packedNormal) {
    vec3 prevViewNormal = coords_octDecode11(unpackSnorm2x16(packedNormal));
    vec3 prevWorldNormal = mat3(gbufferPrevModelViewInverse) * prevViewNormal;
    float sdot = saturate(dot(currWorldNormal, prevWorldNormal));
    return pow(sdot, 2.0);
}

float posWeight(float currViewZ, vec3 currScene, vec2 curr2PrevScreen, uint prevViewZI) {
    float prevViewZ = uintBitsToFloat(prevViewZI);
    vec3 prevView = coords_toViewCoord(curr2PrevScreen, prevViewZ, gbufferPrevProjectionInverse);
    vec4 prevScene = gbufferPrevModelViewInverse * vec4(prevView, 1.0);
    prevScene.xyz -= cameraDelta;

    vec3 diff = currScene.xyz - prevScene.xyz;
    float distSq = dot(diff, diff);
    float a = -currViewZ * 0.001;
    return a / (a + distSq);
}

void bilateralSample(
sampler2D svgfHistoryColor, sampler2D svgfHistoryMoments, usampler2D prevNZTex,
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
    bilateralWeights += 0.01;

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

    vec4 prevColorRs = textureGather(svgfHistoryColor, gatherUV, 0);
    vec4 prevColorGs = textureGather(svgfHistoryColor, gatherUV, 1);
    vec4 prevColorBs = textureGather(svgfHistoryColor, gatherUV, 2);
    vec4 prevColorHLens = textureGather(svgfHistoryColor, gatherUV, 3);

    prevColorHLen.r += dot(bilateralWeights, prevColorRs);
    prevColorHLen.g += dot(bilateralWeights, prevColorGs);
    prevColorHLen.b += dot(bilateralWeights, prevColorBs);
    prevColorHLen.a += dot(bilateralWeights, prevColorHLens);

    vec4 prevMomentXs = textureGather(svgfHistoryMoments, gatherUV, 0);
    vec4 prevMomentYs = textureGather(svgfHistoryMoments, gatherUV, 1);

    prevMoments.x += dot(bilateralWeights, prevMomentXs);
    prevMoments.y += dot(bilateralWeights, prevMomentYs);
}

void svgf_reproject(
sampler2D svgfHistoryColor, sampler2D svgfHistoryMoments, usampler2D prevNZTex,
vec2 screenPos, float currViewZ, vec3 currViewNormal, float isHand,
out vec4 prevColorHLen, out vec2 prevMoments
) {
    vec3 currView = coords_toViewCoord(screenPos, currViewZ, gbufferProjectionInverse);
    vec4 currScene = gbufferModelViewInverse * vec4(currView, 1.0);

    vec4 curr2PrevScene = coord_sceneCurrToPrev(currScene);
    vec4 curr2PrevView = gbufferPrevModelView * curr2PrevScene;
    vec4 curr2PrevClip = gbufferPrevProjection * curr2PrevView;
    vec2 curr2PrevNDC = curr2PrevClip.xy / curr2PrevClip.w;
    vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;
    curr2PrevScreen = mix(curr2PrevScreen, screenPos, isHand);
    vec2 curr2PrevTexel = curr2PrevScreen * global_mainImageSize;

    vec3 currWorldNormal = mat3(gbufferModelViewInverse) * currViewNormal;

    prevColorHLen = vec4(0.0);
    prevMoments = vec2(0.0);
    float weightSum = 0.0;

    #ifdef SETTING_DENOISER_REPROJ_FILTER
    bilateralSample(
        svgfHistoryColor, svgfHistoryMoments, prevNZTex,
        curr2PrevTexel + vec2(0.5), currScene.xyz, currViewZ, currWorldNormal, 1.0,
        prevColorHLen, prevMoments, weightSum
    );

    bilateralSample(
        svgfHistoryColor, svgfHistoryMoments, prevNZTex,
        curr2PrevTexel - vec2(0.5), currScene.xyz, currViewZ, currWorldNormal, 1.0,
        prevColorHLen, prevMoments, weightSum
    );

    bilateralSample(
        svgfHistoryColor, svgfHistoryMoments, prevNZTex,
        curr2PrevTexel + vec2(-0.5, 0.5), currScene.xyz, currViewZ, currWorldNormal, 1.0,
        prevColorHLen, prevMoments, weightSum
    );

    bilateralSample(
        svgfHistoryColor, svgfHistoryMoments, prevNZTex,
        curr2PrevTexel + vec2(0.5, -0.5), currScene.xyz, currViewZ, currWorldNormal, 1.0,
        prevColorHLen, prevMoments, weightSum
    );
    #else
    bilateralSample(
        svgfHistoryColor, svgfHistoryMoments, prevNZTex,
        curr2PrevTexel, currScene.xyz, currViewZ, currWorldNormal, 1.0,
        prevColorHLen, prevMoments, weightSum
    );

    const float WEIGHT_EPSILON = 0.01;
    if (weightSum < WEIGHT_EPSILON) {
        bilateralSample(
            svgfHistoryColor, svgfHistoryMoments, prevNZTex,
            curr2PrevTexel + vec2(-1.0, 0.0), currScene.xyz, currViewZ, currWorldNormal, 0.5,
            prevColorHLen, prevMoments, weightSum
        );

        bilateralSample(
            svgfHistoryColor, svgfHistoryMoments, prevNZTex,
            curr2PrevTexel + vec2(1.0, 0.0), currScene.xyz, currViewZ, currWorldNormal, 0.5,
            prevColorHLen, prevMoments, weightSum
        );

        bilateralSample(
            svgfHistoryColor, svgfHistoryMoments, prevNZTex,
            curr2PrevTexel + vec2(0.0, -1.0), currScene.xyz, currViewZ, currWorldNormal, 0.5,
            prevColorHLen, prevMoments, weightSum
        );

        bilateralSample(
            svgfHistoryColor, svgfHistoryMoments, prevNZTex,
            curr2PrevTexel + vec2(0.0, 1.0), currScene.xyz, currViewZ, currWorldNormal, 0.5,
            prevColorHLen, prevMoments, weightSum
        );
    }
    #endif

    const float WEIGHT_EPSILON_FINAL = 0.0001;
    if (weightSum < WEIGHT_EPSILON_FINAL) {
        prevColorHLen = vec4(0.0);
        prevMoments = vec2(0.0);
    } else {
        float rcpWeightSum = 1.0 / weightSum;
        prevColorHLen *= rcpWeightSum;
        prevColorHLen.a = max(ceil(prevColorHLen.a), 1.0);
        prevMoments *= rcpWeightSum;
    }
}