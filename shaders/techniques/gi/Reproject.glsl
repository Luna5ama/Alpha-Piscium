#include "Common.glsl"
#include "/util/BitPacking.glsl"
#include "/util/Coords.glsl"
#include "/util/Colors.glsl"
#include "/util/Colors2.glsl"
#include "/util/Sampling.glsl"
#include "/util/NZPacking.glsl"

const float BASE_GEOM_WEIGHT = exp2(SETTING_DENOISER_REPROJ_GEOMETRY_EDGE_WEIGHT);
const float BASE_GEOM_WEIGHT_RCP = rcp(exp2(SETTING_DENOISER_REPROJ_GEOMETRY_EDGE_WEIGHT));
const float BASE_NORMAL_WEIGHT = exp2(SETTING_DENOISER_REPROJ_NORMAL_EDGE_WEIGHT);

const float CUBIE_SAMPLE_WEIGHT_EPSILON = 0.5;
const float FINAL_WEIGHT_SUM_EPSILON = 0.1;
const float BILATERAL_WEIGHT_SUM_EPSILON = 0.1;

vec3 reproject_curr2PrevView;
vec3 reproject_currWorldNormal;
vec3 reproject_currWorldGeomNormal;
bool reproject_isHand;

float computeNormalWeight(vec3 prevWorldNormal, float weight) {
    float sdot = saturate(dot(reproject_currWorldNormal, prevWorldNormal));
    return pow(sdot, weight);
}

float computeGeometryWeight(
vec2 curr2PrevScreen, float prevViewZ, vec3 prevGeomViewNormal,
float planeWeight, float normalWeight
) {
    vec3 prevView = coords_toViewCoord(curr2PrevScreen, prevViewZ, global_prevCamProjInverse);

    vec3 posDiff = reproject_curr2PrevView.xyz - prevView.xyz;
    float planeDist1 = abs(dot(posDiff, reproject_currWorldGeomNormal));
    float planeDist2 = abs(dot(posDiff, prevGeomViewNormal));
    float maxPlaneDist = pow2(max(planeDist1, planeDist2));
    float normalDot = saturate(dot(reproject_currWorldGeomNormal, prevGeomViewNormal));

    float result = planeWeight / (planeWeight + maxPlaneDist);
    result *= pow(normalDot, normalWeight);
    
    return result;
}

vec4 computeBilateralWeights(vec2 gatherTexelPos) {
    vec2 screenPos = gatherTexelPos * uval_mainImageSizeRcp;
    vec4 result = vec4(1.0);

    vec3 prevWorldNormal1 = history_worldNormal_fetch(ivec2(gatherTexelPos + vec2(-0.5, 0.5))).xyz * 2.0 - 1.0;
    vec3 prevWorldNormal2 = history_worldNormal_fetch(ivec2(gatherTexelPos + vec2(0.5, 0.5))).xyz * 2.0 - 1.0;
    vec3 prevWorldNormal3 = history_worldNormal_fetch(ivec2(gatherTexelPos + vec2(0.5, -0.5))).xyz * 2.0 - 1.0;
    vec3 prevWorldNormal4 = history_worldNormal_fetch(ivec2(gatherTexelPos + vec2(-0.5, -0.5))).xyz * 2.0 - 1.0;
    
    vec3 prevGeomWorldNormal1 = history_geomWorldNormal_fetch(ivec2(gatherTexelPos + vec2(-0.5, 0.5))).xyz * 2.0 - 1.0;
    vec3 prevGeomWorldNormal2 = history_geomWorldNormal_fetch(ivec2(gatherTexelPos + vec2(0.5, 0.5))).xyz * 2.0 - 1.0;
    vec3 prevGeomWorldNormal3 = history_geomWorldNormal_fetch(ivec2(gatherTexelPos + vec2(0.5, -0.5))).xyz * 2.0 - 1.0;
    vec3 prevGeomWorldNormal4 = history_geomWorldNormal_fetch(ivec2(gatherTexelPos + vec2(-0.5, -0.5))).xyz * 2.0 - 1.0;

    vec4 prevViewZs = history_viewZ_gather(screenPos, 1);

    float geometryPlaneWeight = BASE_GEOM_WEIGHT_RCP * max(abs(reproject_curr2PrevView.z), 0.1);
    float geometryNormalWeight = mix(BASE_GEOM_WEIGHT, BASE_NORMAL_WEIGHT, reproject_isHand);

    result.x *= computeGeometryWeight(
        screenPos + uval_mainImageSizeRcp * vec2(-0.5, 0.5), prevViewZs.x, prevGeomWorldNormal1,
        geometryPlaneWeight, geometryNormalWeight
    );
    result.y *= computeGeometryWeight(
        screenPos + uval_mainImageSizeRcp * vec2(0.5, 0.5), prevViewZs.y, prevGeomWorldNormal2,
        geometryPlaneWeight, geometryNormalWeight
    );
    result.z *= computeGeometryWeight(
        screenPos + uval_mainImageSizeRcp * vec2(0.5, -0.5), prevViewZs.z, prevGeomWorldNormal3,
        geometryPlaneWeight, geometryNormalWeight
    );
    result.w *= computeGeometryWeight(
        screenPos + uval_mainImageSizeRcp * vec2(-0.5, -0.5), prevViewZs.w, prevGeomWorldNormal4,
        geometryPlaneWeight, geometryNormalWeight
    );

    result.x *= computeNormalWeight(prevWorldNormal1, BASE_NORMAL_WEIGHT);
    result.y *= computeNormalWeight(prevWorldNormal2, BASE_NORMAL_WEIGHT);
    result.z *= computeNormalWeight(prevWorldNormal3, BASE_NORMAL_WEIGHT);
    result.w *= computeNormalWeight(prevWorldNormal4, BASE_NORMAL_WEIGHT);

    return result;
}

void bilateralSample(
vec2 gatherTexelPos, vec4 baseWeights,
inout vec3 prevColor, inout vec3 prevFastColor, inout vec2 prevMoments, inout float prevHLen, inout float weightSum
) {
    vec4 bilateralWeights = computeBilateralWeights(gatherTexelPos);
    float bilateralWeightSum = bilateralWeights.x + bilateralWeights.y + bilateralWeights.z + bilateralWeights.w;

    if (bilateralWeightSum > BILATERAL_WEIGHT_SUM_EPSILON) {
        vec4 interpoWeights = baseWeights * bilateralWeights;
        weightSum += interpoWeights.x + interpoWeights.y + interpoWeights.z + interpoWeights.w;

        {
            uvec4 prevColorData = history_gi_gather(gatherTexelPos, 0);
            vec2 temp1 = unpackHalf2x16(prevColorData.x);
            vec2 temp2 = unpackHalf2x16(prevColorData.y);
            vec2 temp3 = unpackHalf2x16(prevColorData.z);
            vec2 temp4 = unpackHalf2x16(prevColorData.w);

            vec4 prevColorR = vec4(temp1.r, temp2.r, temp3.r, temp4.r);
            prevColor.r += dot(interpoWeights, prevColorR);
            vec4 prevColorG = vec4(temp1.g, temp2.g, temp3.g, temp4.g);
            prevColor.g += dot(interpoWeights, prevColorG);
        }

        {
            uvec4 prevColorData = history_gi_gather(gatherTexelPos, 1);
            vec2 temp1 = unpackHalf2x16(prevColorData.x);
            vec2 temp2 = unpackHalf2x16(prevColorData.y);
            vec2 temp3 = unpackHalf2x16(prevColorData.z);
            vec2 temp4 = unpackHalf2x16(prevColorData.w);

            vec4 prevColorR = vec4(temp1.x, temp2.x, temp3.x, temp4.x);
            prevColor.b += dot(interpoWeights, prevColorR);
            vec4 prevMoment2 = vec4(temp1.y, temp2.y, temp3.y, temp4.y);
            prevMoments.y += dot(interpoWeights, prevMoment2);
        }

        {
            uvec4 prevData = history_gi_gather(gatherTexelPos, 2);
            vec2 temp1 = unpackHalf2x16(prevData.x);
            vec2 temp2 = unpackHalf2x16(prevData.y);
            vec2 temp3 = unpackHalf2x16(prevData.z);
            vec2 temp4 = unpackHalf2x16(prevData.w);

            vec4 prevFastColorR = vec4(temp1.x, temp2.x, temp3.x, temp4.x);
            prevFastColor.r += dot(interpoWeights, prevFastColorR);
            vec4 prevFastColorG = vec4(temp1.y, temp2.y, temp3.y, temp4.y);
            prevFastColor.g += dot(interpoWeights, prevFastColorG);
        }

        {
            uvec4 prevData = history_gi_gather(gatherTexelPos, 3);
            vec2 temp1 = unpackHalf2x16(prevData.x);
            vec2 temp2 = unpackHalf2x16(prevData.y);
            vec2 temp3 = unpackHalf2x16(prevData.z);
            vec2 temp4 = unpackHalf2x16(prevData.w);

            vec4 prevFastColorR = vec4(temp1.x, temp2.x, temp3.x, temp4.x);
            prevFastColor.b += dot(interpoWeights, prevFastColorR);
            vec4 prevHLenV = vec4(temp1.y, temp2.y, temp3.y, temp4.y);
            prevHLen += dot(interpoWeights, prevHLenV);
        }
    }
}

GIHistoryData gi_reproject(vec2 screenPos, float currViewZ, vec3 currViewNormal, vec3 currViewGeomNormal, bool isHand){
    reproject_isHand = isHand;

    GIHistoryData historyData = gi_historyData_init();

    vec3 currView = coords_toViewCoord(screenPos, currViewZ, global_camProjInverse);
    vec4 currScene = gbufferModelViewInverse * vec4(currView, 1.0);
    vec4 curr2PrevView = coord_viewCurrToPrev(vec4(currView, 1.0), isHand);
    vec4 curr2PrevClip = global_prevCamProj * curr2PrevView;
    uint clipFlag = uint(curr2PrevClip.z > 0.0);
    clipFlag &= uint(all(lessThan(abs(curr2PrevClip.xy), curr2PrevClip.ww)));
    if (!bool(clipFlag)) {
        return historyData;
    }

    vec2 curr2PrevNDC = curr2PrevClip.xy / curr2PrevClip.w;
    vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;

    if (any(notEqual(curr2PrevScreen, saturate(curr2PrevScreen)))) {
        return historyData;
    }
    vec2 curr2PrevTexel = curr2PrevScreen * uval_mainImageSize;
    vec3 currWorldNormal = mat3(gbufferModelViewInverse) * currViewNormal;
    vec3 currToPrevViewNormal = mat3(gbufferPrevModelView) * currWorldNormal;

    vec3 currWorldGeomNormal = mat3(gbufferModelViewInverse) * currViewGeomNormal;
    vec3 currToPrevViewGeomNormal = mat3(gbufferPrevModelView) * currWorldGeomNormal;

    vec2 centerPixel = curr2PrevTexel - 0.5;
    vec2 centerPixelOrigin = floor(centerPixel);
    vec2 gatherTexelPos = centerPixelOrigin + 1.0;
    vec2 pixelPosFract = centerPixel - centerPixelOrigin;

    reproject_curr2PrevView = curr2PrevView.xyz;
    reproject_currWorldNormal = currWorldNormal;
    reproject_currWorldGeomNormal = currWorldGeomNormal;

    vec4 centerWeights = computeBilateralWeights(gatherTexelPos);

//    float weightSum = 0.0;
//    uint flag = uint(any(lessThan(centerWeights, vec4(CUBIE_SAMPLE_WEIGHT_EPSILON))));
//    flag |= uint(any(lessThan(curr2PrevTexel, vec2(1.0))));
//    flag |= uint(any(greaterThan(curr2PrevTexel, uval_mainImageSize - 1.0)));
//
//    vec4 weights1;
//    vec4 weights2;
//    vec4 weights3;
//    vec4 weights4;
//
//    if (bool(flag)) {
//        vec4 weightX = sampling_bSplineWeights(pixelPosFract.x);
//        vec4 weightY = sampling_bSplineWeights(pixelPosFract.y);
//
//        vec2 bilinearWeights2 = pixelPosFract;
//        vec4 blinearWeights4;
//        blinearWeights4.yz = bilinearWeights2.xx;
//        blinearWeights4.xw = 1.0 - bilinearWeights2.xx;
//        blinearWeights4.xy *= bilinearWeights2.yy;
//        blinearWeights4.zw *= 1.0 - bilinearWeights2.yy;
//
//        weights1 = weightX.xyyx * weightY.wwzz;
//        weights1.z += blinearWeights4.x;
//
//        weights2 = weightX.zwwz * weightY.wwzz;
//        weights2.w += blinearWeights4.y;
//
//        weights3 = weightX.zwwz * weightY.yyxx;
//        weights3.x += blinearWeights4.z;
//
//        weights4 = weightX.xyyx * weightY.yyxx;
//        weights4.y += blinearWeights4.w;
//    } else {
//        vec4 weightX = sampling_catmullRomWeights(pixelPosFract.x);
//        vec4 weightY = sampling_catmullRomWeights(pixelPosFract.y);
//
//        weights1 = weightX.xyyx * weightY.wwzz;
//        weights2 = weightX.zwwz * weightY.wwzz;
//        weights3 = weightX.zwwz * weightY.yyxx;
//        weights4 = weightX.xyyx * weightY.yyxx;
//    }
//
//    bilateralSample(
//        gatherTexelPos + vec2(-1.0, 1.0), weights1,
//        prevColor, prevFastColor, prevMoments, prevHLen, weightSum
//    );
//
//    bilateralSample(
//        gatherTexelPos + vec2(1.0, 1.0), weights2,
//        prevColor, prevFastColor, prevMoments, prevHLen, weightSum
//    );
//
//    bilateralSample(
//        gatherTexelPos + vec2(1.0, -1.0), weights3,
//        prevColor, prevFastColor, prevMoments, prevHLen, weightSum
//    );
//
//    bilateralSample(
//        gatherTexelPos + vec2(-1.0, -1.0), weights4,
//        prevColor, prevFastColor, prevMoments, prevHLen, weightSum
//    );
//
//    if (weightSum < FINAL_WEIGHT_SUM_EPSILON) {
//        prevColor = vec3(0.0);
//        prevFastColor = vec3(0.0);
//        prevMoments = vec2(0.0);
//        prevHLen = 0.0;
//    } else {
//        float rcpWeightSum = 1.0 / weightSum;
//        prevColor = clamp(prevColor * rcpWeightSum, 0.0, FP16_MAX);
//        prevFastColor = clamp(prevFastColor * rcpWeightSum, 0.0, FP16_MAX);
//        prevMoments = clamp(prevMoments * rcpWeightSum, 0.0, FP16_MAX);
//        prevMoments.x = min(colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, prevColor), 256.0);
//        prevHLen = clamp(ceil(prevHLen * rcpWeightSum), 0.0, FP16_MAX);
//    }

    return historyData;
}