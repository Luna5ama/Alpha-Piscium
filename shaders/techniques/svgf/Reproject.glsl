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
vec3 reproject_currToPrevViewNormal;
vec3 reproject_currToPrevViewGeomNormal;
bool reproject_isHand;

float computeNormalWeight(uint packedNormal, float weight) {
    vec3 prevViewNormal = coords_octDecode11(unpackSnorm2x16(packedNormal));
    float sdot = saturate(dot(reproject_currToPrevViewNormal, prevViewNormal));
    return pow(sdot, weight);
}

float computeGeometryWeight(
vec2 curr2PrevScreen, float prevViewZ, uint packedPrevViewGeomNormal,
float planeWeight, float normalWeight
) {
    vec3 prevView = coords_toViewCoord(curr2PrevScreen, prevViewZ, global_prevCamProjInverse);

    vec3 prevViewGeomNormal = unpackSnorm3x10(packedPrevViewGeomNormal);

    vec3 posDiff = reproject_curr2PrevView.xyz - prevView.xyz;
    float planeDist1 = abs(dot(posDiff, reproject_currToPrevViewGeomNormal));
    float planeDist2 = abs(dot(posDiff, prevViewGeomNormal));
    float maxPlaneDist = pow2(max(planeDist1, planeDist2));
    float normalDot = saturate(dot(reproject_currToPrevViewGeomNormal, prevViewGeomNormal));

    float result = planeWeight / (planeWeight + maxPlaneDist);
    result *= pow(normalDot, normalWeight);
    
    return result;
}

vec4 computeBilateralWeights(vec2 gatherTexelPos) {
    vec2 screenPos = gatherTexelPos * global_mainImageSizeRcp;
    vec2 gatherUV = nzpacking_fullResGatherUV(gatherTexelPos);
    vec4 result = vec4(1.0);

    uvec4 prevViewNormals = textureGather(usam_packedZN, gatherUV, 0);
    uvec4 prevViewGeomNormals = textureGather(usam_geometryNormal, screenPos, 0);
    vec4 prevViewZs = uintBitsToFloat(textureGather(usam_packedZN, gatherUV, 1));

    float geometryPlaneWeight = BASE_GEOM_WEIGHT_RCP * max(abs(reproject_curr2PrevView.z), 0.1);
    float geometryNormalWeight = mix(BASE_GEOM_WEIGHT, BASE_NORMAL_WEIGHT, reproject_isHand);

    result.x *= computeGeometryWeight(
        screenPos + global_mainImageSizeRcp * vec2(-0.5, 0.5), prevViewZs.x, prevViewGeomNormals.x,
        geometryPlaneWeight, geometryNormalWeight
    );
    result.y *= computeGeometryWeight(
        screenPos + global_mainImageSizeRcp * vec2(0.5, 0.5), prevViewZs.y, prevViewGeomNormals.y,
        geometryPlaneWeight, geometryNormalWeight
    );
    result.z *= computeGeometryWeight(
        screenPos + global_mainImageSizeRcp * vec2(0.5, -0.5), prevViewZs.z, prevViewGeomNormals.z,
        geometryPlaneWeight, geometryNormalWeight
    );
    result.w *= computeGeometryWeight(
        screenPos + global_mainImageSizeRcp * vec2(-0.5, -0.5), prevViewZs.w, prevViewGeomNormals.w,
        geometryPlaneWeight, geometryNormalWeight
    );

    result.x *= computeNormalWeight(prevViewNormals.x, BASE_NORMAL_WEIGHT);
    result.y *= computeNormalWeight(prevViewNormals.y, BASE_NORMAL_WEIGHT);
    result.z *= computeNormalWeight(prevViewNormals.z, BASE_NORMAL_WEIGHT);
    result.w *= computeNormalWeight(prevViewNormals.w, BASE_NORMAL_WEIGHT);

    return result;
}

void bilateralSample(
vec2 gatherTexelPos, vec4 baseWeights,
inout vec3 prevColor, inout vec3 prevFastColor, inout vec2 prevMoments, inout float prevHLen, inout float weightSum
) {
    vec2 gatherUV1 = gi_diffuseHistory_texelToGatherUV(gatherTexelPos);

    vec4 bilateralWeights = computeBilateralWeights(gatherTexelPos);
    float bilateralWeightSum = bilateralWeights.x + bilateralWeights.y + bilateralWeights.z + bilateralWeights.w;

    if (bilateralWeightSum > BILATERAL_WEIGHT_SUM_EPSILON) {
        vec4 interpoWeights = baseWeights * bilateralWeights;
        weightSum += interpoWeights.x + interpoWeights.y + interpoWeights.z + interpoWeights.w;

        {
            uvec4 prevColorData = textureGather(usam_csrgba32ui, gatherUV1, 0);
            vec2 temp1 = max(unpackHalf2x16(prevColorData.x), 0.0);
            vec2 temp2 = max(unpackHalf2x16(prevColorData.y), 0.0);
            vec2 temp3 = max(unpackHalf2x16(prevColorData.z), 0.0);
            vec2 temp4 = max(unpackHalf2x16(prevColorData.w), 0.0);

            vec4 prevColorR = vec4(temp1.r, temp2.r, temp3.r, temp4.r);
            prevColor.r += dot(interpoWeights, prevColorR);
            vec4 prevColorG = vec4(temp1.g, temp2.g, temp3.g, temp4.g);
            prevColor.g += dot(interpoWeights, prevColorG);
        }

        {
            uvec4 prevColorData = textureGather(usam_csrgba32ui, gatherUV1, 1);
            vec2 temp1 = max(unpackHalf2x16(prevColorData.x), 0.0);
            vec2 temp2 = max(unpackHalf2x16(prevColorData.y), 0.0);
            vec2 temp3 = max(unpackHalf2x16(prevColorData.z), 0.0);
            vec2 temp4 = max(unpackHalf2x16(prevColorData.w), 0.0);

            vec4 prevColorR = vec4(temp1.x, temp2.x, temp3.x, temp4.x);
            prevColor.b += dot(interpoWeights, prevColorR);
            vec4 prevMoment2 = vec4(temp1.y, temp2.y, temp3.y, temp4.y);
            prevMoments.y += dot(interpoWeights, prevMoment2);
        }

        {
            uvec4 prevData = textureGather(usam_csrgba32ui, gatherUV1, 2);
            vec2 temp1 = max(unpackHalf2x16(prevData.x), 0.0);
            vec2 temp2 = max(unpackHalf2x16(prevData.y), 0.0);
            vec2 temp3 = max(unpackHalf2x16(prevData.z), 0.0);
            vec2 temp4 = max(unpackHalf2x16(prevData.w), 0.0);

            vec4 prevFastColorR = vec4(temp1.x, temp2.x, temp3.x, temp4.x);
            prevFastColor.r += dot(interpoWeights, prevFastColorR);
            vec4 prevFastColorG = vec4(temp1.y, temp2.y, temp3.y, temp4.y);
            prevFastColor.g += dot(interpoWeights, prevFastColorG);
        }

        {
            uvec4 prevData = textureGather(usam_csrgba32ui, gatherUV1, 3);
            vec2 temp1 = max(unpackHalf2x16(prevData.x), 0.0);
            vec2 temp2 = max(unpackHalf2x16(prevData.y), 0.0);
            vec2 temp3 = max(unpackHalf2x16(prevData.z), 0.0);
            vec2 temp4 = max(unpackHalf2x16(prevData.w), 0.0);

            vec4 prevFastColorR = vec4(temp1.x, temp2.x, temp3.x, temp4.x);
            prevFastColor.b += dot(interpoWeights, prevFastColorR);
            vec4 prevHLenV = vec4(temp1.y, temp2.y, temp3.y, temp4.y);
            prevHLen += dot(interpoWeights, prevHLenV);
        }
    }
}

void gi_reproject(
vec2 screenPos, float currViewZ, vec3 currViewNormal, vec3 currViewGeomNormal, bool isHand,
out vec3 prevColor, out vec3 prevFastColor, out vec2 prevMoments, out float prevHLen
) {
    reproject_isHand = isHand;

    prevColor = vec3(0.0);
    prevFastColor = vec3(0.0);
    prevMoments = vec2(0.0);
    prevHLen = 0.0;

    vec3 currView = coords_toViewCoord(screenPos, currViewZ, global_camProjInverse);
    vec4 currScene = gbufferModelViewInverse * vec4(currView, 1.0);
    vec4 curr2PrevView = coord_viewCurrToPrev(vec4(currView, 1.0), isHand);
    vec4 curr2PrevClip = global_prevCamProj * curr2PrevView;
    uint clipFlag = uint(curr2PrevClip.z > 0.0);
    clipFlag &= uint(all(lessThan(abs(curr2PrevClip.xy), curr2PrevClip.ww)));
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

    vec3 currWorldGeomNormal = mat3(gbufferModelViewInverse) * currViewGeomNormal;
    vec3 currToPrevViewGeomNormal = mat3(gbufferPrevModelView) * currWorldGeomNormal;

    vec2 centerPixel = curr2PrevTexel - 0.5;
    vec2 centerPixelOrigin = floor(centerPixel);
    vec2 gatherTexelPos = centerPixelOrigin + 1.0;
    vec2 pixelPosFract = centerPixel - centerPixelOrigin;

    reproject_curr2PrevView = curr2PrevView.xyz;
    reproject_currToPrevViewNormal = currToPrevViewNormal;
    reproject_currToPrevViewGeomNormal = currToPrevViewGeomNormal;

    vec4 centerWeights = computeBilateralWeights(gatherTexelPos);

    float weightSum = 0.0;
    uint flag = uint(any(lessThan(centerWeights, vec4(CUBIE_SAMPLE_WEIGHT_EPSILON))));
    flag |= uint(any(lessThan(curr2PrevTexel, vec2(1.0))));
    flag |= uint(any(greaterThan(curr2PrevTexel, global_mainImageSize - 1.0)));

    vec4 weights1;
    vec4 weights2;
    vec4 weights3;
    vec4 weights4;

    if (bool(flag)) {
        vec4 weightX = sampling_bSplineWeights(pixelPosFract.x);
        vec4 weightY = sampling_bSplineWeights(pixelPosFract.y);

        vec2 bilinearWeights2 = pixelPosFract;
        vec4 blinearWeights4;
        blinearWeights4.yz = bilinearWeights2.xx;
        blinearWeights4.xw = 1.0 - bilinearWeights2.xx;
        blinearWeights4.xy *= bilinearWeights2.yy;
        blinearWeights4.zw *= 1.0 - bilinearWeights2.yy;

        weights1 = weightX.xyyx * weightY.wwzz;
        weights1.z += blinearWeights4.x;

        weights2 = weightX.zwwz * weightY.wwzz;
        weights2.w += blinearWeights4.y;

        weights3 = weightX.zwwz * weightY.yyxx;
        weights3.x += blinearWeights4.z;

        weights4 = weightX.xyyx * weightY.yyxx;
        weights4.y += blinearWeights4.w;
    } else {
        vec4 weightX = sampling_catmullRomWeights(pixelPosFract.x);
        vec4 weightY = sampling_catmullRomWeights(pixelPosFract.y);

        weights1 = weightX.xyyx * weightY.wwzz;
        weights2 = weightX.zwwz * weightY.wwzz;
        weights3 = weightX.zwwz * weightY.yyxx;
        weights4 = weightX.xyyx * weightY.yyxx;
    }

    bilateralSample(
        gatherTexelPos + vec2(-1.0, 1.0), weights1,
        prevColor, prevFastColor, prevMoments, prevHLen, weightSum
    );

    bilateralSample(
        gatherTexelPos + vec2(1.0, 1.0), weights2,
        prevColor, prevFastColor, prevMoments, prevHLen, weightSum
    );

    bilateralSample(
        gatherTexelPos + vec2(1.0, -1.0), weights3,
        prevColor, prevFastColor, prevMoments, prevHLen, weightSum
    );

    bilateralSample(
        gatherTexelPos + vec2(-1.0, -1.0), weights4,
        prevColor, prevFastColor, prevMoments, prevHLen, weightSum
    );

    if (weightSum < FINAL_WEIGHT_SUM_EPSILON) {
        prevColor = vec3(0.0);
        prevFastColor = vec3(0.0);
        prevMoments = vec2(0.0);
        prevHLen = 0.0;
    } else {
        float rcpWeightSum = 1.0 / weightSum;
        prevColor = clamp(prevColor * rcpWeightSum, 0.0, FP16_MAX);
        prevFastColor = clamp(prevFastColor * rcpWeightSum, 0.0, FP16_MAX);
        prevMoments = clamp(prevMoments * rcpWeightSum, 0.0, FP16_MAX);
        prevMoments.x = min(colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, prevColor), 256.0);
        prevHLen = clamp(ceil(prevHLen * rcpWeightSum), 0.0, FP16_MAX);
    }
}