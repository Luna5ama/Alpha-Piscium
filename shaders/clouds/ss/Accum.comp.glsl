#extension GL_KHR_shader_subgroup_basic : enable

#include "Common.glsl"
#include "/util/Coords.glsl"
#include "/util/Sampling.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32ui) uniform restrict writeonly uimage2D uimg_tempRGBA32UI;

struct Vec4PackedData {
    vec4 inScatteringViewZ;
    vec4 transmittanceHLen;
};

Vec4PackedData vec4PackedData_init() {
    Vec4PackedData data;
    data.inScatteringViewZ = vec4(0.0);
    data.transmittanceHLen = vec4(0.0);
    return data;
}

Vec4PackedData vec4PackedData_fromHistoryData(CloudSSHistoryData historyData) {
    Vec4PackedData data;
    data.inScatteringViewZ = vec4(historyData.inScattering, historyData.viewZ);
    data.transmittanceHLen = vec4(historyData.transmittance, historyData.hLen);
    return data;
}

CloudSSHistoryData vec4PackedData_toHistoryData(Vec4PackedData packedData) {
    CloudSSHistoryData historyData = clouds_ss_historyData_init();
    historyData.inScattering = packedData.inScatteringViewZ.rgb;
    historyData.viewZ = packedData.inScatteringViewZ.a;
    historyData.transmittance = packedData.transmittanceHLen.rgb;
    historyData.hLen = packedData.transmittanceHLen.a;
    return historyData;
}

Vec4PackedData vec4PackedData_mul(Vec4PackedData a, float b) {
    Vec4PackedData result;
    result.inScatteringViewZ = a.inScatteringViewZ * b;
    result.transmittanceHLen = a.transmittanceHLen * b;
    return result;
}

Vec4PackedData vec4PackedData_add(Vec4PackedData a, Vec4PackedData b) {
    Vec4PackedData result;
    result.inScatteringViewZ = a.inScatteringViewZ + b.inScatteringViewZ;
    result.transmittanceHLen = a.transmittanceHLen + b.transmittanceHLen;
    return result;
}

Vec4PackedData vec4PackData_clamp(Vec4PackedData data, Vec4PackedData minVal, Vec4PackedData maxVal) {
    Vec4PackedData result;
    result.inScatteringViewZ = clamp(data.inScatteringViewZ, minVal.inScatteringViewZ, maxVal.inScatteringViewZ);
    result.transmittanceHLen.xyz = clamp(data.transmittanceHLen.xyz, minVal.transmittanceHLen.xyz, maxVal.transmittanceHLen.xyz);
    return result;
}

Vec4PackedData loadCurrData(ivec2 texelPosD) {
    texelPosD = clamp(texelPosD, ivec2(0), renderSize - 1);
    CloudSSHistoryData historyData = clouds_ss_historyData_init();
    clouds_ss_historyData_unpack(texelFetch(usam_csrgba32ui, gi_diffuseHistory_texelToTexel(texelPosD), 0), historyData);
    return vec4PackedData_fromHistoryData(historyData);
}

Vec4PackedData loadPrevData(ivec2 texelPos) {
    CloudSSHistoryData historyData = clouds_ss_historyData_init();
    clouds_ss_historyData_unpack(texelFetch(usam_csrgba32ui, clouds_ss_history_texelToTexel(texelPos), 0), historyData);
    return vec4PackedData_fromHistoryData(historyData);
}

void loadAndAccumData(Vec4PackedData sampleData, float sampleWeight, inout Vec4PackedData prevSumData, inout float weightSum) {
    prevSumData = vec4PackedData_add(prevSumData, vec4PackedData_mul(sampleData, sampleWeight));
    weightSum += sampleWeight;
}

void loadAndAccumCurr(ivec2 prevSampleTexelPos, float sampleWeight, inout Vec4PackedData prevSumData, inout float weightSum) {
    loadAndAccumData(loadCurrData(prevSampleTexelPos), sampleWeight, prevSumData, weightSum);
}

void loadAndAccumPrev(ivec2 prevSampleTexelPos, float sampleWeight, inout Vec4PackedData prevSumData, inout float weightSum) {
    loadAndAccumData(loadPrevData(prevSampleTexelPos), sampleWeight, prevSumData, weightSum);
}

const float WEIGHT_EPSILON = 0.0001;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 texelCenter = vec2(texelPos) + 0.5;
        vec2 uv = texelCenter * global_mainImageSizeRcp;
        ivec2 texelPosDownScale = DOWNSCALE_DIVIDE(texelPos);

        Vec4PackedData currSumData = vec4PackedData_init();
        {
            float currWeightSum = 0.0;
            vec2 centerTexel = texelCenter / UPSCALE_FACTOR;
            centerTexel -= rand_r2Seq2(frameCounter) - 0.5;
            vec2 centerPixel = centerTexel - 0.5;
            vec2 centerPixelOrigin = floor(centerPixel);
            vec2 pixelPosFract = centerPixel - centerPixelOrigin;

//            vec4 weightX = sampling_catmullRomWeights(pixelPosFract.x);
//            vec4 weightY = sampling_catmullRomWeights(pixelPosFract.y);

            vec4 weightX = sampling_lanczoc2Weights(pixelPosFract.x);
            vec4 weightY = sampling_lanczoc2Weights(pixelPosFract.y);

            ivec2 gatherTexelPos = ivec2(centerPixelOrigin) + ivec2(1);

            Vec4PackedData prevSumData = vec4PackedData_init();
            float prevWeightSum = 0.0;
            vec4 weights4 = weightX.xyyx * weightY.wwzz;
            loadAndAccumCurr(gatherTexelPos + ivec2(-2, 1), weights4.x, currSumData, currWeightSum);
            loadAndAccumCurr(gatherTexelPos + ivec2(-1, 1), weights4.y, currSumData, currWeightSum);
            loadAndAccumCurr(gatherTexelPos + ivec2(-1, 0), weights4.z, currSumData, currWeightSum);
            loadAndAccumCurr(gatherTexelPos + ivec2(-2, 0), weights4.w, currSumData, currWeightSum);

            weights4 = weightX.zwwz * weightY.wwzz;
            loadAndAccumCurr(gatherTexelPos + ivec2(0, 1), weights4.x, currSumData, currWeightSum);
            loadAndAccumCurr(gatherTexelPos + ivec2(1, 1), weights4.y, currSumData, currWeightSum);
            loadAndAccumCurr(gatherTexelPos + ivec2(1, 0), weights4.z, currSumData, currWeightSum);
            loadAndAccumCurr(gatherTexelPos + ivec2(0, 0), weights4.w, currSumData, currWeightSum);

            weights4 = weightX.zwwz * weightY.yyxx;
            loadAndAccumCurr(gatherTexelPos + ivec2(0, -1), weights4.x, currSumData, currWeightSum);
            loadAndAccumCurr(gatherTexelPos + ivec2(1, -1), weights4.y, currSumData, currWeightSum);
            loadAndAccumCurr(gatherTexelPos + ivec2(1, -2), weights4.z, currSumData, currWeightSum);
            loadAndAccumCurr(gatherTexelPos + ivec2(0, -2), weights4.w, currSumData, currWeightSum);

            weights4 = weightX.xyyx * weightY.yyxx;
            loadAndAccumCurr(gatherTexelPos + ivec2(-2, -1), weights4.x, currSumData, currWeightSum);
            loadAndAccumCurr(gatherTexelPos + ivec2(-1, -1), weights4.y, currSumData, currWeightSum);
            loadAndAccumCurr(gatherTexelPos + ivec2(-1, -2), weights4.z, currSumData, currWeightSum);
            loadAndAccumCurr(gatherTexelPos + ivec2(-2, -2), weights4.w, currSumData, currWeightSum);

            currSumData = vec4PackedData_mul(currSumData, rcp(currWeightSum));
        }

        float averageViewZ = currSumData.inScatteringViewZ.w;
        averageViewZ = 65.536;
        averageViewZ *= -1000.0; // Convert to meters
        vec3 currView = coords_toViewCoord(uv, averageViewZ, global_camProjInverse);
        vec4 currScene = gbufferModelViewInverse * vec4(currView, 1.0);
        vec4 curr2PrevScene = coord_sceneCurrToPrev(currScene);
        vec4 curr2PrevView = gbufferPrevModelView * curr2PrevScene;
        vec4 curr2PrevClip = global_prevCamProj * curr2PrevView;
        uint clipFlag = uint(curr2PrevClip.z > 0.0);
        clipFlag &= uint(all(lessThan(abs(curr2PrevClip.xy), curr2PrevClip.ww)));
//        clipFlag = 0u;

        Vec4PackedData prevAvgData = vec4PackedData_init();
        if (bool(clipFlag)) {
            vec2 curr2PrevNDC = curr2PrevClip.xy / curr2PrevClip.w;
            vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;
            vec2 curr2PrevTexel = curr2PrevScreen * global_mainImageSize;

            vec2 centerPixel = curr2PrevTexel - 0.5;
            vec2 centerPixelOrigin = floor(centerPixel);
            vec2 pixelPosFract = centerPixel - centerPixelOrigin;

//            vec4 weightX = sampling_catmullRomWeights(pixelPosFract.x);
//            vec4 weightY = sampling_catmullRomWeights(pixelPosFract.y);

            vec4 weightX = sampling_lanczoc2Weights(pixelPosFract.x);
            vec4 weightY = sampling_lanczoc2Weights(pixelPosFract.y);

            ivec2 gatherTexelPos = ivec2(centerPixelOrigin) + ivec2(1);

            Vec4PackedData prevSumData = vec4PackedData_init();
            float prevWeightSum = 0.0;
            vec4 weights4 = weightX.xyyx * weightY.wwzz;
            loadAndAccumPrev(gatherTexelPos + ivec2(-2, 1), weights4.x, prevSumData, prevWeightSum);
            loadAndAccumPrev(gatherTexelPos + ivec2(-1, 1), weights4.y, prevSumData, prevWeightSum);
            loadAndAccumPrev(gatherTexelPos + ivec2(-1, 0), weights4.z, prevSumData, prevWeightSum);
            loadAndAccumPrev(gatherTexelPos + ivec2(-2, 0), weights4.w, prevSumData, prevWeightSum);

            weights4 = weightX.zwwz * weightY.wwzz;
            loadAndAccumPrev(gatherTexelPos + ivec2(0, 1), weights4.x, prevSumData, prevWeightSum);
            loadAndAccumPrev(gatherTexelPos + ivec2(1, 1), weights4.y, prevSumData, prevWeightSum);
            loadAndAccumPrev(gatherTexelPos + ivec2(1, 0), weights4.z, prevSumData, prevWeightSum);
            loadAndAccumPrev(gatherTexelPos + ivec2(0, 0), weights4.w, prevSumData, prevWeightSum);

            weights4 = weightX.zwwz * weightY.yyxx;
            loadAndAccumPrev(gatherTexelPos + ivec2(0, -1), weights4.x, prevSumData, prevWeightSum);
            loadAndAccumPrev(gatherTexelPos + ivec2(1, -1), weights4.y, prevSumData, prevWeightSum);
            loadAndAccumPrev(gatherTexelPos + ivec2(1, -2), weights4.z, prevSumData, prevWeightSum);
            loadAndAccumPrev(gatherTexelPos + ivec2(0, -2), weights4.w, prevSumData, prevWeightSum);

            weights4 = weightX.xyyx * weightY.yyxx;
            loadAndAccumPrev(gatherTexelPos + ivec2(-2, -1), weights4.x, prevSumData, prevWeightSum);
            loadAndAccumPrev(gatherTexelPos + ivec2(-1, -1), weights4.y, prevSumData, prevWeightSum);
            loadAndAccumPrev(gatherTexelPos + ivec2(-1, -2), weights4.z, prevSumData, prevWeightSum);
            loadAndAccumPrev(gatherTexelPos + ivec2(-2, -2), weights4.w, prevSumData, prevWeightSum);

            if (prevWeightSum > WEIGHT_EPSILON) {
                prevAvgData = vec4PackedData_mul(prevSumData, rcp(prevWeightSum));
                vec3 prevView = coords_toViewCoord(curr2PrevScreen, prevAvgData.inScatteringViewZ.w * -1000.0, global_prevCamProjInverse);
                vec4 prevScene = gbufferPrevModelViewInverse * vec4(prevView, 1.0);
                vec4 prev2CurrScene = coord_scenePrevToCurr(prevScene);
                vec4 prev2CurrView = gbufferModelView * prev2CurrScene;
                prevAvgData.inScatteringViewZ.w = prev2CurrView.z / -1000.0;
            }
        }

        float currWeight = currSumData.transmittanceHLen.w;

        CloudSSHistoryData newData = vec4PackedData_toHistoryData(prevAvgData);
        newData.hLen = min(prevAvgData.transmittanceHLen.w + currWeight, CLOUDS_SS_MAX_ACCUM);
        if (newData.hLen > WEIGHT_EPSILON) {
            Vec4PackedData currAvgData = prevAvgData;
            currAvgData = vec4PackedData_mul(currSumData, rcp(currWeight));

            float alpha = saturate(currWeight / newData.hLen);
            newData.inScattering = mix(newData.inScattering, currAvgData.inScatteringViewZ.xyz, alpha);
            newData.transmittance = mix(newData.transmittance, currAvgData.transmittanceHLen.xyz, alpha);
            newData.viewZ = mix(newData.viewZ, currAvgData.inScatteringViewZ.w, alpha);
            newData.inScattering = max(newData.inScattering, vec3(0.0));
            newData.transmittance = saturate(newData.transmittance);
            newData.viewZ = max(newData.viewZ, 0.0);
        }

        uvec4 packedOutput = uvec4(0u);
        clouds_ss_historyData_pack(packedOutput, newData);
        imageStore(uimg_tempRGBA32UI, texelPos, packedOutput);
    }
}