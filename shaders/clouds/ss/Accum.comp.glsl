#extension GL_KHR_shader_subgroup_basic : enable

#include "Common.glsl"
#include "/util/Coords.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform usampler2D usam_tempRGBA32UI;
uniform usampler2D usam_csrgba32ui;

layout(rgba32ui) uniform writeonly uimage2D uimg_csrgba32ui;

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
    CloudSSHistoryData historyData = clouds_ss_historyData_init();
    clouds_ss_historyData_unpack(texelFetch(usam_tempRGBA32UI, texelPosD, 0), historyData);
    return vec4PackedData_fromHistoryData(historyData);
}

Vec4PackedData loadPrevData(ivec2 texelPos) {
    CloudSSHistoryData historyData = clouds_ss_historyData_init();
    clouds_ss_historyData_unpack(texelFetch(usam_csrgba32ui, clouds_ss_history_texelToTexel(texelPos), 0), historyData);
    return vec4PackedData_fromHistoryData(historyData);
}

float computeSampleWeight(vec2 centerTexelCenter, ivec2 sampleTexelPosD) {
    const float DISTANCE_FACTOR = 0.05 * float(UPSCALE_FACTOR);
    vec2 sampleTexelPos1x1 = getTexelPos1x1(sampleTexelPosD);
    vec2 texelPosDiff = sampleTexelPos1x1 - centerTexelCenter;
    float sampleDist = dot(texelPosDiff, texelPosDiff);
    return DISTANCE_FACTOR / (DISTANCE_FACTOR + sampleDist);
}

void accumSample(Vec4PackedData sampleData, float sampleWeight, inout Vec4PackedData sumData) {
    sumData = vec4PackedData_add(sumData, vec4PackedData_mul(sampleData, sampleWeight));
}

void loadAndAccumCurr(vec2 centerTexelCenter, ivec2 centerTexelPosD, ivec2 offset, inout Vec4PackedData sumData) {
    ivec2 neighborTexelPosD = centerTexelPosD + offset;
    if (all(lessThan(neighborTexelPosD, renderSize))) {
        Vec4PackedData sampleData = loadCurrData(neighborTexelPosD);
        float sampleWeight = computeSampleWeight(centerTexelCenter, neighborTexelPosD);
        accumSample(sampleData, sampleWeight, sumData);
    }
}

void loadAndAccumPrev(ivec2 prevSampleTexelPos, float sampleWeight, inout Vec4PackedData prevSumData, inout float weightSum) {
    Vec4PackedData sampleData = loadPrevData(prevSampleTexelPos);
    prevSumData = vec4PackedData_add(prevSumData, vec4PackedData_mul(sampleData, sampleWeight));
    weightSum += sampleWeight;
}

const float WEIGHT_EPSILON = 0.0001;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 texelCenter = vec2(texelPos) + 0.5;
        vec2 uv = texelCenter * global_mainImageSizeRcp;
        ivec2 texelPosDownScale = texelPos / UPSCALE_FACTOR;

        Vec4PackedData centerData = loadCurrData(texelPosDownScale);
        Vec4PackedData currSumData = vec4PackedData_mul(centerData, computeSampleWeight(texelCenter, texelPosDownScale));

        loadAndAccumCurr(texelCenter, texelPosDownScale, ivec2(-1, -1), currSumData);
        loadAndAccumCurr(texelCenter, texelPosDownScale, ivec2(0, -1), currSumData);
        loadAndAccumCurr(texelCenter, texelPosDownScale, ivec2(1, -1), currSumData);
        loadAndAccumCurr(texelCenter, texelPosDownScale, ivec2(-1, 0), currSumData);
        loadAndAccumCurr(texelCenter, texelPosDownScale, ivec2(1, 0), currSumData);
        loadAndAccumCurr(texelCenter, texelPosDownScale, ivec2(-1, 1), currSumData);
        loadAndAccumCurr(texelCenter, texelPosDownScale, ivec2(0, 1), currSumData);
        loadAndAccumCurr(texelCenter, texelPosDownScale, ivec2(1, 1), currSumData);

        Vec4PackedData prevSumData = vec4PackedData_init();
        float prevWeightSum = 0.0;

        float averageViewZ = currSumData.inScatteringViewZ.w / max(currSumData.transmittanceHLen.w, 1e-6);
        averageViewZ *= -1000.0; // Convert to meters
        vec3 currView = coords_toViewCoord(uv, averageViewZ, global_camProjInverse);
        vec4 currScene = gbufferModelViewInverse * vec4(currView, 1.0);
        vec4 curr2PrevView = coord_viewCurrToPrev(vec4(currView, 1.0), false);
        vec4 curr2PrevClip = global_prevCamProj * curr2PrevView;
        uint clipFlag = uint(curr2PrevClip.z > 0.0);
        clipFlag &= uint(all(lessThan(abs(curr2PrevClip.xy), curr2PrevClip.ww)));
        if (bool(clipFlag)) {
            vec2 curr2PrevNDC = curr2PrevClip.xy / curr2PrevClip.w;
            vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;
            vec2 curr2PrevTexel = curr2PrevScreen * global_mainImageSize;

            vec2 centerPixel = curr2PrevTexel - 0.5;
            vec2 centerPixelOrigin = floor(centerPixel);
            vec2 gatherTexelPos = centerPixelOrigin + 1.0;
            vec2 pixelPosFract = centerPixel - centerPixelOrigin;

            vec2 bilinearWeights2 = pixelPosFract;
            vec4 blinearWeights4;
            blinearWeights4.yz = bilinearWeights2.xx;
            blinearWeights4.xw = 1.0 - bilinearWeights2.xx;
            blinearWeights4.xy *= bilinearWeights2.yy;
            blinearWeights4.zw *= 1.0 - bilinearWeights2.yy;

            ivec2 centerPixelOriginI = ivec2(centerPixelOrigin);

            loadAndAccumPrev(centerPixelOriginI, blinearWeights4.w, prevSumData, prevWeightSum);
            loadAndAccumPrev(centerPixelOriginI + ivec2(1, 0), blinearWeights4.z, prevSumData, prevWeightSum);
            loadAndAccumPrev(centerPixelOriginI + ivec2(0, 1), blinearWeights4.x, prevSumData, prevWeightSum);
            loadAndAccumPrev(centerPixelOriginI + ivec2(1, 1), blinearWeights4.y, prevSumData, prevWeightSum);
        }

        float currWeight = currSumData.transmittanceHLen.w;

        Vec4PackedData prevAvgData = vec4PackedData_init();
        if (prevWeightSum > WEIGHT_EPSILON) {
            prevAvgData = vec4PackedData_mul(prevSumData, rcp(prevWeightSum));
        }

        CloudSSHistoryData newData = vec4PackedData_toHistoryData(prevAvgData);
        newData.hLen = min(prevAvgData.transmittanceHLen.w + currWeight, CLOUDS_SS_MAX_ACCUM);
        if (newData.hLen > WEIGHT_EPSILON) {
            Vec4PackedData currAvgData = prevAvgData;
            if (currWeight > linearStep(0.0, CLOUDS_SS_MAX_ACCUM, newData.hLen) * 0.5 + WEIGHT_EPSILON) {
                currAvgData = vec4PackedData_mul(currSumData, rcp(currWeight));
            }

            float alpha = saturate(currWeight / newData.hLen);
            newData.inScattering = mix(newData.inScattering, currAvgData.inScatteringViewZ.xyz, alpha);
            newData.transmittance = mix(newData.transmittance, currAvgData.transmittanceHLen.xyz, alpha);
            newData.viewZ = mix(newData.viewZ, currAvgData.inScatteringViewZ.w, alpha);
            newData.inScattering = max(newData.inScattering, vec3(0.0));
            newData.transmittance = max(newData.transmittance, vec3(0.0));
            newData.viewZ = max(newData.viewZ, 0.0);
        }

        uvec4 packedOutput = uvec4(0u);
        clouds_ss_historyData_pack(packedOutput, newData);
        imageStore(uimg_csrgba32ui, gi_diffuseHistory_texelToTexel(texelPos), packedOutput);
    }
}