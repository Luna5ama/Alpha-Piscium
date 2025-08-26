#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#define HIZ_SUBGROUP_CHECK a

#include "Common.glsl"
#include "/util/Coords.glsl"
#include "/util/Sampling.glsl"
#include "/techniques/HiZ.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32ui) uniform restrict writeonly uimage2D uimg_csrgba32ui;

struct Vec4PackedData {
    vec4 transmittanceHLen;
    vec3 inScattering;
};

Vec4PackedData vec4PackedData_init() {
    Vec4PackedData data;
    data.inScattering = vec3(0.0);
    data.transmittanceHLen = vec4(0.0);
    return data;
}

Vec4PackedData vec4PackedData_fromHistoryData(CloudSSHistoryData historyData) {
    Vec4PackedData data;
    data.inScattering = historyData.inScattering;
    data.transmittanceHLen = vec4(historyData.transmittance, historyData.hLen);
    return data;
}

CloudSSHistoryData vec4PackedData_toHistoryData(Vec4PackedData packedData) {
    CloudSSHistoryData historyData = clouds_ss_historyData_init();
    historyData.inScattering = packedData.inScattering;
    historyData.transmittance = packedData.transmittanceHLen.rgb;
    historyData.hLen = packedData.transmittanceHLen.a;
    return historyData;
}

Vec4PackedData vec4PackedData_mul(Vec4PackedData a, float b) {
    Vec4PackedData result;
    result.inScattering = a.inScattering * b;
    result.transmittanceHLen = a.transmittanceHLen * b;
    return result;
}

Vec4PackedData vec4PackedData_add(Vec4PackedData a, Vec4PackedData b) {
    Vec4PackedData result;
    result.inScattering = a.inScattering + b.inScattering;
    result.transmittanceHLen = a.transmittanceHLen + b.transmittanceHLen;
    return result;
}

Vec4PackedData vec4PackData_clamp(Vec4PackedData data, Vec4PackedData minVal, Vec4PackedData maxVal) {
    Vec4PackedData result;
    result.inScattering = clamp(data.inScattering, minVal.inScattering, maxVal.inScattering);
    result.transmittanceHLen.xyz = clamp(data.transmittanceHLen.xyz, minVal.transmittanceHLen.xyz, maxVal.transmittanceHLen.xyz);
    return result;
}

Vec4PackedData loadCurrData(ivec2 texelPosD) {
    texelPosD = clamp(texelPosD, ivec2(0), renderSize - 1);
    CloudSSHistoryData historyData = clouds_ss_historyData_init();
    clouds_ss_historyData_unpack(texelFetch(usam_csrgba32ui, csrgba32ui_temp1_texelToTexel(texelPosD), 0), historyData);
    return vec4PackedData_fromHistoryData(historyData);
}

Vec4PackedData loadPrevData(ivec2 texelPos) {
    CloudSSHistoryData historyData = clouds_ss_historyData_init();
    clouds_ss_historyData_unpack(texelFetch(usam_csrgba32ui, clouds_ss_history_texelToTexel(texelPos), 0), historyData);
    return vec4PackedData_fromHistoryData(historyData);
}

const float WEIGHT_EPSILON = 0.0001;

void main() {
    if (hiz_groupSkyCheckSubgroup(gl_WorkGroupID.xy, 3)) {
        ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
        if (all(lessThan(texelPos, global_mainImageSizeI))) {
            vec2 texelCenter = vec2(texelPos) + 0.5;
            vec2 uv = texelCenter * global_mainImageSizeRcp;
            ivec2 texelPosDownScale = DOWNSCALE_DIVIDE(texelPos);

            vec3 currView = coords_toViewCoord(uv, -65536.0, global_camProjInverse);
            vec4 currScene = gbufferModelViewInverse * vec4(currView, 1.0);
            vec4 curr2PrevScene = coord_sceneCurrToPrev(currScene);
            vec4 curr2PrevView = gbufferPrevModelView * curr2PrevScene;
            vec4 curr2PrevClip = global_prevCamProj * curr2PrevView;
            uint clipFlag = uint(curr2PrevClip.z > 0.0);
            clipFlag &= uint(all(lessThan(abs(curr2PrevClip.xy), curr2PrevClip.ww)));

            Vec4PackedData prevAvgData = vec4PackedData_init();
            if (bool(clipFlag)) {
                vec2 curr2PrevNDC = curr2PrevClip.xy / curr2PrevClip.w;
                vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;
                vec2 curr2PrevTexel = curr2PrevScreen * global_mainImageSize;

                vec2 centerPixel = curr2PrevTexel - 0.5;
                vec2 centerPixelOrigin = floor(centerPixel);
                vec2 pixelPosFract = centerPixel - centerPixelOrigin;

                float B = 0.0;
                float C = 0.5;
                vec4 weightX = sampling_mitchellNetravaliWeights(pixelPosFract.x, B, C);
                vec4 weightY = sampling_mitchellNetravaliWeights(pixelPosFract.y, B, C);

                ivec2 gatherTexelPos = ivec2(centerPixelOrigin) + ivec2(1);
                for (int iy = 0; iy < 4; ++iy) {
                    for (int ix = 0; ix < 4; ++ix) {
                        ivec2 offset = ivec2(ix, iy) - 2;
                        Vec4PackedData sampleData = loadPrevData(gatherTexelPos + offset);
                        float weight = weightX[ix] * weightY[iy];
                        prevAvgData = vec4PackedData_add(prevAvgData, vec4PackedData_mul(sampleData, weight));
                    }
                }
            }

            float prevWeight = prevAvgData.transmittanceHLen.w;
            prevWeight *= global_historyResetFactor;

            Vec4PackedData currAvgData = vec4PackedData_init();
            vec3 inSctrMoment1 = vec3(0.0);
            vec3 inSctrMoment2 = vec3(0.0);
            vec3 transmittanceMoment1 = vec3(0.0);
            vec3 transmittanceMoment2 = vec3(0.0);
            {
                vec2 centerTexel = texelCenter / UPSCALE_FACTOR;
                centerTexel -= clouds_ss_upscaleoffset() - 0.5;
                vec2 centerPixel = centerTexel - 0.5;
                vec2 centerPixelOrigin = floor(centerPixel);
                vec2 pixelPosFract = centerPixel - centerPixelOrigin;

                float B = 0.0;
                float C = 1.0;
                vec4 weightX = sampling_mitchellNetravaliWeights(pixelPosFract.x, B, C);
                vec4 weightY = sampling_mitchellNetravaliWeights(pixelPosFract.y, B, C);
                vec4 momentWeightX = sampling_gaussianWeights(pixelPosFract.x, 1.0);
                vec4 momentWeightY = sampling_gaussianWeights(pixelPosFract.y, 1.0);
                float maxWeight = 0.0;
                float totalMomentWeight = 0.0;

                ivec2 gatherTexelPos = ivec2(centerPixelOrigin) + ivec2(1);
                for (int iy = 0; iy < 4; ++iy) {
                    for (int ix = 0; ix < 4; ++ix) {
                        ivec2 offset = ivec2(ix, iy) - 2;
                        Vec4PackedData sampleData = loadCurrData(gatherTexelPos + offset);
                        float weight = weightX[ix] * weightY[iy];
                        maxWeight = max(maxWeight, weight);
                        currAvgData = vec4PackedData_add(currAvgData, vec4PackedData_mul(sampleData, weight));

                        vec3 inSctrYCoCg = colors_SRGBToYCoCg(sampleData.inScattering);
                        vec3 transmittanceYCoCg = colors_SRGBToYCoCg(sampleData.transmittanceHLen.rgb);
                        float momentWeight = momentWeightX[ix] * momentWeightY[iy];
                        inSctrMoment1 += inSctrYCoCg * momentWeight;
                        inSctrMoment2 += inSctrYCoCg * inSctrYCoCg * momentWeight;
                        transmittanceMoment1 += transmittanceYCoCg * momentWeight;
                        transmittanceMoment2 += transmittanceYCoCg * transmittanceYCoCg * momentWeight;
                        totalMomentWeight += momentWeight;
                    }
                }
                inSctrMoment1 /= totalMomentWeight;
                inSctrMoment2 /= totalMomentWeight;
                transmittanceMoment1 /= totalMomentWeight;
                transmittanceMoment2 /= totalMomentWeight;
                vec2 v = pow2(2.0 * pixelPosFract - 1.0);
                currAvgData.transmittanceHLen.w = saturate(pow(dot(v, v), prevWeight * pow2(SETTING_CLOUDS_LOW_SHARPENING)));
            }

            {

                vec3 prevInSctrYCoCg = colors_SRGBToYCoCg(prevAvgData.inScattering);
                vec3 prevTransmittanceYCoCg = colors_SRGBToYCoCg(prevAvgData.transmittanceHLen.rgb);

                // Ellipsoid intersection clipping by Marty
                const float clippingEps = FLT_MIN;
                vec3 inSctrStddev = sqrt(max(inSctrMoment2 - inSctrMoment1 * inSctrMoment1, clippingEps));
                vec3 inSctrDelta = prevInSctrYCoCg - inSctrMoment1;
                inSctrDelta /= max(1.0, length(inSctrDelta / inSctrStddev * SETTING_CLOUDS_LOW_VARIANCE_CLIPPING / global_historyResetFactor));
                prevInSctrYCoCg = inSctrMoment1 + inSctrDelta;

                vec3 transmittanceStddev = sqrt(max(transmittanceMoment2 - transmittanceMoment1 * transmittanceMoment1, clippingEps));
                vec3 transmittanceDelta = prevTransmittanceYCoCg - transmittanceMoment1;
                transmittanceDelta /= max(1.0, length(transmittanceDelta / transmittanceStddev / global_historyResetFactor));
                prevTransmittanceYCoCg = transmittanceMoment1 + transmittanceDelta;

                prevAvgData.inScattering = colors_YCoCgToSRGB(prevInSctrYCoCg);
                prevAvgData.transmittanceHLen.rgb = colors_YCoCgToSRGB(prevTransmittanceYCoCg);
            }

            float currWeight = currAvgData.transmittanceHLen.w;
            float newWeight = min(currWeight + prevWeight, CLOUDS_SS_MAX_ACCUM);

            Vec4PackedData newData = prevAvgData;
            newData.transmittanceHLen.w = newWeight;

            float alpha = saturate(currWeight / newWeight);
            newData.inScattering = mix(newData.inScattering, currAvgData.inScattering, alpha);
            newData.transmittanceHLen.xyz = mix(newData.transmittanceHLen.xyz, currAvgData.transmittanceHLen.xyz, alpha);

            CloudSSHistoryData newHistoryData = vec4PackedData_toHistoryData(newData);
            newHistoryData.inScattering = max(newHistoryData.inScattering, vec3(0.0));
            newHistoryData.transmittance = saturate(newHistoryData.transmittance);

            uvec4 packedOutput = uvec4(0u);
            clouds_ss_historyData_pack(packedOutput, newHistoryData);
            imageStore(uimg_csrgba32ui, csrgba32ui_temp2_texelToTexel(texelPos), packedOutput);
        }
    }
}