#include "/techniques/gi/Common.glsl"
#include "/util/Colors.glsl"
#include "/util/GBufferData.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgba8) uniform writeonly image2D uimg_rgba8;

vec3 _clampColor(vec3 colorRGB, vec3 fastColorRGB, vec3 moment1YCoCG, vec3 moment2YCoCG, float clampingThreshold) {
    vec3 mean = moment1YCoCG;
    vec3 variance = max(moment2YCoCG - moment1YCoCG * moment1YCoCG, 0.0);
    vec3 stddev = sqrt(variance);
    vec3 aabbMin = mean - stddev * clampingThreshold;
    vec3 aabbMax = mean + stddev * clampingThreshold;
    vec3 fastColorYCoCG = colors_SRGBToYCoCg(fastColorRGB);
    vec3 colorYCoCG = colors_SRGBToYCoCg(colorRGB);
    aabbMin = min(aabbMin, fastColorYCoCG);
    aabbMax = max(aabbMax, fastColorYCoCG);
    colorYCoCG = clamp(colorYCoCG, aabbMin, aabbMax);
    return colors_YCoCgToSRGB(colorYCoCG);
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 newDiffuse = transient_ssgiOut_fetch(texelPos);

        GIHistoryData historyData = gi_historyData_init();

        gi_historyData_unpack1(historyData, transient_gi1Reprojected_fetch(texelPos));
        gi_historyData_unpack2(historyData, transient_gi2Reprojected_fetch(texelPos));
        gi_historyData_unpack3(historyData, transient_gi3Reprojected_fetch(texelPos));
        gi_historyData_unpack4(historyData, transient_gi4Reprojected_fetch(texelPos));
        gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));

        // TODO: shared memory variance
        vec3 diffMoment1 = vec3(0.0);
        vec3 diffMoment2 = vec3(0.0);
        vec3 specMoment1 = vec3(0.0);
        vec3 specMoment2 = vec3(0.0);

        for (int dx = -2; dx <= 2; ++dx) {
            for (int dy = -2; dy <= 2; ++dy) {
                ivec2 neighborPos = texelPos + ivec2(dx, dy);
                vec3 neighborDiff = transient_gi2Reprojected_fetch(neighborPos).xyz;
                vec3 neighborSpec = transient_gi4Reprojected_fetch(neighborPos).xyz;
                vec3 neighborDiffYCoCg = colors_SRGBToYCoCg(neighborDiff);
                vec3 neighborSpecYCoCg = colors_SRGBToYCoCg(neighborSpec);
                diffMoment1 += neighborDiffYCoCg;
                diffMoment2 += neighborDiffYCoCg * neighborDiffYCoCg;
                specMoment1 += neighborSpecYCoCg;
                specMoment2 += neighborSpecYCoCg * neighborSpecYCoCg;
            }
        }


        diffMoment1 /= 25.0;
        diffMoment2 /= 25.0;
        specMoment1 /= 25.0;
        specMoment2 /= 25.0;

        #if ENABLE_DENOISER
        float len = historyData.realHistoryLength * REAL_HISTORY_LENGTH;
        float decayFactor = linearStep(FAST_HISTORY_LENGTH * 2.0, 1.0, historyData.realHistoryLength * REAL_HISTORY_LENGTH);
        float clampingThreshold = mix(2.0, 16.0, pow2(decayFactor));

        vec3 diffClamped = _clampColor(historyData.diffuseColor, historyData.diffuseFastColor, diffMoment1, diffMoment2, clampingThreshold);
        vec3 diffDiff = abs(diffClamped - historyData.diffuseColor);
        float diffDiffLuma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, diffDiff);
        float diffMeanLuma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, colors_YCoCgToSRGB(diffMoment1));

        vec3 specClamped = _clampColor(historyData.specularColor, historyData.specularFastColor, specMoment1, specMoment2, clampingThreshold);
        vec3 specDiff = abs(specClamped - historyData.specularColor);
        float specDiffLuma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, specDiff);
        float specMeanLuma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, colors_YCoCgToSRGB(specMoment1));

        float resetFactor = exp2(-(diffDiffLuma + specDiffLuma) * 64.0);
        historyData.historyLength *= resetFactor;
        historyData.realHistoryLength *= sqrt(resetFactor);
        imageStore(uimg_temp3, texelPos, vec4(resetFactor));

        historyData.diffuseColor = diffClamped;
        historyData.specularColor = specClamped;
        #endif

        transient_gi1Reprojected_store(texelPos, gi_historyData_pack1(historyData));
        transient_gi2Reprojected_store(texelPos, gi_historyData_pack2(historyData));
        transient_gi3Reprojected_store(texelPos, gi_historyData_pack3(historyData));
        transient_gi4Reprojected_store(texelPos, gi_historyData_pack4(historyData));
        transient_gi5Reprojected_store(texelPos, gi_historyData_pack5(historyData));

        vec4 diffInput = vec4(historyData.diffuseColor, 0.0);
        vec4 specInput = vec4(historyData.specularColor, 0.0);
//        imageStore(uimg_temp3, texelPos, diffInput);

        transient_gi_blurDiff1_store(texelPos, diffInput);
        transient_gi_blurSpec1_store(texelPos, specInput);
    }
}
