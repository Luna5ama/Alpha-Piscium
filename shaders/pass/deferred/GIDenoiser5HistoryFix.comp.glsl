#include "/techniques/gi/Common.glsl"
#include "/util/Colors.glsl"
#include "/util/GBufferData.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgba8) uniform writeonly image2D uimg_rgba8;

vec3 _clampColor(vec3 colorRGB, vec3 fastColorRGB, vec3 moment1YCoCG, vec3 moment2YCoCG) {
    vec3 mean = moment1YCoCG;
    vec3 variance = max(moment2YCoCG - moment1YCoCG * moment1YCoCG, 0.0);
    vec3 stddev = sqrt(variance);
    const float clampingThreshold = 1.0;
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
        historyData.diffuseColor = _clampColor(historyData.diffuseColor, historyData.diffuseFastColor, diffMoment1, diffMoment2);
        historyData.specularColor = _clampColor(historyData.specularColor, historyData.specularFastColor, specMoment1, specMoment2);
        #endif

        transient_gi1Reprojected_store(texelPos, gi_historyData_pack1(historyData));
        transient_gi2Reprojected_store(texelPos, gi_historyData_pack2(historyData));
        transient_gi3Reprojected_store(texelPos, gi_historyData_pack3(historyData));
        transient_gi4Reprojected_store(texelPos, gi_historyData_pack4(historyData));
        transient_gi5Reprojected_store(texelPos, gi_historyData_pack5(historyData));

        vec4 diffInput = vec4(historyData.diffuseColor, 0.0);
        vec4 specInput = vec4(historyData.specularColor, 0.0);

        transient_gi_blurDiff1_store(texelPos, diffInput);
        transient_gi_blurSpec1_store(texelPos, specInput);
    }
}
