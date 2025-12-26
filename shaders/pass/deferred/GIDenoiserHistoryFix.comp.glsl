#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/gi/Common.glsl"
#include "/util/Colors.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Rand.glsl"
#include "/techniques/HiZCheck.glsl"

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
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(gl_WorkGroupID.xy, 4, texelPos);
        if (viewZ > -65536.0) {
            GIHistoryData historyData = gi_historyData_init();

            gi_historyData_unpack1(historyData, transient_gi1Reprojected_fetch(texelPos));
            gi_historyData_unpack2(historyData, transient_gi2Reprojected_fetch(texelPos));
            gi_historyData_unpack3(historyData, transient_gi3Reprojected_fetch(texelPos));
            gi_historyData_unpack4(historyData, transient_gi4Reprojected_fetch(texelPos));
            gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));

            float historyLengthInt = historyData.historyLength * HISTORY_LENGTH;
            float historyFixMix = 1.0 - pow2(linearStep(4.0, 1.0, historyLengthInt));

            #if DENOISER_HISTORY_FIX
            if (historyFixMix < 1.0) {
                vec2 texelPos0 = vec2(texelPos) + 0.5;
                vec3 geomNormal0 = normalize(transient_viewNormal_fetch(texelPos).xyz * 2.0 - 1.0);
                float viweZ0 = texelFetch(usam_gbufferViewZ, texelPos, 0).x;
                vec4 diffSum = vec4(0.0);
                vec4 specSum = vec4(0.0);
                float weightSum = 0.0;
                const float baseReductionFactor = ldexp(1.0, -8);
                float baseDepthWeight = max(2.0, abs(viweZ0)) * ldexp(1.0, -4);
                for (int mip = 6; mip >= 1; mip--) {
                    ivec2 stbnPos = ivec2(texelPos0 + rand_r2Seq2(mip) * 128.0);
                    vec2 stbnRand = rand_stbnVec2(stbnPos, RANDOM_FRAME);
                    vec2 texelPosMip = ldexp(texelPos0, ivec2(-mip)) + stbnRand - 0.5;

                    ivec2 mipOffset = ivec2(global_mipmapSizePrefixes[mip - 1].x - uval_mainImageSizeI.x, 0);
                    ivec2 mipSize = global_mipmapSizesI[mip];

                    vec2 texelPosMipTile = clamp(texelPosMip, vec2(0.5), vec2(mipSize) - vec2(0.5));
                    texelPosMipTile += vec2(mipOffset);
                    vec2 screenPosMipTile = texelPosMipTile * uval_mainImageSizeRcp;
                    ivec4 mipTileMin = global_mipmapTiles[1][mip];
                    ivec4 mipTileMax = global_mipmapTiles[0][mip];
                    vec2 hiZMinReadPos = mipTileMin.xy + ivec2(texelPosMip);
                    vec2 hiZMaxReadPos = mipTileMax.xy + ivec2(texelPosMip);

                    float hiZMin = texelFetch(usam_hiz, ivec2(hiZMinReadPos), 0).r;
                    float hiZMax = texelFetch(usam_hiz, ivec2(hiZMaxReadPos), 0).r;

                    vec3 geomNormalMipRaw = transient_geomNormalMip_sample(screenPosMipTile).xyz;
                    vec4 mipDiff = transient_gi_diffMip_sample(screenPosMipTile);
                    vec4 mipSpec = transient_gi_specMip_sample(screenPosMipTile);
                    geomNormalMipRaw = geomNormalMipRaw * 2.0 - 1.0;
                    vec3 geomNormalMip = normalize(geomNormalMipRaw);

                    float geomNormalLengthSq = saturate(lengthSq(geomNormalMipRaw));
                    float geomNormalDot = dot(geomNormal0, geomNormalMipRaw * inversesqrt(geomNormalLengthSq));
                    float geomNormalWeight = geomNormalLengthSq * pow2(saturate(geomNormalDot));
                    geomNormalWeight = pow(geomNormalWeight, ldexp(1.0, mip));
                    float reductionFactor = baseReductionFactor / (baseReductionFactor + weightSum);
                    hiZMax = coords_reversedZToViewZ(hiZMax, nearPlane);
                    hiZMin = coords_reversedZToViewZ(hiZMin, nearPlane);
                    float maxZWeight = baseDepthWeight / (baseDepthWeight + abs(hiZMax - viweZ0));
                    float minZWeight = baseDepthWeight / (baseDepthWeight + abs(hiZMin - viweZ0));

                    float sampleWeight = 1.0;
                    sampleWeight *= geomNormalWeight;
                    sampleWeight *= maxZWeight;
                    sampleWeight *= minZWeight;
                    sampleWeight *= reductionFactor;

                    diffSum += mipDiff * sampleWeight;
                    specSum += mipSpec * sampleWeight;
                    weightSum += sampleWeight;
                }

                float baseMipWeight = (baseReductionFactor / (baseReductionFactor + weightSum)) * 0.00000001;
                diffSum += vec4(historyData.diffuseColor * baseMipWeight, 0.0);
                specSum += vec4(historyData.specularColor * baseMipWeight, 0.0);
                weightSum += baseMipWeight;

                float rcpWeightSum = 1.0 / weightSum;
                diffSum *= rcpWeightSum;
                specSum *= rcpWeightSum;

                historyData.diffuseColor = mix(diffSum.rgb, historyData.diffuseColor, historyFixMix);
                historyData.specularColor = mix(specSum.rgb, historyData.specularColor, historyFixMix);
            }
            #endif

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

            transient_geomNormalMip_fetch(texelPos);
            transient_gi_diffMip_fetch(texelPos);
            transient_gi_specMip_fetch(texelPos);

            diffMoment1 /= 25.0;
            diffMoment2 /= 25.0;
            specMoment1 /= 25.0;
            specMoment2 /= 25.0;

            #if ENABLE_DENOISER_FAST_CLAMP
            float len = historyData.realHistoryLength * REAL_HISTORY_LENGTH;
            float decayFactor = linearStep(FAST_HISTORY_LENGTH * 2.0, 1.0, historyData.realHistoryLength * REAL_HISTORY_LENGTH);
            float clampingThreshold = mix(2.0, 16.0, pow2(decayFactor));

            vec3 diffClamped = _clampColor(historyData.diffuseColor, historyData.diffuseFastColor, diffMoment1, diffMoment2, clampingThreshold);
            vec3 diffDiff = abs(diffClamped - historyData.diffuseColor);
            float diffDiffLuma = colors2_colorspaces_luma(SETTING_WORKING_COLOR_SPACE, diffDiff);
            float diffMeanLuma = colors2_colorspaces_luma(SETTING_WORKING_COLOR_SPACE, colors_YCoCgToSRGB(diffMoment1));

            vec3 specClamped = _clampColor(historyData.specularColor, historyData.specularFastColor, specMoment1, specMoment2, clampingThreshold);
            vec3 specDiff = abs(specClamped - historyData.specularColor);
            float specDiffLuma = colors2_colorspaces_luma(SETTING_WORKING_COLOR_SPACE, specDiff);
            float specMeanLuma = colors2_colorspaces_luma(SETTING_WORKING_COLOR_SPACE, colors_YCoCgToSRGB(specMoment1));

            float resetFactor = exp2(-(diffDiffLuma + specDiffLuma) * 1.0);
            historyData.historyLength *= resetFactor;
            historyData.realHistoryLength *= sqrt(resetFactor);
            //        imageStore(uimg_temp3, texelPos, vec4(resetFactor));

            historyData.diffuseColor = mix(historyData.diffuseColor, diffClamped, historyFixMix);
            historyData.specularColor = mix(historyData.specularColor, specClamped, historyFixMix);
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
}
