#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/gi/Common.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/util/Colors.glsl"
#include "/util/GBufferData.glsl"
#include "/util/AgxInvertible.glsl"
#include "/util/Rand.glsl"
#include "/util/Sampling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgba8) uniform writeonly image2D uimg_rgba8;

#if ENABLE_DENOISER_FAST_CLAMP
// Shared memory with padding for 5x5 tap (-2 to +2)
// Each work group is 16x16, need +2 padding on each side for 5x5 taps
shared uvec4 shared_YCoCgData[20][20];

uvec2 groupOriginTexelPos = gl_WorkGroupID.xy << 4u;

void loadSharedDataMoments(uint index) {
    if (index < 400u) { // 20 * 20 = 400
        uvec2 sharedXY = uvec2(index % 20u, index / 20u);
        ivec2 srcXY = ivec2(groupOriginTexelPos) + ivec2(sharedXY) - 2;
        srcXY = clamp(srcXY, ivec2(0), ivec2(uval_mainImageSize - 1));

        vec3 neighborDiff = transient_gi2Reprojected_fetch(srcXY).xyz;
        vec3 neighborSpec = transient_gi4Reprojected_fetch(srcXY).xyz;
        vec3 neighborDiffYCoCg = colors_SRGBToYCoCg(neighborDiff);
        vec3 neighborSpecYCoCg = colors_SRGBToYCoCg(neighborSpec);

        uvec4 packedData;
        packedData.xy = packHalf4x16(vec4(neighborDiffYCoCg, 0.0));
        packedData.zw = packHalf4x16(vec4(neighborSpecYCoCg, 0.0));
        shared_YCoCgData[sharedXY.y][sharedXY.x] = packedData;
    }
}

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
#endif

void main() {
    if (hiz_groupGroundCheck(gl_WorkGroupID.xy, 4)) {
        #if ENABLE_DENOISER_FAST_CLAMP
        loadSharedDataMoments(gl_LocalInvocationIndex);
        loadSharedDataMoments(gl_LocalInvocationIndex + 256u);
        #endif

        ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
        if (all(lessThan(texelPos, uval_mainImageSizeI))) {
            float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(gl_WorkGroupID.xy, 4, texelPos);
            if (viewZ > -65536.0) {
                GIHistoryData historyData = gi_historyData_init();

                vec4 giData1 = transient_gi1Reprojected_fetch(texelPos);
                vec4 giData2 = transient_gi3Reprojected_fetch(texelPos);

                gi_historyData_unpack1(historyData, transient_gi1Reprojected_fetch(texelPos));
                gi_historyData_unpack2(historyData, transient_gi2Reprojected_fetch(texelPos));
                gi_historyData_unpack3(historyData, transient_gi3Reprojected_fetch(texelPos));
                gi_historyData_unpack4(historyData, transient_gi4Reprojected_fetch(texelPos));
                gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));

                vec3 diffMoment1 = vec3(0.0);
                vec3 diffMoment2 = vec3(0.0);
                vec3 specMoment1 = vec3(0.0);
                vec3 specMoment2 = vec3(0.0);

                float historyFixMix = 1.0;

                vec2 texelPos0 = vec2(texelPos) + 0.5;

                vec2 localLuma = vec2(0.0);

                #if DENOISER_HISTORY_FIX
                vec3 geomNormal0 = normalize(transient_geomViewNormal_fetch(texelPos).xyz * 2.0 - 1.0);
                float viweZ0 = texelFetch(usam_gbufferViewZ, texelPos, 0).x;

                vec4 diffWeightedSum = vec4(0.0);
                vec4 specWeightedSum = vec4(0.0);
                float weightSum = 0.0;

                const float baseReductionFactor = ldexp(1.0, -12);
                float baseDepthWeight = max(2.0, abs(viweZ0)) * ldexp(1.0, -8 + SETTING_DENOISER_HISTORY_FIX_DEPTH_WEIGHT);
                #endif

                for (int mip = 6; mip >= 1; mip--) {
                    ivec2 stbnPos = ivec2(texelPos0 + rand_r2Seq2(mip) * 128.0);
                    vec2 stbnRand = rand_stbnVec2(stbnPos, RANDOM_FRAME);
                    vec2 texelPosMip = ldexp(texelPos0, ivec2(-mip)) + stbnRand - 0.5;

                    ivec4 giMipTile = global_mipmapTileCeil[mip];

                    vec2 texelPosMipTile = clamp(texelPosMip, vec2(1.5), vec2(giMipTile.zw) - vec2(1.5));
                    texelPosMipTile += vec2(giMipTile.xy);
                    vec2 screenPosMipTile = texelPosMipTile * uval_mainImageSizeRcp;

                    vec4 mipDiff = transient_gi_diffMip_sample(screenPosMipTile);
                    vec4 mipSpec = transient_gi_specMip_sample(screenPosMipTile);

                    #if DENOISER_HISTORY_FIX
                    ivec4 hizTile = global_hizTiles[mip];
                    vec2 hiZReadPos = hizTile.xy + ivec2(texelPosMip);
                    vec2 hiZVal = texelFetch(usam_hiz, ivec2(hiZReadPos), 0).rg;
                    float hiZMin = hiZVal.x;
                    float hiZMax = hiZVal.y;

                    vec3 geomNormalMipRaw = transient_geomNormalMip_sample(screenPosMipTile).xyz;
                    geomNormalMipRaw = geomNormalMipRaw * 2.0 - 1.0;
                    vec3 geomNormalMip = normalize(geomNormalMipRaw);
                    float geomNormalLengthSq = saturate(lengthSq(geomNormalMipRaw));
                    float geomNormalDot = dot(geomNormal0, geomNormalMipRaw * inversesqrt(geomNormalLengthSq));
                    float geomNormalWeight = geomNormalLengthSq * pow2(saturate(geomNormalDot));
                    geomNormalWeight = pow(geomNormalWeight, ldexp(0.1, mip + SETTING_DENOISER_HISTORY_FIX_NORMAL_WEIGHT));

                    hiZMax = coords_reversedZToViewZ(hiZMax, nearPlane);
                    hiZMin = coords_reversedZToViewZ(hiZMin, nearPlane);
                    float maxZWeight = baseDepthWeight / (baseDepthWeight + abs(hiZMax - viweZ0));
                    float minZWeight = baseDepthWeight / (baseDepthWeight + abs(hiZMin - viweZ0));

                    float reductionFactor = baseReductionFactor / (baseReductionFactor + weightSum);

                    float sampleWeight = 1.0;
                    sampleWeight *= geomNormalWeight;
                    sampleWeight *= maxZWeight;
                    sampleWeight *= minZWeight;
                    sampleWeight *= reductionFactor;

                    diffWeightedSum += mipDiff * sampleWeight;
                    specWeightedSum += mipSpec * sampleWeight;
                    weightSum += sampleWeight;
                    #endif

                    float mipWeight = ldexp(1.0, mip - 1);
                    float diffLuma = colors2_colorspaces_luma(SETTING_WORKING_COLOR_SPACE, mipDiff.rgb);
                    float specLuma = colors2_colorspaces_luma(SETTING_WORKING_COLOR_SPACE, mipSpec.rgb);
                    localLuma += vec2(diffLuma, specLuma) * mipWeight;
                }

//                float rcpMipWeightSum = rcp(1.0 + 2.0 + 4.0 + 8.0 + 16.0 + 32.0);
//                float rcpMipWeightSum = rcp(63.0);
                float rcpMipWeightSum = 0.0158730159;
                localLuma *= rcpMipWeightSum;
                vec2 localLumaRcp = safeRcp(localLuma);
                transient_gi_localLuma_store(texelPos, vec4(localLuma, 0.0, 0.0));

//                imageStore(uimg_temp3, texelPos, localLuma.xxxx);
                #if DENOISER_HISTORY_FIX
                float baseMipWeight = max((baseReductionFactor / (baseReductionFactor + weightSum)) * 1e-8, 1e-16);
                diffWeightedSum += vec4(historyData.diffuseColor * baseMipWeight, 0.0);
                specWeightedSum += vec4(historyData.specularColor * baseMipWeight, 0.0);
                weightSum += baseMipWeight;

                float historyLengthInt = historyData.historyLength * HISTORY_LENGTH;
                historyFixMix = 1.0 - pow2(linearStep(4.0, 1.0, historyLengthInt));
                float rcpWeightSum = 1.0 / weightSum;
                diffWeightedSum *= rcpWeightSum;
                specWeightedSum *= rcpWeightSum;

                diffWeightedSum = max(diffWeightedSum, vec4(0.0));
                specWeightedSum = max(specWeightedSum, vec4(0.0));

                #if SETTING_DEBUG_OUTPUT
//                imageStore(uimg_temp3, texelPos, diffWeightedSum);
                #endif

                historyData.diffuseColor = mix(diffWeightedSum.rgb, historyData.diffuseColor, historyFixMix);
                historyData.specularColor = mix(specWeightedSum.rgb, historyData.specularColor, historyFixMix);
                #endif

                vec2 denoiserBlurVariance = vec2(0.0);

                #if ENABLE_DENOISER_FAST_CLAMP
                {
                    float len = historyData.realHistoryLength * REAL_HISTORY_LENGTH;
                    float decayFactor = linearStep(FAST_HISTORY_LENGTH * 2.0, 1.0, historyData.realHistoryLength * REAL_HISTORY_LENGTH);
                    float clampingThreshold = mix(2.0, 16.0, pow2(decayFactor));

                    barrier();
                    ivec2 localPos = ivec2(gl_LocalInvocationID.xy) + 2; // +2 for padding
                    // 5x5 neighborhood using shared memory
                    for (int dy = -2; dy <= 2; ++dy) {
                        for (int dx = -2; dx <= 2; ++dx) {
                            ivec2 samplePos = localPos + ivec2(dx, dy);
                            uvec4 packedData = shared_YCoCgData[samplePos.y][samplePos.x];
                            vec3 neighborDiffYCoCg = unpackHalf4x16(packedData.xy).xyz;
                            vec3 neighborSpecYCoCg = unpackHalf4x16(packedData.zw).xyz;

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

                    float expMul = exp2(global_aeData.expValues.z);

                    vec3 diffClamped = _clampColor(historyData.diffuseColor, historyData.diffuseFastColor, diffMoment1, diffMoment2, clampingThreshold);
                    vec3 diffDiff = abs(agxInvertible_forward(diffClamped) - agxInvertible_forward(historyData.diffuseColor));
                    float diffDiffLuma = colors2_colorspaces_luma(SETTING_OUTPUT_COLOR_SPACE, diffDiff);
                    diffClamped = mix(historyData.diffuseColor, diffClamped, historyFixMix);
                    historyData.diffuseColor = diffClamped;

                    vec3 specClamped = _clampColor(historyData.specularColor, historyData.specularFastColor, specMoment1, specMoment2, clampingThreshold);
                    vec3 specDiff = abs(agxInvertible_forward(specClamped) - agxInvertible_forward(historyData.specularColor));
                    float specDiffLuma = colors2_colorspaces_luma(SETTING_OUTPUT_COLOR_SPACE, specDiff);
                    specClamped = mix(historyData.specularColor, specClamped, historyFixMix);
                    historyData.specularColor = specClamped;

                    vec2 diffLuma2 = sqrt(vec2(diffDiffLuma, specDiffLuma));
                    denoiserBlurVariance += diffLuma2;

                    vec2 localLumaClamped = max(localLuma, 1e-16);
                    vec2 resetFactor2 = pow2(linearStep(1.0, 0.0, diffLuma2));
                    float resetFactor = resetFactor2.x * resetFactor2.y;
                    historyData.historyLength *= resetFactor;
                    historyData.realHistoryLength *= sqrt(resetFactor);
                }
                #endif

                transient_gi1Reprojected_store(texelPos, gi_historyData_pack1(historyData));
                transient_gi2Reprojected_store(texelPos, gi_historyData_pack2(historyData));
                transient_gi3Reprojected_store(texelPos, gi_historyData_pack3(historyData));
                transient_gi4Reprojected_store(texelPos, gi_historyData_pack4(historyData));
                transient_gi5Reprojected_store(texelPos, gi_historyData_pack5(historyData));

                vec4 diffInput = vec4(historyData.diffuseColor, denoiserBlurVariance.x);
                vec4 specInput = vec4(historyData.specularColor, denoiserBlurVariance.y);

                transient_gi_blurDiff2_store(texelPos, diffInput);
                transient_gi_blurSpec2_store(texelPos, specInput);
            }
        }
    }
}
