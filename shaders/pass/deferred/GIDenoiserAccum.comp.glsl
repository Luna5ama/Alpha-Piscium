#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/gi/Common.glsl"
#include "/util/GBufferData.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/util/AgxInvertible.glsl"
#include "/util/Rand.glsl"
#include "/util/Dither.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_temp2;
layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(rgba8) uniform writeonly image2D uimg_rgba8;

float computeOutputLumaDiffWeight(vec3 prevLinearColor, vec3 newLinearColor, float expMul, float threshold) {
    vec3 prevOutputSim = colors_reversibleTonemap(prevLinearColor * expMul);
    float prevOutputSimLuma = colors2_colorspaces_luma(SETTING_WORKING_COLOR_SPACE, prevOutputSim);
    vec3 newInputSim = colors_reversibleTonemap(newLinearColor * expMul);
    float newInputSimLuma = colors2_colorspaces_luma(SETTING_WORKING_COLOR_SPACE, newInputSim);
    float lumaDiff = newInputSimLuma - prevOutputSimLuma;

    return threshold / (threshold + pow2(lumaDiff));
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        transient_gi_diffMip_store(texelPos, vec4(0.0));
        transient_gi_specMip_store(texelPos, vec4(0.0));
        transient_geomNormalMip_store(texelPos, vec4(0.0));
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(gl_WorkGroupID.xy, 4, texelPos);
        if (viewZ > -65536.0) {
            vec4 newDiffuse = transient_ssgiOut_fetch(texelPos);
            vec4 newSpecular = vec4(0.0); // TODO: specular input

            GIHistoryData historyData = gi_historyData_init();

            gi_historyData_unpack1(historyData, transient_gi1Reprojected_fetch(texelPos));
            gi_historyData_unpack2(historyData, transient_gi2Reprojected_fetch(texelPos));
            gi_historyData_unpack3(historyData, transient_gi3Reprojected_fetch(texelPos));
            gi_historyData_unpack4(historyData, transient_gi4Reprojected_fetch(texelPos));
            gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));

            if (RANDOM_FRAME >= 0 && RANDOM_FRAME < MAX_FRAMES) {
                float currEdgeMask = transient_edgeMask_fetch(texelPos).r;
                historyData.edgeMask = currEdgeMask;

                float historyLength = 1.0;
                float realHistoryLength = 1.0;
                // x: diffuse
                // y: specular
                vec2 newWeights = vec2(1.0);

                #if ENABLE_DENOISER_ACCUM
                historyLength = historyData.historyLength * TOTAL_HISTORY_LENGTH * global_historyResetFactor;
                historyLength += 1.0;
                realHistoryLength = historyData.realHistoryLength * TOTAL_HISTORY_LENGTH * global_historyResetFactor;
                realHistoryLength += 1.0;

                #if SETTING_DENOISER_FIREFLY_SUPPRESSION
                // Idea from Belmu to limit firefly based on luma difference
                float expMul = exp2(global_aeData.expValues.z);
                float threshold = ldexp(1.0, -SETTING_DENOISER_FIREFLY_SUPPRESSION);
                newWeights.x = computeOutputLumaDiffWeight(historyData.diffuseColor, newDiffuse.rgb, expMul, threshold);
                newWeights.y = computeOutputLumaDiffWeight(historyData.specularColor, newSpecular.rgb, expMul, threshold);
                #endif
                #endif

                // Accumulate
                // x: regular history length
                // y: fast history length
                vec2 accumHistoryLength = min(vec2(historyLength), vec2(HISTORY_LENGTH, FAST_HISTORY_LENGTH));
                vec2 rcpAccumHistoryLength = rcp(accumHistoryLength);
                // x: regular, diffuse
                // y: regular, specular
                // z: fast, diffuse
                // w: fast, specular
                vec4 alpha = vec4(newWeights.xy, sqrt(newWeights.xy)) * rcpAccumHistoryLength.xxyy;

                historyData.diffuseColor = mix(historyData.diffuseColor, newDiffuse.rgb, alpha.x);
                historyData.specularColor = mix(historyData.specularColor, newSpecular.rgb, alpha.y);

                historyData.diffuseFastColor = mix(historyData.diffuseFastColor, newDiffuse.rgb, alpha.z);
                historyData.specularFastColor = mix(historyData.specularFastColor, newSpecular.rgb, alpha.w);// TODO: specular input

                InitialSampleData initialSample = initialSampleData_unpack(transient_restir_initialSample_fetch(texelPos));
                float newHitDistance = initialSample.directionAndLength.w;

                float regularFastAlpha = rcpAccumHistoryLength.y;

                if (newHitDistance >= 0.0) {
                    newHitDistance = min(newHitDistance, MAX_HIT_DISTANCE);
                    historyData.diffuseHitDistance = mix(historyData.diffuseHitDistance, newHitDistance, regularFastAlpha);
                }

                historyLength = clamp(historyLength, 1.0, TOTAL_HISTORY_LENGTH);
                realHistoryLength = clamp(realHistoryLength, 1.0, TOTAL_HISTORY_LENGTH);
                historyData.historyLength = saturate(historyLength / TOTAL_HISTORY_LENGTH);
                historyData.realHistoryLength = saturate(realHistoryLength / TOTAL_HISTORY_LENGTH);

                float ditherNoise = rand_stbnVec1(texelPos, frameCounter);
                historyData.diffuseColor = dither_fp16(historyData.diffuseColor, ditherNoise);
                historyData.specularColor = dither_fp16(historyData.specularColor, ditherNoise);
                historyData.diffuseFastColor = dither_fp16(historyData.diffuseFastColor, ditherNoise);
                historyData.specularFastColor = dither_fp16(historyData.specularFastColor, ditherNoise);
            }

            transient_gi1Reprojected_store(texelPos, gi_historyData_pack1(historyData));
            transient_gi2Reprojected_store(texelPos, gi_historyData_pack2(historyData));
            transient_gi3Reprojected_store(texelPos, gi_historyData_pack3(historyData));
            transient_gi4Reprojected_store(texelPos, gi_historyData_pack4(historyData));
            transient_gi5Reprojected_store(texelPos, gi_historyData_pack5(historyData));
        }
    }
}
