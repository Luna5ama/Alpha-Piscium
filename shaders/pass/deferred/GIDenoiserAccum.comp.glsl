#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/gi/Common.glsl"
#include "/util/GBufferData.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/util/AgxInvertible.glsl"
#include "/util/Rand.glsl"
#include "/util/Dither.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_temp2;
layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(rgba8) uniform writeonly image2D uimg_rgba8;

// Shared memory with padding for 5x5 tap (-2 to +2)
// Each work group is 16x16, need +2 padding on each side for 5x5 taps
shared vec2 shared_historyLengths[18][18];

void loadSharedHistoryLengths(uvec2 groupOriginTexelPos, uint index) {
    if (index < 324u) { // 18 * 18 = 324
        uvec2 sharedXY = uvec2(index % 18u, index / 18u);
        ivec2 srcXY = ivec2(groupOriginTexelPos) + ivec2(sharedXY) - 1;
        vec4 data5 = transient_gi5Reprojected_fetch(srcXY);
        shared_historyLengths[sharedXY.y][sharedXY.x] = data5.xy;
    }
}

float computeOutputLumaDiffWeight(vec3 prevLinearColor, vec3 newLinearColor, float expMul, float threshold) {
    vec3 prevOutputSim = colors_reversibleTonemap(prevLinearColor * expMul);
    float prevOutputSimLuma = colors2_colorspaces_luma(SETTING_WORKING_COLOR_SPACE, prevOutputSim);
    vec3 newInputSim = colors_reversibleTonemap(newLinearColor * expMul);
    float newInputSimLuma = colors2_colorspaces_luma(SETTING_WORKING_COLOR_SPACE, newInputSim);
    float lumaDiff = newInputSimLuma - prevOutputSimLuma;

    return threshold / (threshold + pow2(lumaDiff));
}

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        transient_gi_diffMip_store(texelPos, vec4(0.0));
        transient_gi_specMip_store(texelPos, vec4(0.0));
        transient_geomNormalMip_store(texelPos, vec4(0.0));
    }

    if (hiz_groupGroundCheck(swizzledWGPos, 4)) {
        loadSharedHistoryLengths(workGroupOrigin, gl_LocalInvocationIndex);
        loadSharedHistoryLengths(workGroupOrigin, gl_LocalInvocationIndex + 256u);

        if (all(lessThan(texelPos, uval_mainImageSizeI))) {
            float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).x;
            if (viewZ > -65536.0) {
                vec4 newDiffuse = transient_ssgiOut_fetch(texelPos);
                vec4 newSpecular = vec4(0.0); // TODO: specular input

                GIHistoryData historyData = gi_historyData_init();

                gi_historyData_unpack1(historyData, transient_gi1Reprojected_fetch(texelPos));
                gi_historyData_unpack2(historyData, transient_gi2Reprojected_fetch(texelPos));
                gi_historyData_unpack3(historyData, transient_gi3Reprojected_fetch(texelPos));
                gi_historyData_unpack4(historyData, transient_gi4Reprojected_fetch(texelPos));
                gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));
                barrier();

                if (RANDOM_FRAME >= 0 && RANDOM_FRAME < MAX_FRAMES) {
                    float currEdgeMask = transient_edgeMask_fetch(texelPos).r;
                    historyData.edgeMask = currEdgeMask;

                    float historyLength = 1.0;
                    float realHistoryLength = 1.0;
                    // x: diffuse
                    // y: specular
                    vec2 newWeights = vec2(1.0);

                    #ifdef SETTING_DENOISER_ACCUM

                    historyLength = historyData.historyLength * TOTAL_HISTORY_LENGTH * global_historyResetFactor;
                    historyLength += 1.0;
                    realHistoryLength = historyData.realHistoryLength * TOTAL_HISTORY_LENGTH * global_historyResetFactor;
                    realHistoryLength += 1.0;

                    #if SETTING_DENOISER_FIREFLY_SUPPRESSION
                    // Idea from Belmu to limit firefly based on luma difference
                    if (historyData.realHistoryLength > 0.0) {
                        float expMul = exp2(global_aeData.expValues.z);
                        float threshold = ldexp(1.0, -SETTING_DENOISER_FIREFLY_SUPPRESSION);
                        newWeights.x = computeOutputLumaDiffWeight(historyData.diffuseColor, newDiffuse.rgb, expMul, threshold);
                        newWeights.y = computeOutputLumaDiffWeight(historyData.specularColor, newSpecular.rgb, expMul, threshold);
                    }
                    #endif
                    #endif

                    // Accumulate
                    // x: regular history length
                    // y: fast history length
                    vec2 accumHistoryLength = min(vec2(historyLength, realHistoryLength), vec2(HISTORY_LENGTH, FAST_HISTORY_LENGTH));
                    vec2 rcpAccumHistoryLength = rcp(accumHistoryLength);
                    // x: regular, diffuse
                    // y: regular, specular
                    // z: fast, diffuse
                    // w: fast, specular
                    vec4 alpha = vec4(newWeights.xy, pow(newWeights.xy, vec2(0.1))) * rcpAccumHistoryLength.xxyy;

                    historyData.diffuseColor = mix(historyData.diffuseColor, newDiffuse.rgb, alpha.x);
                    historyData.specularColor = mix(historyData.specularColor, newSpecular.rgb, alpha.y);

                    historyData.diffuseFastColor = mix(historyData.diffuseFastColor, newDiffuse.rgb, alpha.z);
                    historyData.specularFastColor = mix(historyData.specularFastColor, newSpecular.rgb, alpha.w);// TODO: specular input

                    float newHitDistance = transient_gi_initialSampleHitDistance_fetch(texelPos).x;
                    float regularFastAlpha = rcpAccumHistoryLength.y;

                    if (newHitDistance >= 0.0) {
                        newHitDistance = min(newHitDistance, MAX_HIT_DISTANCE);
                        historyData.diffuseHitDistance = mix(historyData.diffuseHitDistance, newHitDistance, regularFastAlpha);
                    }

                    historyLength = clamp(historyLength, 1.0, TOTAL_HISTORY_LENGTH);
                    realHistoryLength = clamp(realHistoryLength, 1.0, TOTAL_HISTORY_LENGTH);
                    historyData.historyLength = saturate(historyLength / TOTAL_HISTORY_LENGTH);
                    historyData.realHistoryLength = saturate(realHistoryLength / TOTAL_HISTORY_LENGTH);
                }

                // 3x3 max kernel on history lengths
                vec2 maxHistoryLengths = vec2(0.0);
                ivec2 localPos = ivec2(mortonPos) + 1; // +1 for padding
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        ivec2 samplePos = localPos + ivec2(dx, dy);
                        vec2 neighborHistoryLengths = shared_historyLengths[samplePos.y][samplePos.x];
                        maxHistoryLengths = max(maxHistoryLengths, neighborHistoryLengths);
                    }
                }

                float maxMixWeight = linearStep(1.0, 4.0, historyData.realHistoryLength * TOTAL_HISTORY_LENGTH);
                historyData.historyLength = mix(historyData.historyLength, max(historyData.historyLength, maxHistoryLengths.x), maxMixWeight);
                historyData.realHistoryLength = mix(historyData.realHistoryLength, max(historyData.realHistoryLength, maxHistoryLengths.y), maxMixWeight);
                #if SETTING_DEBUG_OUTPUT
                imageStore(uimg_temp3, texelPos, gi_historyData_pack1(historyData));
                #endif

                float ditherNoise = rand_stbnVec1(rand_newStbnPos(texelPos, 1u), frameCounter);
                vec4 packedData1 = clamp(gi_historyData_pack1(historyData), 0.0, FP16_MAX);
                packedData1 = dither_fp16(packedData1, ditherNoise);
                vec4 packedData2 = clamp(gi_historyData_pack2(historyData), 0.0, FP16_MAX);
                packedData2 = dither_fp16(packedData2, ditherNoise);
                vec4 packedData3 = clamp(gi_historyData_pack3(historyData), 0.0, FP16_MAX);
                packedData3 = dither_fp16(packedData3, ditherNoise);
                vec4 packedData4 = clamp(gi_historyData_pack4(historyData), 0.0, FP16_MAX);
                packedData4 = dither_fp16(packedData4, ditherNoise);
                vec4 packedData5 = gi_historyData_pack5(historyData);
                packedData5 = dither_u8(packedData5, ditherNoise);

                transient_gi1Reprojected_store(texelPos, packedData1);
                transient_gi2Reprojected_store(texelPos, packedData2);
                transient_gi3Reprojected_store(texelPos, packedData3);
                transient_gi4Reprojected_store(texelPos, packedData4);
                transient_gi5Reprojected_store(texelPos, packedData5);
            }
        }
    }
}
