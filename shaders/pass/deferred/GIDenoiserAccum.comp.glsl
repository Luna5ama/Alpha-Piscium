#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/gi/Common.glsl"
#include "/util/GBufferData.glsl"
#include "/techniques/HiZCheck.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_temp2;
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(rgba8) uniform writeonly image2D uimg_rgba8;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        transient_gi_diffMip_store(texelPos, vec4(0.0));
        transient_gi_specMip_store(texelPos, vec4(0.0));
        transient_geomNormalMip_store(texelPos, vec4(0.0));
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(gl_WorkGroupID.xy, 4, texelPos);
        if (viewZ > -65536.0) {
            vec4 newDiffuse = transient_ssgiOut_fetch(texelPos);

            GIHistoryData historyData = gi_historyData_init();

            gi_historyData_unpack1(historyData, transient_gi1Reprojected_fetch(texelPos));
            gi_historyData_unpack2(historyData, transient_gi2Reprojected_fetch(texelPos));
            gi_historyData_unpack3(historyData, transient_gi3Reprojected_fetch(texelPos));
            gi_historyData_unpack4(historyData, transient_gi4Reprojected_fetch(texelPos));
            gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));

            if (RANDOM_FRAME >= 0) {
                float currEdgeMask = transient_edgeMask_fetch(texelPos).r;
                historyData.edgeMask = currEdgeMask;

                const float HISTORY_LENGTH_1 = HISTORY_LENGTH - 1.0;
                float historyLength = historyData.historyLength * HISTORY_LENGTH_1;
                #if ENABLE_DENOISER_ACCUM

                historyLength += 1.0;
                #else
                historyLength = 0.0;
                #endif
                historyLength = clamp(historyLength, 1.0, HISTORY_LENGTH);

                float realHistoryLength = historyData.realHistoryLength * REAL_HISTORY_LENGTH;
                #if ENABLE_DENOISER_ACCUM
                realHistoryLength += 1.0;
                #else
                realHistoryLength = 0.0;
                #endif
                historyData.realHistoryLength = saturate(realHistoryLength / REAL_HISTORY_LENGTH);

                // Accumulate
                float alpha = 1.0 / historyLength;
                historyData.diffuseColor = mix(historyData.diffuseColor, newDiffuse.rgb, alpha);
                historyData.specularColor = mix(historyData.specularColor, vec3(0.0), alpha);// TODO: specular input
                historyData.historyLength = saturate(historyLength / HISTORY_LENGTH_1);

                float fastAlpha = 1.0 / min(historyLength, FAST_HISTORY_LENGTH);
                historyData.diffuseFastColor = mix(historyData.diffuseFastColor, newDiffuse.rgb, fastAlpha);
                historyData.specularFastColor = mix(historyData.specularFastColor, vec3(0.0), fastAlpha);

                InitialSampleData initialSample = initialSampleData_unpack(transient_restir_initialSample_fetch(texelPos));
                float newHitDistance = initialSample.directionAndLength.w;

                if (newHitDistance >= 0.0) {
                    newHitDistance = min(newHitDistance, MAX_HIT_DISTANCE);
                    historyData.diffuseHitDistance = mix(historyData.diffuseHitDistance, newHitDistance, fastAlpha);
                }
            }

            transient_gi1Reprojected_store(texelPos, gi_historyData_pack1(historyData));
            transient_gi2Reprojected_store(texelPos, gi_historyData_pack2(historyData));
            transient_gi3Reprojected_store(texelPos, gi_historyData_pack3(historyData));
            transient_gi4Reprojected_store(texelPos, gi_historyData_pack4(historyData));
            transient_gi5Reprojected_store(texelPos, gi_historyData_pack5(historyData));
        }
    }
}
