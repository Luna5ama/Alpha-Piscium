#include "Common.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Sampling.glsl"


vec4 bileratralSum(vec4 xs, vec4 ys, vec4 zs, vec4 ws, vec4 weights) {
    return vec4(
        dot(xs, weights),
        dot(ys, weights),
        dot(zs, weights),
        dot(ws, weights)
    );
}

void gi_reproject(ivec2 texelPos, float currViewZ, GBufferData gData) {
    GIHistoryData historyData = gi_historyData_init();
    ReprojectInfo reprojInfo = reprojectInfo_init();
    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    float currEdgeFactor = min4(transient_edgeMaskTemp_gather(screenPos, 0));

    screenPos -= global_taaJitter * uval_mainImageSizeRcp;

    vec3 currViewPos = coords_toViewCoord(screenPos, currViewZ, global_camProjInverse);
    vec4 curr2PrevViewPos = coord_viewCurrToPrev(vec4(currViewPos, 1.0), gData.isHand);
    vec4 curr2PrevClipPos = global_prevCamProj * curr2PrevViewPos;
    uint clipFlag = uint(curr2PrevClipPos.z > 0.0);
    clipFlag &= uint(all(lessThan(abs(curr2PrevClipPos.xy), curr2PrevClipPos.ww)));
    vec3 currViewNormal = gData.normal;
    vec3 currViewGeomNormal = gData.geomNormal;
    vec3 currWorldGeomNormal = coords_dir_viewToWorld(currViewGeomNormal);
    vec3 currWorldNormal = coords_dir_viewToWorld(currViewNormal);

    float glazingCosTheta = saturate(dot(currViewGeomNormal, -normalize(currViewPos.xyz)));
    float glazingAngleFactor = glazingCosTheta;
    float glazingAngleFactorHistory = pow4(1.0 - glazingCosTheta);

    if (bool(clipFlag)) {
        vec2 curr2PrevNDC = curr2PrevClipPos.xy / curr2PrevClipPos.w;
        vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;

        if (all(equal(curr2PrevScreen, saturate(curr2PrevScreen)))) {
            curr2PrevScreen += global_prevTaaJitter * uval_mainImageSizeRcp;
            vec2 curr2PrevTexelPos = curr2PrevScreen * uval_mainImageSize;

            vec3 curr2PrevViewNormal = coords_dir_worldToViewPrev(currWorldNormal);
            vec3 curr2PrevViewGeomNormal = coords_dir_worldToViewPrev(currWorldGeomNormal);

            vec2 centerPixel = curr2PrevTexelPos - 0.5;
            vec2 gatherOrigin = floor(centerPixel);
            vec2 gatherTexelPos = gatherOrigin + 1.0;
            vec2 gatherScreenPos = gatherTexelPos * uval_mainImageSizeRcp;
            vec2 pixelPosFract = fract(centerPixel);

            float prevEdgeMask = min4(history_gi5_gather(gatherScreenPos, 2));

            vec4 viewZs = history_viewZ_gatherTexel(gatherTexelPos, 0);
            vec4 geomViewNormalXs = history_geomViewNormal_gatherTexel(gatherTexelPos, 0) * 2.0 - 1.0;
            vec4 geomViewNormalYs = history_geomViewNormal_gatherTexel(gatherTexelPos, 1) * 2.0 - 1.0;
            vec4 geomViewNormalZs = history_geomViewNormal_gatherTexel(gatherTexelPos, 2) * 2.0 - 1.0;

            vec4 viewNormalXs = history_viewNormal_gatherTexel(gatherTexelPos, 0) * 2.0 - 1.0;
            vec4 viewNormalYs = history_viewNormal_gatherTexel(gatherTexelPos, 1) * 2.0 - 1.0;
            vec4 viewNormalZs = history_viewNormal_gatherTexel(gatherTexelPos, 2) * 2.0 - 1.0;

            vec2 halfTexel = 0.5 * uval_mainImageSizeRcp;
            vec2 gatherScreenPos1 = gatherScreenPos + halfTexel * vec2(-1.0, 1.0);
            vec2 gatherScreenPos2 = gatherScreenPos + halfTexel * vec2(1.0, 1.0);
            vec2 gatherScreenPos3 = gatherScreenPos + halfTexel * vec2(1.0, -1.0);
            vec2 gatherScreenPos4 = gatherScreenPos + halfTexel * vec2(-1.0, -1.0);

            vec3 prevViewPos1 = coords_toViewCoord(gatherScreenPos1, viewZs.x, global_prevCamProjInverse);
            vec3 prevViewPos2 = coords_toViewCoord(gatherScreenPos2, viewZs.y, global_prevCamProjInverse);
            vec3 prevViewPos3 = coords_toViewCoord(gatherScreenPos3, viewZs.z, global_prevCamProjInverse);
            vec3 prevViewPos4 = coords_toViewCoord(gatherScreenPos4, viewZs.w, global_prevCamProjInverse);

            vec3 geomViewNormal1 = normalize(vec3(geomViewNormalXs.x, geomViewNormalYs.x, geomViewNormalZs.x));
            vec3 geomViewNormal2 = normalize(vec3(geomViewNormalXs.y, geomViewNormalYs.y, geomViewNormalZs.y));
            vec3 geomViewNormal3 = normalize(vec3(geomViewNormalXs.z, geomViewNormalYs.z, geomViewNormalZs.z));
            vec3 geomViewNormal4 = normalize(vec3(geomViewNormalXs.w, geomViewNormalYs.w, geomViewNormalZs.w));

            float planeDistance1 = gi_planeDistance(curr2PrevViewPos.xyz, curr2PrevViewGeomNormal, prevViewPos1, geomViewNormal1);
            float planeDistance2 = gi_planeDistance(curr2PrevViewPos.xyz, curr2PrevViewGeomNormal, prevViewPos2, geomViewNormal2);
            float planeDistance3 = gi_planeDistance(curr2PrevViewPos.xyz, curr2PrevViewGeomNormal, prevViewPos3, geomViewNormal3);
            float planeDistance4 = gi_planeDistance(curr2PrevViewPos.xyz, curr2PrevViewGeomNormal, prevViewPos4, geomViewNormal4);

            float planeDistanceThreshold = exp2(mix(-8.0, -10.0, glazingAngleFactor)) * max(8.0, pow2(currViewZ));

            float geomViewNormalDot1 = dot(curr2PrevViewGeomNormal, geomViewNormal1);
            float geomViewNormalDot2 = dot(curr2PrevViewGeomNormal, geomViewNormal2);
            float geomViewNormalDot3 = dot(curr2PrevViewGeomNormal, geomViewNormal3);
            float geomViewNormalDot4 = dot(curr2PrevViewGeomNormal, geomViewNormal4);

            vec4 geomViewNormalDots = vec4(geomViewNormalDot1, geomViewNormalDot2, geomViewNormalDot3, geomViewNormalDot4);
            vec4 planeDistances = vec4(planeDistance1, planeDistance2, planeDistance3, planeDistance4);
            float totalEdgeFactor = (currEdgeFactor + prevEdgeMask) * 0.5;

            uint edgeFlag = 0u;
            edgeFlag |= uint(totalEdgeFactor < 0.99);
            edgeFlag |= uint(any(lessThan(geomViewNormalDots, vec4(0.5))));
            edgeFlag |= uint(any(greaterThan(planeDistances, vec4(planeDistanceThreshold))));

            float historyResetFactor = 1.0;

            vec4 geomNormalWeights = pow(saturate(geomViewNormalDots), vec4(128.0));
            float geomDepthBaseWeight = mix(32.0, 4.0, totalEdgeFactor) * mix(4.0, 1.0, glazingAngleFactor);
            vec4 geomDepthWegiths = exp2(-geomDepthBaseWeight * (planeDistances / max(abs(currViewZ), 2.0)));
            geomDepthWegiths *= saturate(step(planeDistances, vec4(planeDistanceThreshold)));
            vec4 edgeWeights = geomNormalWeights * geomDepthWegiths;

            vec2 bilinearWeights2 = pixelPosFract;
            vec4 blinearWeights4;
            blinearWeights4.yz = bilinearWeights2.xx;
            blinearWeights4.xw = 1.0 - bilinearWeights2.xx;
            blinearWeights4.xy *= bilinearWeights2.yy;
            blinearWeights4.zw *= 1.0 - bilinearWeights2.yy;

            bool edgeFlagBool = bool(edgeFlag);

            vec4 finalWeights = edgeWeights * blinearWeights4;
            float weightSum = dot(finalWeights, vec4(1.0));
            float rcpWeightSum = safeRcp(weightSum);
            finalWeights *= rcpWeightSum;

            reprojInfo.bilateralWeights = edgeWeights * safeRcp(max4(edgeWeights));
            reprojInfo.curr2PrevScreenPos = curr2PrevScreen;

            if (edgeFlagBool) {
                bool validFlag = weightSum > 0.001;
                historyResetFactor *= float(validFlag);
                // Only applying this to reprojInfo (which is used in ReSTIR temporal reuse)
                // Because killing denoiser history causes flickering bs
                reprojInfo.historyResetFactor *= weightSum;
                if (weightSum > 0.001) {
                    vec4 giData1 = bileratralSum(
                        history_gi1_gatherTexel(gatherTexelPos, 0),
                        history_gi1_gatherTexel(gatherTexelPos, 1),
                        history_gi1_gatherTexel(gatherTexelPos, 2),
                        history_gi1_gatherTexel(gatherTexelPos, 3),
                        finalWeights
                    );
                    gi_historyData_unpack1(historyData, giData1);

                    vec4 giData2 = bileratralSum(
                        history_gi2_gatherTexel(gatherTexelPos, 0),
                        history_gi2_gatherTexel(gatherTexelPos, 1),
                        history_gi2_gatherTexel(gatherTexelPos, 2),
                        history_gi2_gatherTexel(gatherTexelPos, 3),
                        finalWeights
                    );
                    gi_historyData_unpack2(historyData, giData2);

                    vec4 giData3 = bileratralSum(
                        history_gi3_gatherTexel(gatherTexelPos, 0),
                        history_gi3_gatherTexel(gatherTexelPos, 1),
                        history_gi3_gatherTexel(gatherTexelPos, 2),
                        history_gi3_gatherTexel(gatherTexelPos, 3),
                        finalWeights
                    );
                    gi_historyData_unpack3(historyData, giData3);

                    vec4 giData4 = bileratralSum(
                        history_gi4_gatherTexel(gatherTexelPos, 0),
                        history_gi4_gatherTexel(gatherTexelPos, 1),
                        history_gi4_gatherTexel(gatherTexelPos, 2),
                        history_gi4_gatherTexel(gatherTexelPos, 3),
                        finalWeights
                    );
                    gi_historyData_unpack4(historyData, giData4);

                    vec4 giData5 = bileratralSum(
                        history_gi5_gatherTexel(gatherTexelPos, 0),
                        history_gi5_gatherTexel(gatherTexelPos, 1),
                        history_gi5_gatherTexel(gatherTexelPos, 2),
                        history_gi5_gatherTexel(gatherTexelPos, 3),
                        finalWeights
                    );
                    gi_historyData_unpack5(historyData, giData5);
                }
            } else {
                CatmullBicubic5TapData tapData = sampling_catmullBicubic5Tap_init(curr2PrevTexelPos, 0.5, uval_mainImageSizeRcp);
                vec4 giData1 = sampling_catmullBicubic5Tap_sum(
                    history_gi1_sample(tapData.uv1AndWeight.xy),
                    history_gi1_sample(tapData.uv2AndWeight.xy),
                    history_gi1_sample(tapData.uv3AndWeight.xy),
                    history_gi1_sample(tapData.uv4AndWeight.xy),
                    history_gi1_sample(tapData.uv5AndWeight.xy),
                    tapData
                );
                gi_historyData_unpack1(historyData, giData1);

                vec4 giData2 = sampling_catmullBicubic5Tap_sum(
                    history_gi2_sample(tapData.uv1AndWeight.xy),
                    history_gi2_sample(tapData.uv2AndWeight.xy),
                    history_gi2_sample(tapData.uv3AndWeight.xy),
                    history_gi2_sample(tapData.uv4AndWeight.xy),
                    history_gi2_sample(tapData.uv5AndWeight.xy),
                    tapData
                );
                gi_historyData_unpack2(historyData, giData2);

                vec4 giData3 = sampling_catmullBicubic5Tap_sum(
                    history_gi3_sample(tapData.uv1AndWeight.xy),
                    history_gi3_sample(tapData.uv2AndWeight.xy),
                    history_gi3_sample(tapData.uv3AndWeight.xy),
                    history_gi3_sample(tapData.uv4AndWeight.xy),
                    history_gi3_sample(tapData.uv5AndWeight.xy),
                    tapData
                );
                gi_historyData_unpack3(historyData, giData3);

                vec4 giData4 = sampling_catmullBicubic5Tap_sum(
                    history_gi4_sample(tapData.uv1AndWeight.xy),
                    history_gi4_sample(tapData.uv2AndWeight.xy),
                    history_gi4_sample(tapData.uv3AndWeight.xy),
                    history_gi4_sample(tapData.uv4AndWeight.xy),
                    history_gi4_sample(tapData.uv5AndWeight.xy),
                    tapData
                );
                gi_historyData_unpack4(historyData, giData4);

                gi_historyData_unpack5(historyData, saturate(history_gi5_sample(curr2PrevScreen)));
            }

            float antiStretching = smoothstep(0.2, 0.0, historyData.glazingAngleFactor - glazingAngleFactorHistory);
            historyResetFactor *= antiStretching;

            historyData.historyLength *= historyResetFactor;
            reprojInfo.historyResetFactor *= historyResetFactor;
        }
    }

    historyData.glazingAngleFactor = glazingAngleFactorHistory;

    transient_gi1Reprojected_store(texelPos, clamp(gi_historyData_pack1(historyData), 0.0, FP16_MAX));
    transient_gi2Reprojected_store(texelPos, clamp(gi_historyData_pack2(historyData), 0.0, FP16_MAX));
    transient_gi3Reprojected_store(texelPos, clamp(gi_historyData_pack3(historyData), 0.0, FP16_MAX));
    transient_gi4Reprojected_store(texelPos, clamp(gi_historyData_pack4(historyData), 0.0, FP16_MAX));
    transient_gi5Reprojected_store(texelPos, gi_historyData_pack5(historyData));

    transient_gi_diffuse_reprojInfo_store(texelPos, reprojectInfo_pack(reprojInfo));
}
