#include "Common.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Sampling.glsl"
#include "/util/Rand.glsl"
#include "/util/Dither.glsl"

layout(rgba16f) uniform restrict writeonly image2D uimg_temp3;

vec4 bileratralSum(vec4 xs, vec4 ys, vec4 zs, vec4 ws, vec4 weights) {
    return vec4(
        dot(xs, weights),
        dot(ys, weights),
        dot(zs, weights),
        dot(ws, weights)
    );
}

vec4 computeEdgeWeights(
vec2 screenPos,
vec2 gatherTexelPos,
vec3 currViewNormal,
vec3 currViewGeomNormal,
vec3 curr2PrevViewPos,
float glazingAngleFactor,
float normalBaseWeight,
out vec4 edgeWeights,
out bool edgeFlag
) {

    vec4 viewZs = history_viewZ_gatherTexel(gatherTexelPos, 0);
    vec4 geomViewNormalXs = history_geomViewNormal_gatherTexel(gatherTexelPos, 0);
    vec4 geomViewNormalYs = history_geomViewNormal_gatherTexel(gatherTexelPos, 1);
    vec4 geomViewNormalZs = history_geomViewNormal_gatherTexel(gatherTexelPos, 2);
    geomViewNormalXs = geomViewNormalXs * 2.0 - 1.0;
    geomViewNormalYs = geomViewNormalYs * 2.0 - 1.0;
    geomViewNormalZs = geomViewNormalZs * 2.0 - 1.0;

    vec4 viewNormalXs = history_viewNormal_gatherTexel(gatherTexelPos, 0);
    vec4 viewNormalYs = history_viewNormal_gatherTexel(gatherTexelPos, 1);
    vec4 viewNormalZs = history_viewNormal_gatherTexel(gatherTexelPos, 2);
    viewNormalXs = viewNormalXs * 2.0 - 1.0;
    viewNormalYs = viewNormalYs * 2.0 - 1.0;
    viewNormalZs = viewNormalZs * 2.0 - 1.0;

    vec3 currWorldGeomNormal = coords_dir_viewToWorld(currViewGeomNormal);
    vec3 currWorldNormal = coords_dir_viewToWorld(currViewNormal);
    vec3 curr2PrevViewNormal = coords_dir_worldToViewPrev(currWorldNormal);
    vec3 curr2PrevViewGeomNormal = coords_dir_worldToViewPrev(currWorldGeomNormal);

    float currEdgeFactor = min4(transient_edgeMaskTemp_gather(screenPos, 0));

    vec2 gatherScreenPos = gatherTexelPos * uval_mainImageSizeRcp;
    float prevEdgeMask = min4(history_gi5_gather(gatherScreenPos, 2));

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

    vec3 viewNormal1 = normalize(vec3(viewNormalXs.x, viewNormalYs.x, viewNormalZs.x));
    vec3 viewNormal2 = normalize(vec3(viewNormalXs.y, viewNormalYs.y, viewNormalZs.y));
    vec3 viewNormal3 = normalize(vec3(viewNormalXs.z, viewNormalYs.z, viewNormalZs.z));
    vec3 viewNormal4 = normalize(vec3(viewNormalXs.w, viewNormalYs.w, viewNormalZs.w));

    float planeDistance1 = gi_planeDistance(curr2PrevViewPos.xyz, curr2PrevViewGeomNormal, prevViewPos1, geomViewNormal1);
    float planeDistance2 = gi_planeDistance(curr2PrevViewPos.xyz, curr2PrevViewGeomNormal, prevViewPos2, geomViewNormal2);
    float planeDistance3 = gi_planeDistance(curr2PrevViewPos.xyz, curr2PrevViewGeomNormal, prevViewPos3, geomViewNormal3);
    float planeDistance4 = gi_planeDistance(curr2PrevViewPos.xyz, curr2PrevViewGeomNormal, prevViewPos4, geomViewNormal4);

    float planeDistanceThreshold = exp2(mix(-8.0, -10.0, glazingAngleFactor)) * max(8.0, pow2(curr2PrevViewPos.z));

    float geomViewNormalDot1 = saturate(dot(curr2PrevViewGeomNormal, geomViewNormal1));
    float geomViewNormalDot2 = saturate(dot(curr2PrevViewGeomNormal, geomViewNormal2));
    float geomViewNormalDot3 = saturate(dot(curr2PrevViewGeomNormal, geomViewNormal3));
    float geomViewNormalDot4 = saturate(dot(curr2PrevViewGeomNormal, geomViewNormal4));

    float viewNormalDot1 = saturate(dot(curr2PrevViewNormal, viewNormal1));
    float viewNormalDot2 = saturate(dot(curr2PrevViewNormal, viewNormal2));
    float viewNormalDot3 = saturate(dot(curr2PrevViewNormal, viewNormal3));
    float viewNormalDot4 = saturate(dot(curr2PrevViewNormal, viewNormal4));

    vec4 geomViewNormalDots = vec4(geomViewNormalDot1, geomViewNormalDot2, geomViewNormalDot3, geomViewNormalDot4);
    vec4 viewNormalDots = vec4(viewNormalDot1, viewNormalDot2, viewNormalDot3, viewNormalDot4);
    vec4 planeDistances = vec4(planeDistance1, planeDistance2, planeDistance3, planeDistance4);
    float totalEdgeFactor = (currEdgeFactor + prevEdgeMask) * 0.5;

    uint edgeFlagI = 0u;
    edgeFlagI |= uint(totalEdgeFactor < 0.99);
    edgeFlagI |= uint(any(lessThan(geomViewNormalDots, vec4(0.5))));
    edgeFlagI |= uint(any(lessThan(viewNormalDots, vec4(0.5))));
    edgeFlagI |= uint(any(greaterThan(planeDistances, vec4(planeDistanceThreshold))));
    edgeFlag = bool(edgeFlagI);

    vec4 geomNormalWeights = pow(geomViewNormalDots, vec4(256.0));
    vec4 normalWeights = pow(viewNormalDots, vec4(normalBaseWeight));
    float geomDepthBaseWeight = mix(32.0, 4.0, totalEdgeFactor) * mix(4.0, 1.0, glazingAngleFactor);
    vec4 geomDepthWeights = exp2(-geomDepthBaseWeight * (planeDistances / max(abs(curr2PrevViewPos.z), 2.0)));
    geomDepthWeights *= saturate(step(planeDistances, vec4(planeDistanceThreshold)));
    edgeWeights = geomNormalWeights * normalWeights * geomDepthWeights;
    return normalWeights;
}

void gi_reproject(ivec2 texelPos, float currViewZ) {
    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    float currEdgeFactor = min4(transient_edgeMaskTemp_gather(screenPos, 0));
    bool currEdgeFlag = currEdgeFactor < 0.99;

    if (currEdgeFlag){
        screenPos -= uval_taaJitter * uval_mainImageSizeRcp;
    }
    GBufferData gData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, texelPos, 0), gData);
    gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelPos, 0), gData);

    vec3 currViewPos = coords_toViewCoord(screenPos, currViewZ, global_camProjInverse);
    vec4 curr2PrevViewPos = coord_viewCurrToPrev(vec4(currViewPos, 1.0), gData.isHand);
    vec4 curr2PrevClipPos = global_prevCamProj * curr2PrevViewPos;
    uint clipFlag = uint(curr2PrevClipPos.z > 0.0);
    clipFlag &= uint(all(lessThan(abs(curr2PrevClipPos.xy), curr2PrevClipPos.ww)));
    vec3 currViewNormal = gData.normal;
    vec3 currViewGeomNormal = gData.geomNormal;

    float glazingCosTheta = saturate(dot(currViewGeomNormal, -normalize(currViewPos.xyz)));
    float glazingAngleFactor = glazingCosTheta;
    float glazingAngleFactorHistory = pow2(1.0 - glazingCosTheta);
    bool valid = false;
    float specularHitDistance = 0.0;

    if (bool(clipFlag)) {
        vec2 curr2PrevNDC = curr2PrevClipPos.xy / curr2PrevClipPos.w;
        vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;
        vec2 curr2PrevScreenClamped = saturate(curr2PrevScreen);

        if (all(lessThan(abs(curr2PrevScreen - curr2PrevScreenClamped), uval_mainImageSizeRcp * 2.0))) {
            if (currEdgeFlag){
                curr2PrevScreen += uval_prevTaaJitter * uval_mainImageSizeRcp;
            }
            vec2 curr2PrevTexelPos = curr2PrevScreen * uval_mainImageSize;
            curr2PrevTexelPos = clamp(curr2PrevTexelPos, vec2(1.0), uval_mainImageSize - 1.0);

            vec2 gatherTexelPos = floor(curr2PrevTexelPos - 0.5) + 1.0;

            bool edgeFlag;
            vec4 edgeWeights;
            vec4 extraNormalWeights = computeEdgeWeights(
                screenPos,
                gatherTexelPos,
                currViewNormal,
                currViewGeomNormal,
                curr2PrevViewPos.xyz,
                glazingAngleFactor,
                1.0,
                edgeWeights,
                edgeFlag
            );

            vec2 pixelPosFract = fract(curr2PrevTexelPos - 0.5);
            vec2 bilinearWeights2 = pixelPosFract;
            vec4 bilinearWeights4;
            bilinearWeights4.yz = bilinearWeights2.xx;
            bilinearWeights4.xw = 1.0 - bilinearWeights2.xx;
            bilinearWeights4.xy *= bilinearWeights2.yy;
            bilinearWeights4.zw *= 1.0 - bilinearWeights2.yy;

            vec4 finalWeights = edgeWeights * bilinearWeights4;
            float weightSum = dot(finalWeights, vec4(1.0));
            float rcpWeightSum = safeRcp(weightSum);
            finalWeights *= rcpWeightSum;

            float historyResetFactor = 1.0;
            if (edgeFlag) {
                bool validFlag = weightSum > 0.001;
                historyResetFactor *= float(validFlag);
                // Only applying this to reprojInfo (which is used in ReSTIR temporal reuse)
                // Because killing denoiser history causes flickering bs
                // Seems unused
                // reprojInfo.historyResetFactor = weightSum;
                if (weightSum > 0.001) {
                    vec4 data5X = history_gi5_gatherTexel(gatherTexelPos, 0);
                    vec4 data5Y = history_gi5_gatherTexel(gatherTexelPos, 1);
                    vec4 data5Z = history_gi5_gatherTexel(gatherTexelPos, 2);
                    vec4 data5W = history_gi5_gatherTexel(gatherTexelPos, 3);

                    vec4 packedData5 = bileratralSum(
                        data5X,
                        data5Y,
                        data5Z,
                        data5W,
                        finalWeights
                    );
                    float antiStretching = pow2(linearStep(0.2, 0.0, pow2(packedData5.w) - pow2(glazingAngleFactorHistory)));
                    historyResetFactor *= antiStretching;
                    packedData5.x *= historyResetFactor;
                    packedData5.y *= antiStretching;
                    packedData5.w = glazingAngleFactorHistory;
                    packedData5 = saturate(packedData5);
                    transient_gi5Reprojected_store(texelPos, packedData5);

                    float ditherNoise = rand_stbnVec1(rand_newStbnPos(texelPos, 8u), frameCounter);

                    vec4 data1X = history_gi1_gatherTexel(gatherTexelPos, 0);
                    vec4 data1Y = history_gi1_gatherTexel(gatherTexelPos, 1);
                    vec4 data1Z = history_gi1_gatherTexel(gatherTexelPos, 2);
                    vec4 data1W = history_gi1_gatherTexel(gatherTexelPos, 3);

                    vec4 data2X = history_gi2_gatherTexel(gatherTexelPos, 0);
                    vec4 data2Y = history_gi2_gatherTexel(gatherTexelPos, 1);
                    vec4 data2Z = history_gi2_gatherTexel(gatherTexelPos, 2);
                    vec4 data2W = history_gi2_gatherTexel(gatherTexelPos, 3);

                    vec4 data3X = history_gi3_gatherTexel(gatherTexelPos, 0);
                    vec4 data3Y = history_gi3_gatherTexel(gatherTexelPos, 1);
                    vec4 data3Z = history_gi3_gatherTexel(gatherTexelPos, 2);
                    vec4 data3W = history_gi3_gatherTexel(gatherTexelPos, 3);

                    vec4 data4X = history_gi4_gatherTexel(gatherTexelPos, 0);
                    vec4 data4Y = history_gi4_gatherTexel(gatherTexelPos, 1);
                    vec4 data4Z = history_gi4_gatherTexel(gatherTexelPos, 2);
                    vec4 data4W = history_gi4_gatherTexel(gatherTexelPos, 3);

                    vec4 packedData1 = bileratralSum(
                        data1X,
                        data1Y,
                        data1Z,
                        data1W,
                        finalWeights
                    );
                    packedData1 = clamp(packedData1, 0.0, FP16_MAX);
                    specularHitDistance = packedData1.w;
                    packedData1 = dither_fp16(packedData1, ditherNoise);
                    transient_gi1Reprojected_store(texelPos, packedData1);

                    vec4 packedData2 = bileratralSum(
                        data2X,
                        data2Y,
                        data2Z,
                        data2W,
                        finalWeights
                    );
                    packedData2 = clamp(packedData2, 0.0, FP16_MAX);
                    packedData2 = dither_fp16(packedData2, ditherNoise);
                    transient_gi2Reprojected_store(texelPos, packedData2);

                    vec4 packedData3 = bileratralSum(
                        data3X,
                        data3Y,
                        data3Z,
                        data3W,
                        finalWeights
                    );
                    packedData3 = clamp(packedData3, 0.0, FP16_MAX);
                    packedData3 = dither_fp16(packedData3, ditherNoise);
                    transient_gi3Reprojected_store(texelPos, packedData3);

                    vec4 packedData4 = bileratralSum(
                        data4X,
                        data4Y,
                        data4Z,
                        data4W,
                        finalWeights
                    );
                    packedData4 = clamp(packedData4, 0.0, FP16_MAX);
                    packedData4 = dither_fp16(packedData4, ditherNoise);
                    transient_gi4Reprojected_store(texelPos, packedData4);

                    valid = true;
                }
            } else {
                vec4 packedData5 = history_gi5_sample(curr2PrevScreen);

                float antiStretching = pow2(linearStep(0.2, 0.0, pow2(packedData5.w) - pow2(glazingAngleFactorHistory)));
                historyResetFactor *= antiStretching;

                packedData5.x *= historyResetFactor;
                packedData5.y *= antiStretching;
                packedData5.w = glazingAngleFactorHistory;

                packedData5 = saturate(packedData5);
                transient_gi5Reprojected_store(texelPos, packedData5);

                float ditherNoise = rand_stbnVec1(rand_newStbnPos(texelPos, 8u), frameCounter);

                CatmullRomBicubic5TapData tapData = sampling_catmullRomBicubic5Tap_init(curr2PrevTexelPos, 0.5, uval_mainImageSizeRcp);
                vec4 packData11 = history_gi1_sample(tapData.uv1AndWeight.xy);
                vec4 packData12 = history_gi1_sample(tapData.uv2AndWeight.xy);
                vec4 packData13 = history_gi1_sample(tapData.uv3AndWeight.xy);
                vec4 packData14 = history_gi1_sample(tapData.uv4AndWeight.xy);
                vec4 packData15 = history_gi1_sample(tapData.uv5AndWeight.xy);

                vec4 packData31 = history_gi3_sample(tapData.uv1AndWeight.xy);
                vec4 packData32 = history_gi3_sample(tapData.uv2AndWeight.xy);
                vec4 packData33 = history_gi3_sample(tapData.uv3AndWeight.xy);
                vec4 packData34 = history_gi3_sample(tapData.uv4AndWeight.xy);
                vec4 packData35 = history_gi3_sample(tapData.uv5AndWeight.xy);

                vec4 packedData1 = sampling_catmullBicubic5Tap_sum(
                    packData11,
                    packData12,
                    packData13,
                    packData14,
                    packData15,
                    tapData
                );
                packedData1 = clamp(packedData1, 0.0, FP16_MAX);
                specularHitDistance = packedData1.w;
                packedData1 = dither_fp16(packedData1, ditherNoise);
                transient_gi1Reprojected_store(texelPos, packedData1);

                vec4 packedData2 = history_gi2_sample(curr2PrevScreen);
                packedData2 = clamp(packedData2, 0.0, FP16_MAX);
                packedData2 = dither_fp16(packedData2, ditherNoise);
                transient_gi2Reprojected_store(texelPos, packedData2);

                vec4 packedData3 = sampling_catmullBicubic5Tap_sum(
                    packData31,
                    packData32,
                    packData33,
                    packData34,
                    packData35,
                    tapData
                );
                packedData3 = clamp(packedData3, 0.0, FP16_MAX);
                packedData3 = dither_fp16(packedData3, ditherNoise);
                transient_gi3Reprojected_store(texelPos, packedData3);

                vec4 packedData4 = history_gi4_sample(curr2PrevScreen);
                packedData4 = clamp(packedData4, 0.0, FP16_MAX);
                packedData4 = dither_fp16(packedData4, ditherNoise);
                transient_gi4Reprojected_store(texelPos, packedData4);

                valid = true;
            }

            if (valid) {
                ReprojectInfo reprojInfo = reprojectInfo_init();
                // Most edge values are very close to 1.0
                // And we also want stricter weights for ReSTIR temporal
                reprojInfo.bilateralWeights = pow(edgeWeights * pow(extraNormalWeights, vec4(128.0)), vec4(16.0));
                reprojInfo.curr2PrevScreenPos = curr2PrevScreen;
                reprojInfo.historyResetFactor = historyResetFactor;
                transient_gi_diffuse_reprojInfo_store(texelPos, reprojectInfo_pack(reprojInfo));
            }
        }
    }

    if (!valid) {
        transient_gi1Reprojected_store(texelPos, vec4(0.0, 0.0, 0.0, 16.0));
        transient_gi2Reprojected_store(texelPos, vec4(0.0, 0.0, 0.0, 16.0));
        transient_gi5Reprojected_store(texelPos, vec4(0.0));

        ReprojectInfo reprojInfo = reprojectInfo_init();
        transient_gi_diffuse_reprojInfo_store(texelPos, reprojectInfo_pack(reprojInfo));
    }

    // Virtual-point specular reprojection
    // Surface point is already done along with diffuse reprojection above.
    // But we also want to try to reproject using virtual point and blend the result based on roughness
    // This is to handle low roughness surface where the specular is more view-dependent
    {
        bool specValid = valid;

        Material material = material_decode(gData);
        // Goes to 1.0 when roughness is 0.0 and vise-versa
        float mirrorParallaxFactor = pow4(saturate(1.0 - material.roughness));

        vec3 viewDir = normalize(currViewPos);
        vec3 virtualViewPos = currViewPos + viewDir * specularHitDistance * mirrorParallaxFactor;

        vec4 virtualPrevViewPos = coord_viewCurrToPrev(vec4(virtualViewPos, 1.0), gData.isHand);
        vec4 virtualPrevClipPos = global_prevCamProj * virtualPrevViewPos;
        uint clipFlag = uint(curr2PrevClipPos.z > 0.0);
        clipFlag &= uint(all(lessThan(abs(curr2PrevClipPos.xy), curr2PrevClipPos.ww)));

        if (bool(clipFlag)) {
            vec2 virtualPrevNDC = virtualPrevClipPos.xy / virtualPrevClipPos.w;
            vec2 virtualPrevScreen = virtualPrevNDC * 0.5 + 0.5;
            vec2 virtualPrevScreenClamped = saturate(virtualPrevScreen);

            if (all(lessThan(abs(virtualPrevScreen - virtualPrevScreenClamped), uval_mainImageSizeRcp * 2.0))) {
                virtualPrevScreen += uval_prevTaaJitter * uval_mainImageSizeRcp;
                vec2 virtualPrevTexelPos = virtualPrevScreen * uval_mainImageSize;
                virtualPrevTexelPos = clamp(virtualPrevTexelPos, vec2(1.0), uval_mainImageSize - 1.0);

                vec2 gatherTexelPos = floor(virtualPrevTexelPos - 0.5) + 1.0;

                bool edgeFlag;
                vec4 edgeWeights;
                computeEdgeWeights(
                    screenPos,
                    gatherTexelPos,
                    currViewNormal,
                    currViewGeomNormal,
                    curr2PrevViewPos.xyz,
                    glazingAngleFactor,
                    128.0 * mirrorParallaxFactor + 128.0,
                    edgeWeights,
                    edgeFlag
                );

                vec2 pixelPosFract = fract(virtualPrevTexelPos - 0.5);
                vec2 bilinearWeights2 = pixelPosFract;
                vec4 blinearWeights4;
                blinearWeights4.yz = bilinearWeights2.xx;
                blinearWeights4.xw = 1.0 - bilinearWeights2.xx;
                blinearWeights4.xy *= bilinearWeights2.yy;
                blinearWeights4.zw *= 1.0 - bilinearWeights2.yy;

                vec4 finalWeights = edgeWeights * blinearWeights4;
                float weightSum = dot(finalWeights, vec4(1.0));
                float rcpWeightSum = safeRcp(weightSum);
                finalWeights *= rcpWeightSum;

                float ditherNoiseV = rand_stbnVec1(rand_newStbnPos(texelPos, 9u), frameCounter);

                if (edgeFlag) {
                    if (weightSum > 0.001) {
                        vec4 data3X = history_gi3_gatherTexel(gatherTexelPos, 0);
                        vec4 data3Y = history_gi3_gatherTexel(gatherTexelPos, 1);
                        vec4 data3Z = history_gi3_gatherTexel(gatherTexelPos, 2);
                        vec4 data3W = history_gi3_gatherTexel(gatherTexelPos, 3);

                        vec4 data4X = history_gi4_gatherTexel(gatherTexelPos, 0);
                        vec4 data4Y = history_gi4_gatherTexel(gatherTexelPos, 1);
                        vec4 data4Z = history_gi4_gatherTexel(gatherTexelPos, 2);
                        vec4 data4W = history_gi4_gatherTexel(gatherTexelPos, 3);

                        vec4 packedData3 = bileratralSum(data3X, data3Y, data3Z, data3W, finalWeights);
                        vec4 packedData4 = bileratralSum(data4X, data4Y, data4Z, data4W, finalWeights);
                        packedData3 = clamp(packedData3, 0.0, FP16_MAX);
                        packedData3 = dither_fp16(packedData3, ditherNoiseV);
                        transient_gi3Reprojected_store(texelPos, packedData3);

                        packedData4 = clamp(packedData4, 0.0, FP16_MAX);
                        packedData4 = dither_fp16(packedData4, ditherNoiseV);
                        transient_gi4Reprojected_store(texelPos, packedData4);

                        specValid = true;
                    }
                } else {
                    CatmullRomBicubic5TapData vTapData = sampling_catmullRomBicubic5Tap_init(virtualPrevTexelPos, 0.5, uval_mainImageSizeRcp);

                    vec4 specData3 = sampling_catmullBicubic5Tap_sum(
                        history_gi3_sample(vTapData.uv1AndWeight.xy),
                        history_gi3_sample(vTapData.uv2AndWeight.xy),
                        history_gi3_sample(vTapData.uv3AndWeight.xy),
                        history_gi3_sample(vTapData.uv4AndWeight.xy),
                        history_gi3_sample(vTapData.uv5AndWeight.xy),
                        vTapData
                    );
                    vec4 specData4 = sampling_catmullBicubic5Tap_sum(
                        history_gi4_sample(vTapData.uv1AndWeight.xy),
                        history_gi4_sample(vTapData.uv2AndWeight.xy),
                        history_gi4_sample(vTapData.uv3AndWeight.xy),
                        history_gi4_sample(vTapData.uv4AndWeight.xy),
                        history_gi4_sample(vTapData.uv5AndWeight.xy),
                        vTapData
                    );

                    specData3 = clamp(specData3, 0.0, FP16_MAX);
                    specData4 = clamp(specData4, 0.0, FP16_MAX);
                    specData3 = dither_fp16(specData3, ditherNoiseV);
                    specData4 = dither_fp16(specData4, ditherNoiseV);

                    transient_gi3Reprojected_store(texelPos, specData3);
                    transient_gi4Reprojected_store(texelPos, specData4);

                    specValid = true;
                }
            }
        }

        if (!specValid) {
            transient_gi3Reprojected_store(texelPos, vec4(0.0));
            transient_gi4Reprojected_store(texelPos, vec4(0.0));
        }
    }
}
