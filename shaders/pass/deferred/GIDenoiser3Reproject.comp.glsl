#include "/techniques/gi/Common.glsl"
#include "/techniques/gi/Reproject.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Sampling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_temp1;
layout(rgba16f) uniform writeonly image2D uimg_temp2;
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgba8) uniform writeonly image2D uimg_rgba8;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
        float currViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        GBufferData gData = gbufferData_init();
        gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
        gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

        GIHistoryData historyData = gi_historyData_init();

        vec3 currViewPos = coords_toViewCoord(screenPos, currViewZ, global_camProjInverse);
        vec4 curr2PrevViewPos = coord_viewCurrToPrev(vec4(currViewPos, 1.0), gData.isHand);
        vec4 curr2PrevClipPos = global_prevCamProj * curr2PrevViewPos;
        uint clipFlag = uint(curr2PrevClipPos.z > 0.0);
        clipFlag &= uint(all(lessThan(abs(curr2PrevClipPos.xy), curr2PrevClipPos.ww)));
        if (bool(clipFlag)) {
            vec2 curr2PrevNDC = curr2PrevClipPos.xy / curr2PrevClipPos.w;
            vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;

            if (all(equal(curr2PrevScreen, saturate(curr2PrevScreen)))) {
                vec2 curr2PrevTexelPos = curr2PrevScreen * uval_mainImageSize;
                vec3 currViewNormal = gData.normal;
                vec3 currViewGeomNormal = gData.geomNormal;

                vec2 centerPixel = curr2PrevTexelPos - 0.5;
                ivec2 gatherTexelPos = ivec2(centerPixel);
                vec2 gatherScreenPos = coords_texelToUV(gatherTexelPos, uval_mainImageSizeRcp);

                float currEdgeFactor = min4(transient_edgeMaskTemp_gather(screenPos, 0));
                float prevEdgeMask = min4(history_gi5_gather(gatherScreenPos, 1));

                vec4 viewZs = history_viewZ_gather(gatherScreenPos, 0);
                vec4 geomViewNormalXs = history_geomViewNormal_gather(gatherScreenPos, 0) * 2.0 - 1.0;
                vec4 geomViewNormalYs = history_geomViewNormal_gather(gatherScreenPos, 1) * 2.0 - 1.0;
                vec4 geomViewNormalZs = history_geomViewNormal_gather(gatherScreenPos, 2) * 2.0 - 1.0;

                vec4 viewNormalXs = history_viewNormal_gather(gatherScreenPos, 0) * 2.0 - 1.0;
                vec4 viewNormalYs = history_viewNormal_gather(gatherScreenPos, 1) * 2.0 - 1.0;
                vec4 viewNormalZs = history_viewNormal_gather(gatherScreenPos, 2) * 2.0 - 1.0;

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

                float planeDistance1 = gi_planeDistance(currViewPos, currViewGeomNormal, prevViewPos1, geomViewNormal1);
                float planeDistance2 = gi_planeDistance(currViewPos, currViewGeomNormal, prevViewPos2, geomViewNormal2);
                float planeDistance3 = gi_planeDistance(currViewPos, currViewGeomNormal, prevViewPos3, geomViewNormal3);
                float planeDistance4 = gi_planeDistance(currViewPos, currViewGeomNormal, prevViewPos4, geomViewNormal4);

                float glazingAngleFactor = sqrt(saturate(dot(currViewGeomNormal, normalize(currViewPos))));
                float geomDepthThreshold = exp2(mix(-10.0, -16.0, glazingAngleFactor) * 0.9) * pow2(currViewZ);

                float geomViewNormalDot1 = dot(currViewGeomNormal, geomViewNormal1);
                float geomViewNormalDot2 = dot(currViewGeomNormal, geomViewNormal2);
                float geomViewNormalDot3 = dot(currViewGeomNormal, geomViewNormal3);
                float geomViewNormalDot4 = dot(currViewGeomNormal, geomViewNormal4);

                vec4 geomViewNormalDots = vec4(geomViewNormalDot1, geomViewNormalDot2, geomViewNormalDot3, geomViewNormalDot4);
                vec4 planeDistances = vec4(planeDistance1, planeDistance2, planeDistance3, planeDistance4);

                uint edgeFlag = 0u;
                edgeFlag |= uint((currEdgeFactor + prevEdgeMask) < 1.99);
                edgeFlag |= uint(any(lessThan(geomViewNormalDots, vec4(0.999))));
                edgeFlag |= uint(any(greaterThan(planeDistances, vec4(geomDepthThreshold))));

                if (bool(edgeFlag)) {

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

                    gi_historyData_unpack5(historyData, history_gi5_sample(curr2PrevScreen));
                }
            }
        }

        transient_gi1Reprojected_store(texelPos, gi_historyData_pack1(historyData));
        transient_gi2Reprojected_store(texelPos, gi_historyData_pack2(historyData));
        transient_gi3Reprojected_store(texelPos, gi_historyData_pack3(historyData));
        transient_gi4Reprojected_store(texelPos, gi_historyData_pack4(historyData));
        transient_gi5Reprojected_store(texelPos, gi_historyData_pack5(historyData));
    }
}
