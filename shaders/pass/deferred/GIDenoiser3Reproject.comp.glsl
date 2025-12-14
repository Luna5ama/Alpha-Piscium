#include "/techniques/gi/Common.glsl"
#include "/techniques/gi/Reproject.glsl"
#include "/util/GBufferData.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_temp1;
layout(rgba16f) uniform writeonly restrict image2D uimg_rgba16f;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
        float currViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        GBufferData gData = gbufferData_init();
        gbufferData1_unpack_world(texelFetch(usam_gbufferData1, texelPos, 0), gData);
        gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

        if (false) {
            gi_reproject(screenPos, currViewZ, gData.normal, gData.geomNormal, gData.isHand);
        }

        GIHistoryData historyData = gi_historyData_init();

        vec3 currViewPos = coords_toViewCoord(screenPos, currViewZ, global_camProjInverse);
        vec4 currScenePos = gbufferModelViewInverse * vec4(currViewPos, 1.0);
        vec4 curr2PrevViewPos = coord_viewCurrToPrev(vec4(currViewPos, 1.0), gData.isHand);
        vec4 curr2PrevClipPos = global_prevCamProj * curr2PrevViewPos;
        uint clipFlag = uint(curr2PrevClipPos.z > 0.0);
        clipFlag &= uint(all(lessThan(abs(curr2PrevClipPos.xy), curr2PrevClipPos.ww)));
        if (bool(clipFlag)) {
            vec2 curr2PrevNDC = curr2PrevClipPos.xy / curr2PrevClipPos.w;
            vec2 curr2PrevScreen = curr2PrevNDC * 0.5 + 0.5;

            if (all(equal(curr2PrevScreen, saturate(curr2PrevScreen)))) {
                vec2 curr2PrevTexelPos = curr2PrevScreen * uval_mainImageSize;
                vec3 currWorldNormal = gData.normal;

                vec3 currWorldGeomNormal = gData.geomNormal;

                vec2 centerPixel = curr2PrevTexelPos - 0.5;
                vec2 centerPixelOrigin = floor(centerPixel);
                vec2 gatherTexelPos = centerPixelOrigin + 1.0;
                vec2 pixelPosFract = centerPixel - centerPixelOrigin;

                vec3 curr2PrevView = curr2PrevViewPos.xyz;

                gi_historyData_unpack1(historyData, history_gi1_sample(curr2PrevScreen));
                gi_historyData_unpack2(historyData, history_gi2_sample(curr2PrevScreen));
                gi_historyData_unpack3(historyData, history_gi3_sample(curr2PrevScreen));
                gi_historyData_unpack4(historyData, history_gi4_sample(curr2PrevScreen));
                gi_historyData_unpack5(historyData, history_gi5_sample(curr2PrevScreen));
            }
        }

        imageStore(uimg_temp1, texelPos, gi_historyData_pack1(historyData));

//        transient_gi1Reprojected_store(texelPos, gi_historyData_pack1(historyData));
//        transient_gi2Reprojected_store(texelPos, gi_historyData_pack2(historyData));
//        transient_gi3Reprojected_store(texelPos, gi_historyData_pack3(historyData));
//        transient_gi4Reprojected_store(texelPos, gi_historyData_pack4(historyData));
//        transient_gi5Reprojected_store(texelPos, gi_historyData_pack5(historyData));

    }
}
