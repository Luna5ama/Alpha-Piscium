#include "/techniques/gi/Common.glsl"
#include "/techniques/gi/Reproject.glsl"
#include "/util/GBufferData.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_temp1;
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgba8) uniform writeonly image2D uimg_rgba8;

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

        const float HISTORY_LENGTH = 64.0;
        const float HISTORY_LENGTH_1 = HISTORY_LENGTH - 1.0;
        float historyLength = historyData.historyLength * HISTORY_LENGTH_1;
        historyLength = clamp(historyLength + 1.0, 1.0, HISTORY_LENGTH);

        // Accumulate
        float alpha = 1.0 / historyLength;
        historyData.diffuseColor = mix(historyData.diffuseColor, newDiffuse.rgb, alpha);
        historyData.historyLength = saturate(historyLength / HISTORY_LENGTH_1);

        history_gi1_store(texelPos, gi_historyData_pack1(historyData));
        history_gi2_store(texelPos, gi_historyData_pack2(historyData));
        history_gi3_store(texelPos, gi_historyData_pack3(historyData));
        history_gi4_store(texelPos, gi_historyData_pack4(historyData));
        history_gi5_store(texelPos, gi_historyData_pack5(historyData));
    }
}
