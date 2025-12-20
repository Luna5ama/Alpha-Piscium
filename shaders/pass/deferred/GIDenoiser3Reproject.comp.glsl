#include "/techniques/gi/Reproject.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgba8) uniform writeonly image2D uimg_rgba8;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        float currViewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        GBufferData gData = gbufferData_init();
        gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
        gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);

        GIHistoryData historyData = gi_reproject(texelPos, currViewZ, gData);

        transient_gi1Reprojected_store(texelPos, gi_historyData_pack1(historyData));
        transient_gi2Reprojected_store(texelPos, gi_historyData_pack2(historyData));
        transient_gi3Reprojected_store(texelPos, gi_historyData_pack3(historyData));
        transient_gi4Reprojected_store(texelPos, gi_historyData_pack4(historyData));
        transient_gi5Reprojected_store(texelPos, gi_historyData_pack5(historyData));
    }
}
