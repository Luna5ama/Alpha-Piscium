#include "/techniques/gi/Common.glsl"
#include "/techniques/textile/CSRGBA32UI.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict writeonly image2D uimg_temp1;
layout(rgba16f) uniform restrict writeonly image2D uimg_temp2;
layout(rgba16f) uniform restrict writeonly image2D uimg_temp3;
layout(rgba16f) uniform restrict writeonly image2D uimg_temp4;
layout(rgba8) uniform restrict writeonly image2D uimg_temp5;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        GIHistoryData historyData = gi_historyData_init();
        gi_historyData_unpack1(historyData, texelFetch(usam_csrgba32ui, gi_history1_texelToTexel(texelPos), 0));
        gi_historyData_unpack2(historyData, texelFetch(usam_csrgba32ui, gi_history2_texelToTexel(texelPos), 0));

        imageStore(uimg_temp1, texelPos, vec4(historyData.diffuseColor, historyData.diffuseMoments));
        imageStore(uimg_temp2, texelPos, vec4(historyData.diffuseFastColor, historyData.shadow));
        imageStore(uimg_temp3, texelPos, vec4(historyData.specularColor, historyData.specularMoments));
        imageStore(uimg_temp4, texelPos, vec4(historyData.specularFastColor, 0.0));
        imageStore(uimg_temp5, texelPos, vec4(historyData.historyLength, historyData.fastHistoryLength, 0.0, 0.0));
    }
}
