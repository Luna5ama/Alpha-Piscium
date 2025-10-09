#extension GL_KHR_shader_subgroup_ballot : enable

#define GLOBAL_DATA_MODIFIER \

#include "/techniques/atmospherics/LocalComposite.glsl"
#include "/techniques/textile/CSRGBA16F.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Colors2.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) restrict uniform image2D uimg_main;

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_main, texelPos, 0);
        if (isEyeInWater == 1) {
            ScatteringResult sctrResult = atmospherics_localComposite(1, texelPos);
            outputColor.rgb = scatteringResult_apply(sctrResult, outputColor.rgb);
        }
        ScatteringResult sctrResult = atmospherics_localComposite(2, texelPos);
        outputColor.rgb = scatteringResult_apply(sctrResult, outputColor.rgb);
        imageStore(uimg_main, texelPos, outputColor);
    }
}