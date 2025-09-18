#version 460 compatibility

#extension GL_KHR_shader_subgroup_ballot : enable

#define GLOBAL_DATA_MODIFIER \

#include "/techniques/atmospherics/LocalComposite.glsl"
#include "/techniques/textile/CSRGBA16F.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Colors2.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) restrict uniform image2D uimg_temp1;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = texelFetch(usam_main, texelPos, 0);

        vec3 albedo = colors2_material_idt(texelFetch(usam_temp5, texelPos, 0).rgb);

        vec3 giRadiance = texelFetch(usam_csrgba16f, csrgba16f_temp1_texelToTexel(texelPos), 0).rgb;
        outputColor.rgb += giRadiance.rgb * albedo;
        ScatteringResult sctrResult = atmospherics_localComposite(texelPos);
        outputColor.rgb = scatteringResult_apply(sctrResult, outputColor.rgb);

        imageStore(uimg_temp1, texelPos, outputColor);
    }
}