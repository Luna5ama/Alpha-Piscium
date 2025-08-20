#version 460 compatibility

#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/atmospherics/LocalComposite.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) restrict uniform image2D uimg_main;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);

        vec3 albedo = colors_sRGB_decodeGamma(texelFetch(usam_gbufferData8UN, texelPos, 0).rgb);
        vec3 giRadiance = texelFetch(usam_temp2, texelPos, 0).rgb;
        outputColor.rgb += giRadiance.rgb * albedo;
        ScatteringResult sctrResult = atmospherics_localComposite(texelPos);
        outputColor.rgb = scatteringResult_apply(sctrResult, outputColor.rgb);

        imageStore(uimg_main, texelPos, outputColor);
    }
}