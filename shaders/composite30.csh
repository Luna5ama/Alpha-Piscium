#version 460 compatibility

#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/atmospherics/LocalComposite.glsl"
#include "/techniques/textile/CSRGBA16F.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) restrict uniform image2D uimg_main;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);

        GBufferData gData = gbufferData_init();
        gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
        Material material = material_decode(gData);

        vec3 giRadiance = texelFetch(usam_csrgba16f, csrgba16f_temp1_texelToTexel(texelPos), 0).rgb;
        outputColor.rgb += giRadiance.rgb * material.albedo;
        ScatteringResult sctrResult = atmospherics_localComposite(texelPos);
        outputColor.rgb = scatteringResult_apply(sctrResult, outputColor.rgb);

        imageStore(uimg_main, texelPos, outputColor);
    }
}