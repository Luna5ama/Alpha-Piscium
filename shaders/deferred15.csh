#version 460 compatibility

#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/atmospherics/SkyComposite.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) restrict uniform image2D uimg_main;
layout(rgba8) uniform restrict image2D uimg_overlays;

void main() {
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 temp6Out = vec4(0.0);
        vec4 mainOut = celestial_render(texelPos, temp6Out);

        #ifdef SETTING_CONSTELLATIONS
        vec4 prevTemp6Value = imageLoad(uimg_overlays, texelPos);
        temp6Out.rgb += temp6Out.rgb;
        temp6Out.a += temp6Out.a;
        imageStore(uimg_overlays, texelPos, temp6Out);
        #endif


        ScatteringResult sctrResult = atmospherics_skyComposite(texelPos);
        mainOut.rgb = scatteringResult_apply(sctrResult, mainOut.rgb);
        mainOut.rgb = clamp(mainOut.rgb, 0.0, FP16_MAX);
        imageStore(uimg_main, texelPos, mainOut);
    }
}