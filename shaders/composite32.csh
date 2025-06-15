#version 460 compatibility

#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Colors.glsl"
#include "/post/DOF.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp2;

uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferData8UN;
uniform sampler2D usam_translucentColor;
uniform sampler2D usam_gbufferViewZ;

layout(rgba16f) uniform restrict image2D uimg_main;
layout(rgba16f) uniform writeonly image2D uimg_temp1;
void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 translucentColorSample = texelFetch(usam_translucentColor, texelPos, 0);

        vec4 outputColor = imageLoad(uimg_main, texelPos);

        GBufferData gData = gbufferData_init();
        gbufferData2_unpack(texelFetch(usam_gbufferData8UN, texelPos, 0), gData);

        float albedoLuminance = all(equal(gData.albedo, vec3(0.0))) ? 0.1 : colors_sRGB_luma(gData.albedo);
        float luminanceC = colors_sRGB_luma(outputColor.rgb) / albedoLuminance;
        outputColor.rgb = mix(outputColor.rgb, translucentColorSample.rgb * luminanceC, translucentColorSample.a);

        imageStore(uimg_main, texelPos, outputColor);

        #ifdef SETTING_DOF
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        outputColor.a = abs(viewZ);
        imageStore(uimg_temp1, texelPos, outputColor);
        #endif
    }
}