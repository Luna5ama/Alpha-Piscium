#version 460 compatibility

#include "/atmosphere/UnwarpEpipolar.glsl"
#include "/atmosphere/Scattering.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_temp2;

uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferData8UN;
uniform sampler2D usam_translucentColor;

layout(rgba16f) restrict uniform image2D uimg_main;

void applyAtmosphere(vec2 screenPos, vec3 viewPos, float viewZ, inout vec4 outputColor) {
    ScatteringResult sctrResult;

    #ifndef SETTING_DEPTH_BREAK_CORRECTION
    unwarpEpipolarInsctrImage(screenPos * 2.0 - 1.0, viewZ, sctrResult);
    #else
    if (!unwarpEpipolarInsctrImage(screenPos * 2.0 - 1.0, viewZ, sctrResult)) {
        float ignValue = rand_IGN(texelPos, frameCounter);
        AtmosphereParameters atmosphere = getAtmosphereParameters();
        sctrResult = computeSingleScattering(atmosphere, vec3(0.0), viewPos, ignValue);
    }
    #endif

    outputColor.rgb *= sctrResult.transmittance;
    outputColor.rgb += sctrResult.inScattering;
}

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);

        GBufferData gData;
        gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos, 0), gData);
        gbufferData2_unpack(texelFetch(usam_gbufferData8UN, texelPos, 0), gData);
        Material material = material_decode(gData);

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        vec3 viewPos = coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse);

        vec3 giRadiance = texelFetch(usam_temp2, texelPos, 0).rgb;

        outputColor.rgb += giRadiance.rgb * material.albedo;
        applyAtmosphere(screenPos, viewPos, viewZ, outputColor);

        float albedoLuminance = all(equal(gData.albedo, vec3(0.0))) ? 0.1 : colors_srgbLuma(material.albedo);
        float luminanceC = colors_srgbLuma(outputColor.rgb) / albedoLuminance;
        vec4 translucentColorSample = texelFetch(usam_translucentColor, texelPos, 0);
        outputColor.rgb = mix(outputColor.rgb, translucentColorSample.rgb * luminanceC, translucentColorSample.a);

        imageStore(uimg_main, texelPos, outputColor);
    }
}