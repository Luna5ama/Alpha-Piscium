#version 460 compatibility

#include "/atmosphere/UnwarpEpipolar.glsl"
#include "/atmosphere/Scattering.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform usampler2D usam_gbufferData;
uniform sampler2D usam_ssvbil;
uniform sampler2D usam_translucentColor;

layout(rgba16f) restrict uniform image2D uimg_main;

void applyAtmosphere(inout vec4 outputColor) {
    vec2 texCoord = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    ScatteringResult sctrResult;

    #ifndef SETTING_DEPTH_BREAK_CORRECTION
    unwarpEpipolarInsctrImage(texCoord * 2.0 - 1.0, viewZ, sctrResult);
    #else
    if (!unwarpEpipolarInsctrImage(texCoord * 2.0 - 1.0, viewZ, sctrResult)) {
        float ignValue = rand_IGN(texelPos, frameCounter);
        AtmosphereParameters atmosphere = getAtmosphereParameters();
        vec3 viewCoord = coords_toViewCoord(texCoord, viewZ, gbufferProjectionInverse);
        sctrResult = computeSingleScattering(atmosphere, vec3(0.0), viewCoord, ignValue);
    }
    #endif

    outputColor.rgb *= sctrResult.transmittance;
    outputColor.rgb += sctrResult.inScattering;
}

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);

        GBufferData gData;
        gbuffer_unpack(texelFetch(usam_gbufferData, texelPos, 0), gData);
        Material material = material_decode(gData);

        vec4 ssvbilSample = texelFetch(usam_ssvbil, texelPos, 0);
        vec3 indirectV = ssvbilSample.rgb * material.albedo;

        outputColor.rgb += indirectV;

        applyAtmosphere(outputColor);

        float albedoLuminance = all(equal(gData.albedo, vec3(0.0))) ? 0.1 : colors_srgbLuma(material.albedo);
        float luminanceC = colors_srgbLuma(outputColor.rgb) / albedoLuminance;
        vec4 translucentColorSample = texelFetch(usam_translucentColor, texelPos, 0);
        outputColor.rgb = mix(outputColor.rgb, translucentColorSample.rgb * luminanceC, translucentColorSample.a);

        imageStore(uimg_main, texelPos, outputColor);
    }
}