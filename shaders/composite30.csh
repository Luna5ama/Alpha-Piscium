#version 460 compatibility

#include "util/FullScreenComp.glsl"
#include "atmosphere/UnwrapEpipolar.comp"

uniform usampler2D usam_gbufferData;
uniform sampler2D usam_ssvbil;
uniform sampler2D usam_translucentColor;

layout(rgba16f) restrict uniform image2D uimg_main;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec3 inScattering;
        vec3 transmittance;
        vec2 texCoord = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        UnwarpEpipolarInsctrImage(texCoord * 2.0 - 1.0, viewZ, inScattering, transmittance);

        vec4 outputColor = imageLoad(uimg_main, texelPos);

        GBufferData gData;
        gbuffer_unpack(texelFetch(usam_gbufferData, texelPos, 0), gData);
        Material material = material_decode(gData);

        vec4 ssvbilSample = texelFetch(usam_ssvbil, texelPos, 0);
        vec3 indirectV = ssvbilSample.rgb * material.albedo;

        float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));
        outputColor.rgb *= mix(sqrt(ssvbilSample.a), 1.0, shadowIsSun);
        outputColor.rgb += indirectV;

        outputColor.rgb *= transmittance;
        vec3 sunRadiance = global_sunRadiance.rgb * global_sunRadiance.a;
        sunRadiance *= mix(MOON_RADIANCE_MUL, vec3(1.0), shadowIsSun);
        outputColor.rgb += sunRadiance * inScattering;

        vec4 translucentColorSample = texelFetch(usam_translucentColor, texelPos, 0);
        float luminanceC = colors_srgbLuma(outputColor.rgb) * 4.0;
        float luminanceT = max(colors_srgbLuma(translucentColorSample.rgb), 1.0);
        outputColor.rgb = mix(outputColor.rgb, translucentColorSample.rgb * (luminanceC / luminanceT), translucentColorSample.a);

        imageStore(uimg_main, texelPos, outputColor);
    }
}