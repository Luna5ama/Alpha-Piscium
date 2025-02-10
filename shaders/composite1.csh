#version 460 compatibility

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "util/FullScreenComp.glsl"

#include "_Util.glsl"
#include "atmosphere/Common.glsl"
#include "general/Lighting.glsl"
#include "general/NDPacking.glsl"
#include "atmosphere/SunMoon.glsl"
#include "svgf/Reproject.glsl"

uniform usampler2D usam_gbufferData;
uniform sampler2D usam_ssvbil;
uniform usampler2D usam_prevNZ;
uniform sampler2D usam_svgfHistoryColor;
uniform sampler2D usam_svgfHistoryMoments;

layout(r32f) uniform readonly image2D uimg_gbufferViewZ;
layout(rg8) uniform writeonly image2D uimg_projReject;
layout(rgba16f) uniform writeonly image2D uimg_main;
layout(rgba16f) uniform writeonly image2D uimg_temp1;
layout(rgba16f) uniform writeonly image2D uimg_temp2;
layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba16f) uniform writeonly image2D uimg_temp4;

void doLighting(Material material, vec3 N, inout vec3 mainOut, inout vec3 ssgiOut) {
    vec3 emissiveV = material.emissive;

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    float alpha = material.roughness * material.roughness;

    vec3 feetPlayerPos = (gbufferModelViewInverse * vec4(lighting_viewCoord, 1.0)).xyz;
    vec3 worldPos = feetPlayerPos + cameraPosition;
    float viewAltitude = atmosphere_height(atmosphere, worldPos);
    vec3 sunRadiance = global_sunRadiance.rgb * global_sunRadiance.a;

    vec3 shadow = calcShadow(material.sss);

    float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));

    float cosSunZenith = dot(uval_sunDirWorld, vec3(0.0, 1.0, 0.0));
    vec3 tSun = sampleTransmittanceLUT(atmosphere, cosSunZenith, viewAltitude);
    vec3 sunShadow = mix(vec3(1.0), shadow, shadowIsSun);
    vec4 sunIrradiance = vec4(sunRadiance * tSun * sunShadow, colors_srgbLuma(sunShadow));
    LightingResult sunLighting = directLighting(material, sunIrradiance, uval_sunDirView, N);

    float cosMoonZenith = dot(uval_moonDirWorld, vec3(0.0, 1.0, 0.0));
    vec3 tMoon = sampleTransmittanceLUT(atmosphere, cosMoonZenith, viewAltitude);
    vec3 moonShadow = mix(shadow, vec3(1.0), shadowIsSun);
    vec4 moonIrradiance = vec4(sunRadiance * MOON_RADIANCE_MUL * tMoon * moonShadow, colors_srgbLuma(moonShadow));
    LightingResult moonLighting = directLighting(material, moonIrradiance, uval_moonDirView, N);

    LightingResult combinedLighting = lightingResult_add(sunLighting, moonLighting);

    vec3 skyReflectionV = skyReflection(material, gData.lmCoord.y, N);

    mainOut += 0.001 * material.albedo;
    mainOut += emissiveV;
    mainOut += mix(combinedLighting.diffuse, combinedLighting.diffuseLambertian, gData.isHand);
    mainOut += combinedLighting.specular;
    mainOut += combinedLighting.sss;
    mainOut += skyReflectionV;

    ssgiOut += emissiveV;
    ssgiOut += combinedLighting.diffuseLambertian;
    ssgiOut += combinedLighting.sss;
}

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;

        float viewZ = imageLoad(uimg_gbufferViewZ, texelPos).r;
        gbuffer_unpack(texelFetch(usam_gbufferData, texelPos, 0), gData);

        lighting_init(coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse));

        vec2 projRejectOut;
        ndpacking_updateProjReject(usam_prevNZ, texelPos, screenPos, gData.normal, lighting_viewCoord, projRejectOut);
        imageStore(uimg_projReject, texelPos, vec4(projRejectOut, 0.0, 0.0));

        vec4 prevColorHLen;
        vec2 prevMoments;

        svgf_reproject(
            usam_svgfHistoryColor, usam_svgfHistoryMoments, usam_prevNZ,
            screenPos, viewZ, gData.normal, float(gData.isHand),
            prevColorHLen, prevMoments
        );

        imageStore(uimg_temp3, texelPos, prevColorHLen);
        imageStore(uimg_temp4, texelPos, vec4(prevMoments, 0.0, 0.0));

        vec4 mainOut = vec4(0.0, 0.0, 0.0, 1.0);
        vec4 temp1Out = vec4(0.0);
        vec4 ssgiOut = vec4(0.0);

        if (viewZ != -65536.0) {
            Material material = material_decode(gData);

            temp1Out.rgb = gData.normal;
            temp1Out.a = float(any(greaterThan(material.emissive, vec3(0.0))));
            ssgiOut.a = gData.lmCoord.y;

            if (gData.materialID == 65534u) {
                mainOut = vec4(material.albedo, 1.0);
                ssgiOut = vec4(0.0, 0.0, 0.0, 0.0);
            } else {
                float multiBounceV = (SETTING_SSVBIL_GI_MB / SETTING_SSVBIL_GI_STRENGTH) * 2.0 * RCP_PI;
                ssgiOut.rgb = multiBounceV * max(prevColorHLen.rgb, 0.0) * material.albedo;
                doLighting(material, gData.normal, mainOut.rgb, ssgiOut.rgb);
            }
        } else {
            mainOut.rgb += renderSunMoon(texelPos);
        }

        imageStore(uimg_main, texelPos, mainOut);
        imageStore(uimg_temp1, texelPos, temp1Out);
        imageStore(uimg_temp2, texelPos, ssgiOut);
    }
}