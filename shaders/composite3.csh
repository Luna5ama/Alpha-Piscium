#version 460 compatibility

#include "/util/FullScreenComp.glsl"
#include "/atmosphere/Common.glsl"
#include "/general/Lighting.glsl"
#include "/atmosphere/SunMoon.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_temp5;
uniform usampler2D usam_gbufferData;

layout(rgba16f) uniform writeonly image2D uimg_main;
layout(rgba16f) uniform restrict image2D uimg_temp2;

void doLighting(Material material, vec3 N, inout vec3 mainOut, inout vec3 ssgiOut) {
    vec3 emissiveV = material.emissive;

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    float alpha = material.roughness * material.roughness;

    vec3 feetPlayerPos = (gbufferModelViewInverse * vec4(lighting_viewCoord, 1.0)).xyz;
    vec3 worldPos = feetPlayerPos + cameraPosition;
    float viewAltitude = atmosphere_height(atmosphere, worldPos);
    vec3 sunRadiance = global_sunRadiance.rgb * global_sunRadiance.a;

    #ifdef SETTING_SHADOW_HALF_RES
    vec3 shadow = texelFetch(usam_temp5, texelPos, 0).rgb;
    #else
    vec3 shadow = calcShadow(material.sss);
    #endif


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

    mainOut += 0.001 * material.albedo;
    mainOut += emissiveV;
    mainOut += mix(combinedLighting.diffuse, combinedLighting.diffuseLambertian, gData.isHand);
    mainOut += combinedLighting.specular;
    mainOut += combinedLighting.sss;

    ssgiOut += emissiveV * RCP_PI;
    ssgiOut += combinedLighting.diffuseLambertian;
    ssgiOut += combinedLighting.sss;
}

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 screenPos = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;

        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;

        vec4 mainOut = vec4(0.0, 0.0, 0.0, 1.0);

        if (viewZ != -65536.0) {
            gbuffer_unpack(texelFetch(usam_gbufferData, texelPos, 0), gData);
            Material material = material_decode(gData);

            lighting_init(coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse));

            vec4 ssgiOut = imageLoad(uimg_temp2, texelPos);
            ssgiOut.a = gData.lmCoord.y;

            if (gData.materialID == 65534u) {
                mainOut = vec4(material.albedo, 1.0);
                ssgiOut = vec4(0.0, 0.0, 0.0, 0.0);
            } else {
                doLighting(material, gData.normal, mainOut.rgb, ssgiOut.rgb);
            }

            imageStore(uimg_temp2, texelPos, ssgiOut);
        } else {
            mainOut.rgb += renderSunMoon(texelPos);
        }

        imageStore(uimg_main, texelPos, mainOut);
    }
}