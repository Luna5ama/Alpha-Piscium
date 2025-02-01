#version 460 compatibility

#include "_Util.glsl"
#include "atmosphere/Common.glsl"
#include "general/Lighting.glsl"
#include "general/NDPacking.glsl"
#include "atmosphere/SunMoon.comp"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform usampler2D usam_gbufferData;
uniform sampler2D usam_ssvbil;
uniform usampler2D usam_lastNZ;

layout(r32f) uniform readonly image2D uimg_gbufferViewZ;
layout(rg8) uniform writeonly image2D uimg_projReject;
layout(rgba16f) uniform writeonly image2D uimg_main;
layout(rgba16f) uniform writeonly image2D uimg_temp1;
layout(rgba16f) uniform writeonly image2D uimg_temp2;

void doLighting(Material material, vec3 N, vec3 V, out vec3 mainOut, out vec3 ssgiOut) {
    float NDotV = dot(N, V);

    vec3 emissiveV = material.emissive;

    vec4 ssvbilSample = texelFetch(usam_ssvbil, texelPos, 0);
    vec3 multiBounceV = (SETTING_SSVBIL_GI_MB / SETTING_SSVBIL_GI_STRENGTH) * 2.0 * RCP_PI * max(ssvbilSample.rgb, 0.0) * material.albedo;

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    vec3 fresnel = calcFresnel(material, saturate(NDotV));
    float alpha = material.roughness * material.roughness;

    vec3 feetPlayerPos = (gbufferModelViewInverse * vec4(g_viewCoord, 1.0)).xyz;
    vec3 worldPos = feetPlayerPos + cameraPosition;
    float viewAltitude = atmosphere_height(atmosphere, worldPos);
    vec3 sunRadiance = global_sunRadiance.rgb * global_sunRadiance.a;

    float cosSunZenith = dot(uval_sunDirView, uval_upDirView);
    vec3 tSun = sampleTransmittanceLUT(atmosphere, cosSunZenith, viewAltitude, usam_transmittanceLUT);
    vec3 sunIrradiance = sunRadiance * tSun;

    float cosMoonZenith = dot(uval_moonDirView, uval_upDirView);
    vec3 tMoon = sampleTransmittanceLUT(atmosphere, cosMoonZenith, viewAltitude, usam_transmittanceLUT);
    vec3 moonIrradiance = sunRadiance * MOON_RADIANCE_MUL * tMoon;

    vec3 shadow = calcShadow(material.sss);

    float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));
    vec3 sunShadow = mix(vec3(1.0), shadow, shadowIsSun);
    vec3 sunLighting = directLighting(material, sunShadow, sunIrradiance, uval_sunDirView, N, V, fresnel);

    vec3 moonShadow = mix(shadow, vec3(1.0), shadowIsSun);
    vec3 moonLighting = directLighting(material, moonShadow, moonIrradiance, uval_moonDirView, N, V, fresnel);

    // Sky reflection
    //    vec3 reflectDirView = normalize(H + reflect(-V, gData.normal));
    //    vec3 reflectDir = normalize(mat3(gbufferModelViewInverse) * reflectDirView);
    //    vec2 reflectLUTUV = coords_polarAzimuthEqualArea(reflectDir);
    //    vec3 reflectRadiance = texture(usam_skyLUT, reflectLUTUV).rgb;
    //    vec3 skySpecularV = fresnel * sunRadiance * reflectRadiance;

    mainOut = vec3(0.0);
    mainOut += 0.001 * material.albedo;
    mainOut += emissiveV;
    mainOut += sunLighting;
    mainOut += moonLighting;
    //    mainOut += skySpecularV;

    ssgiOut = vec3(0.0);
    ssgiOut += multiBounceV;
    ssgiOut += emissiveV;
    ssgiOut += sunLighting;
    ssgiOut += moonLighting;
    //    ssgiOut += skySpecularV;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec2 screenCoord = (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;

        float viewZ = imageLoad(uimg_gbufferViewZ, texelPos).r;
        gbuffer_unpack(texelFetch(usam_gbufferData, texelPos, 0), gData);

        g_viewCoord = coords_toViewCoord(screenCoord, viewZ, gbufferProjectionInverse);
        g_viewDir = normalize(-g_viewCoord);
        coord3Rand[0] = rand_hash31(floatBitsToUint(g_viewCoord.xyz)) & 1023u;
        coord3Rand[1] = rand_hash31(floatBitsToUint(g_viewCoord.xzy)) & 1023u;

        vec2 projRejectOut;
        ndpacking_updateProjReject(usam_lastNZ, texelPos, screenCoord, gData.normal, g_viewCoord, projRejectOut);
        imageStore(uimg_projReject, texelPos, vec4(projRejectOut, 0.0, 0.0));

        vec4 mainOut = vec4(0.0);
        vec4 temp1Out = vec4(0.0);
        vec4 temp2Out = vec4(0.0);

        mainOut.a = 1.0;
        mainOut.rgb += renderSunMoon(texelPos);

        if (viewZ != -65536.0) {
            Material material = material_decode(gData);

            temp1Out.rgb = gData.normal;
            temp1Out.a = float(any(greaterThan(material.emissive, vec3(0.0))));
            temp2Out.a = gData.lmCoord.y;

            if (all(equal(gData.normal, vec3(1.0)))) {
                mainOut = vec4(material.albedo, 1.0);
                temp2Out = vec4(0.0, 0.0, 0.0, 1.0);
            } else {
                doLighting(material, gData.normal, g_viewDir, mainOut.rgb, temp2Out.rgb);
            }
        }

        imageStore(uimg_main, texelPos, mainOut);
        imageStore(uimg_temp1, texelPos, temp1Out);
        imageStore(uimg_temp2, texelPos, temp2Out);
    }
}