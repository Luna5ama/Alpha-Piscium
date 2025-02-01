#version 460 compatibility

#include "_Util.glsl"
#include "atmosphere/Common.glsl"
#include "general/Lighting.glsl"

uniform sampler2D usam_main;
uniform sampler2D usam_temp1;
uniform usampler2D usam_gbufferData;
uniform sampler2D usam_gbufferViewZ;

uniform sampler2D usam_ssvbil;

in vec2 frag_texCoord;

/* RENDERTARGETS:0,1,2 */
layout(location = 0) out vec4 rt_main;
layout(location = 1) out vec4 rt_temp1;
layout(location = 2) out vec4 rt_temp2;

void doLighting(Material material, vec3 N, vec3 V) {
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

    rt_main.rgb = vec3(0.0);
    rt_main.rgb += 0.001 * material.albedo;
    rt_main.rgb += emissiveV;
    rt_main.rgb += sunLighting;
    rt_main.rgb += moonLighting;
//    rt_main.rgb += skySpecularV;

    rt_temp2.rgb = vec3(0.0);
    rt_temp2.rgb += multiBounceV;
    rt_temp2.rgb += emissiveV;
    rt_temp2.rgb += sunLighting;
    rt_temp2.rgb += moonLighting;
//    rt_temp2.rgb += skySpecularV;
}

void doStuff() {
    Material material = material_decode(gData);

    rt_main.a = 1.0;
    rt_temp1.rgb = gData.normal;
    rt_temp1.a = float(any(greaterThan(material.emissive, vec3(0.0))));
    rt_temp2.a = gData.lmCoord.y;

    if (all(equal(gData.normal, vec3(1.0)))) {
        rt_main = vec4(material.albedo, 1.0);
        rt_temp2 = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        doLighting(material, gData.normal, g_viewDir);
    }
}

void main() {
    texelPos = ivec2(gl_FragCoord.xy);
    rt_main = vec4(0.0);
    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    if (viewZ == -65536.0) {
        rt_main.rgb = texelFetch(usam_main, texelPos, 0).rgb;
        return;
    }

    gbuffer_unpack(texelFetch(usam_gbufferData, texelPos, 0), gData);

    g_viewCoord = coords_toViewCoord(frag_texCoord, viewZ, gbufferProjectionInverse);
    g_viewDir = normalize(-g_viewCoord);
    coord3Rand[0] = rand_hash31(floatBitsToUint(g_viewCoord.xyz)) & 1023u;
    coord3Rand[1] = rand_hash31(floatBitsToUint(g_viewCoord.xzy)) & 1023u;

    doStuff();
}