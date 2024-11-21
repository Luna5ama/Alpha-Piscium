#version 460 compatibility

#include "_Util.glsl"
#include "rtwsm/RTWSM.glsl"
#include "atmosphere/Common.glsl"

uniform sampler2D usam_main;
uniform usampler2D usam_gbuffer;
uniform sampler2D usam_viewZ;

uniform sampler2D shadowcolor0;
uniform sampler2D usam_rtwsm_warpingMap;

uniform sampler2D usam_transmittanceLUT;
uniform sampler2D usam_skyLUT;

uniform sampler2D usam_ssvbil;

in vec2 frag_texCoord;

ivec2 intTexCoord = ivec2(gl_FragCoord.xy);
GBufferData gData;
vec3 g_viewCoord;
vec3 g_viewDir;

uint coord3Rand[2];

/* RENDERTARGETS:0 */
layout(location = 0) out vec4 rt_main;

void doLighting(Material material, vec3 shadow, vec3 L, vec3 N, vec3 V) {
    vec4 ssvbilSample = texelFetch(usam_ssvbil, intTexCoord, 0);
    vec3 indirectV = SSVBIL_GI_STRENGTH * ssvbilSample.rgb * material.albedo;

    rt_main.rgb += indirectV;
}

void doStuff() {
    vec3 shadow = vec3(1.0);

    Material material = material_decode(gData);

    doLighting(material, shadow, sunPosition * 0.01, gData.normal, g_viewDir);
}

void main() {
    rt_main = texelFetch(usam_main, intTexCoord, 0);
    float viewZ = texelFetch(usam_viewZ, intTexCoord, 0).r;
    if (viewZ == 1.0) {
        return;
    }

    gbuffer_unpack(texelFetch(usam_gbuffer, ivec2(gl_FragCoord.xy), 0), gData);
    g_viewCoord = coords_toViewCoord(frag_texCoord, viewZ, gbufferProjectionInverse);
    g_viewDir = normalize(-g_viewCoord);

    coord3Rand[0] = rand_hash31(floatBitsToUint(g_viewCoord.xyz)) & 1023u;
    coord3Rand[1] = rand_hash31(floatBitsToUint(g_viewCoord.xzy)) & 1023u;

    doStuff();
}