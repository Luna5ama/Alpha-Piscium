#version 460 compatibility

#include "_Util.glsl"
#include "rtwsm/RTWSM.glsl"
#include "atmosphere/Common.glsl"

uniform sampler2D usam_main;
uniform usampler2D usam_gbuffer;
uniform sampler2D usam_ssvbil;

in vec2 frag_texCoord;

ivec2 intTexCoord = ivec2(gl_FragCoord.xy);

/* RENDERTARGETS:0 */
layout(location = 0) out vec4 rt_main;

void main() {
    rt_main = texelFetch(usam_main, intTexCoord, 0);

    GBufferData gData;
    gbuffer_unpack(texelFetch(usam_gbuffer, ivec2(gl_FragCoord.xy), 0), gData);
    Material material = material_decode(gData);

    vec4 ssvbilSample = texelFetch(usam_ssvbil, intTexCoord, 0);
    vec3 indirectV = ssvbilSample.rgb * material.albedo;

    float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));
    rt_main.rgb *= mix(sqrt(ssvbilSample.a), 1.0, shadowIsSun);

    rt_main.rgb += indirectV;
}