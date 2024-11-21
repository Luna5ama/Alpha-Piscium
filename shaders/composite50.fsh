#version 460 compatibility

#include "../_Util.glsl"

uniform sampler2D usam_main;
uniform sampler2D usam_temp1;
uniform sampler2D usam_temp3;
uniform sampler2D usam_ssvbil;
uniform sampler2D usam_bentNormal;

in vec2 frag_texCoord;

/* RENDERTARGETS:0 */
layout(location = 0) out vec4 rt_out;

void main() {
    ivec2 intTexCoord = ivec2(gl_FragCoord.xy);
    vec4 color = texelFetch(usam_main, intTexCoord, 0);

    #if defined(SETTING_DEBUG_SSVBIL_GI)
    color = vec4(texelFetch(usam_ssvbil, intTexCoord, 0).rgb, 1.0);
    #elif defined(SETTING_DEBUG_SSVBIL_AO)
    color = vec4(texelFetch(usam_ssvbil, intTexCoord, 0).aaa, 1.0);
    #elif defined(SETTING_DEBUG_SSVBIL_BENT_NORMAL)
    color = vec4(texelFetch(usam_bentNormal, intTexCoord, 0).rgb, 1.0);
    #elif defined(SETTING_DEBUG_WORLD_NORMAL)
    vec3 viewNormal = texelFetch(usam_temp1, intTexCoord, 0).rgb;
    vec3 worldNormal = mat3(gbufferModelViewInverse) * viewNormal;
    color = vec4(worldNormal * 0.5 + 0.5, 1.0);
    #endif

    rt_out = color;
}