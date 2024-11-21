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
    #endif

    rt_out = color;
}