#version 460 compatibility

#include "../_Util.glsl"

uniform sampler2D usam_main;
uniform sampler2D usam_SSVBILLast;

in vec2 frag_texCoord;

/* RENDERTARGETS:0 */
layout(location = 0) out vec4 rt_out;

void main() {
    ivec2 intTexCoord = ivec2(gl_FragCoord.xy);
    #if defined(DEBUG_SSVBIL_GI)
    vec4 color = vec4(texelFetch(usam_SSVBILLast, intTexCoord, 0).rgb, 1.0);
    #elif defined(DEBUG_SSVBIL_AO)
    vec4 color = vec4(texelFetch(usam_SSVBILLast, intTexCoord, 0).aaa, 1.0);
    #else
    vec4 color = texelFetch(usam_main, intTexCoord, 0);
    #endif
    rt_out = color;
}