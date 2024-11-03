#include "../_Util.glsl"

uniform sampler2D colortex0;

in vec2 frag_texCoord;

/* RENDERTARGETS:0 */
layout(location = 0) out vec4 rt_out;

void main() {
	ivec2 intTexCoord = ivec2(gl_FragCoord.xy);
	vec4 color = texelFetch(colortex0, intTexCoord, 0);
	rt_out = color;
}