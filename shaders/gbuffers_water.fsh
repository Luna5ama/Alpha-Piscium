#version 460 compatibility

#include "_Util.glsl"

uniform sampler2D lightmap;
uniform sampler2D gtexture;

in vec2 lmcoord;
in vec2 texcoord;
in vec4 glcolor;

/* DRAWBUFFERS:0 */
layout(location = 0) out vec4 color;

void main() {
	color = texture(gtexture, texcoord) * glcolor;
	color *= texture(lightmap, lmcoord);
	if (color.a < 0.1) {
		discard;
	}
	color.rgb = colors_srgbToLinear(color.rgb) * 4.0;
}