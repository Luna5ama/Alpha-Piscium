#version 460 compatibility

uniform sampler2D gtexture;

in vec2 texcoord;
in vec4 glcolor;

/* DRAWBUFFERS:0 */
layout(location = 0) out vec4 color;

void main() {
	discard;
	color = texture(gtexture, texcoord) * glcolor;
	if (color.a < 0.1) {
		discard;
	}
}