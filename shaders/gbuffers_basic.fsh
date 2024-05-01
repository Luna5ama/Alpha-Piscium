#version 460 compatibility

uniform sampler2D lightmap;

in vec2 lmcoord;
in vec4 glcolor;

/* DRAWBUFFERS:0 */
layout(location = 0) out vec4 color;

void main() {
	color = glcolor * texture(lightmap, lmcoord);
}