#version 460 compatibility

#include "utils/Settings.glsl"

uniform sampler2D lightmap;
uniform sampler2D texture;

varying vec2 texcoord;
varying vec4 glcolor;

void main() {
	vec4 color = texture2D(texture, texcoord) * glcolor;

	gl_FragData[0] = color;
}