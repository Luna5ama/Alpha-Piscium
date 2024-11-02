#version 460 compatibility

#include "utils/Common.glsl"

uniform sampler2D lightmap;
uniform sampler2D texture;

varying vec2 texcoord;
varying vec4 glcolor;

void main() {
	vec4 color = textureLod(texture, texcoord, 0.0) * glcolor;

	gl_FragData[0] = color;
}