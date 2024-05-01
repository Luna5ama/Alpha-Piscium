#version 460 compatibility

out vec2 lmcoord;
out vec4 glcolor;

void main() {
	gl_Position = ftransform();
	lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
	glcolor = gl_Color;
}