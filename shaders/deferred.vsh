#version 460 compatibility

#include "utils/Common.glsl"

out vec2 frag_texCoord;
out vec2 frag_scaledTexCoord;

void main() {
	gl_Position = ((ftransform() * 0.5 + 0.5) * RENDER_RESOLUTION) * 2.0 - 1.0;
	frag_texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
	frag_scaledTexCoord = frag_texCoord * RENDER_RESOLUTION;
}