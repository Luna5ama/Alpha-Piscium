#include "/Base.glsl"

out vec2 frag_texCoord;

void main() {
	gl_Position = ftransform();
	frag_texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
}