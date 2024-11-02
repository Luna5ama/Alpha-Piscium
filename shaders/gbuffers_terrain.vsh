#version 460 compatibility

#include "utils/Common.glsl"

attribute vec4 mc_Entity;

varying vec3 frag_viewCoord;
varying vec4 frag_colorMul;
varying vec2 frag_texCoord;
varying vec2 frag_lightMapCoord;
varying vec3 frag_worldNormal;

void main() {
	frag_worldNormal = (gl_NormalMatrix * gl_Normal.xyz);
	frag_texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
	frag_lightMapCoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
	frag_colorMul = gl_Color;

	vec4 pos = gbufferModelView * vec4(gl_Vertex.xyz, 1.0);
	frag_viewCoord = pos.xyz;
	gl_Position = gbufferProjection * pos;
}