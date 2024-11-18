#version 460 compatibility

#include "_Util.glsl"
#include "rtwsm/RTWSM.glsl"

uniform sampler2D usam_rtwsm_warpingMap;

attribute vec4 mc_Entity;

out vec2 texcoord;
out vec4 glcolor;
out float frag_viewZ;

void main() {
	texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
	glcolor = gl_Color;

	gl_Position = global_shadowRotationMatrix  * ftransform();
	frag_viewZ = -gl_Position.w;
	gl_Position /= gl_Position.w;

	vec2 texelSize;
	vec2 vPosTS = gl_Position.xy * 0.5 + 0.5;
	vPosTS = rtwsm_warpTexCoordTexelSize(usam_rtwsm_warpingMap, vPosTS, texelSize);

	gl_Position.xy = vPosTS * 2.0 - 1.0;
}