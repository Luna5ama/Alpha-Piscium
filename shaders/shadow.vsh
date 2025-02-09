#version 460 compatibility

#include "_Util.glsl"
#include "rtwsm/RTWSM.glsl"

uniform sampler2D usam_rtwsm_imap;

layout(r32i) uniform iimage2D uimg_rtwsm_imap;

attribute vec4 mc_Entity;

out vec2 texcoord;
out vec4 glcolor;
out float frag_viewZ;

void main() {
	texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
	glcolor = gl_Color;

	vec4 shadowClipPos = ftransform();
	vec4 shadowClipPosRotated = global_shadowRotationMatrix * shadowClipPos;
	gl_Position = shadowClipPosRotated;
	frag_viewZ = -gl_Position.w;
	gl_Position /= gl_Position.w;

	vec2 vPosTS = gl_Position.xy * 0.5 + 0.5;
	vPosTS = rtwsm_warpTexCoord(usam_rtwsm_imap, vPosTS);

	gl_Position.xy = vPosTS * 2.0 - 1.0;

	#ifdef SETTING_RTWSM_F
	vec4 camViewPos = gbufferModelView * shadowModelViewInverse * shadowProjectionInverse * shadowClipPos;
	camViewPos /= camViewPos.w;
	vec4 camClipPos = gbufferProjection * camViewPos;

	uint survived = uint((gl_VertexID & 3) == 0);
	survived &= uint(all(lessThan(abs(shadowClipPosRotated.xyz), shadowClipPosRotated.www)));
	survived &= uint(camClipPos.w > 0.0);
	survived &= uint(all(lessThan(abs(camClipPos.xyz), camClipPos.www)));

	if (bool(survived)){
		vec2 shadowNdcPos = shadowClipPosRotated.xy / shadowClipPosRotated.w;
		vec2 shadowScreenPos = shadowNdcPos * 0.5 + 0.5;
		ivec2 importanceTexelPos = ivec2(shadowScreenPos * vec2(SETTING_RTWSM_IMAP_SIZE));

		float importance = SETTING_RTWSM_F_BASE;

		// Distance function
		#if SETTING_RTWSM_F_D > 0.0
		importance *= (SETTING_RTWSM_F_D) / (SETTING_RTWSM_F_D + dot(camViewPos, camViewPos));
		#endif

		importance = max(importance, uval_rtwsmMin.x);

		imageAtomicMax(uimg_rtwsm_imap, importanceTexelPos, floatBitsToInt(importance));
 	}
	#endif
}