#version 460 compatibility

#include "utils/Uniforms.glsl"
#include "utils/Coords.glsl"
#include "utils/PackedGBuffers.glsl"
#include "utils/R2Seqs.glsl"
#include "utils/Rand.glsl"
#include "rtwsm/RTWSM.glsl"

uniform usampler2D colortex1;
uniform usampler2D colortex2;
uniform sampler2D depthtex0;

uniform sampler2DShadow shadowtex0;
const bool shadowHardwareFiltering0 = true;
uniform sampler2D usam_rtwsm_warpingMap;

varying vec2 texcoord;

/* DRAWBUFFERS:0 */
layout(location = 0) out vec4 rt_out;

#define SHADOW_SAMPLES 16

void main() {
	ivec2 intTexCoord = ivec2(gl_FragCoord.xy);
	float gDepth = texelFetch(depthtex0, intTexCoord, 0).r;
	if (gDepth == 1.0) {
		discard;
	}
	PackedGBufferData gData;
	pgbuffer_unpack(texelFetch(colortex1, intTexCoord, 0), texelFetch(colortex2, intTexCoord, 0), gData);


	vec3 color = vec3(0.0);

//	vec3 viewCoord = vec3(gData.viewCoord, -coords_linearizeDepth(gDepth, near, far));
	vec3 viewCoord = gData.viewCoord;
	vec4 sceneCoord = gbufferModelViewInverse * vec4(viewCoord, 1.0);
	sceneCoord.xyz += gData.worldNormal * max(length(gData.viewCoord), 1.0) * 0.005;

	vec4 shadowCS = shadowProjection * (shadowModelView * sceneCoord);
	shadowCS /= shadowCS.w;
	vec3 shadowTS = shadowCS.xyz * 0.5 + 0.5;
	vec2 texelSize;
	shadowTS.xy = rtwsm_warpTexCoordTexelSize(usam_rtwsm_warpingMap, shadowTS.xy, texelSize);
	shadowTS.z -= 1.0 / float(shadowMapResolution);

	float shadowValue = 0.0;

	uint sceneCoordRand = uint(rand(sceneCoord.xyz) * 1024.0);
	uint r2idx = sceneCoordRand * SHADOW_SAMPLES;

	vec2 ssRange = texelSize * 256.0;

	for (int i = 0; i < SHADOW_SAMPLES; i++) {
		vec2 offset = r2Seq2(r2idx) * ssRange - ssRange * 0.5;
		shadowValue += rtwsm_sampleShadowDepthOffset(shadowtex0, shadowTS, 0.0, offset);
		r2idx++;
	}

	shadowValue /= float(SHADOW_SAMPLES);
	shadowValue = shadowValue * 0.8 + 0.2;

	color += gData.albedo.rgb * shadowValue;


	rt_out = vec4(color, 1.0);
}