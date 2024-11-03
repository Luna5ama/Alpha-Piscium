#version 460 compatibility

#include "_Util.glsl"
#include "rtwsm/RTWSM.glsl"

uniform usampler2D usam_gbuffer;
uniform sampler2D usam_viewZ;

uniform sampler2D depthtex0;

const bool generateShadowMipmap = true;
const bool shadowtex0Mipmap = true;
const bool shadowtex1Mipmap = true;

uniform sampler2DShadow shadowtex0;
const bool shadowHardwareFiltering0 = true;

uniform sampler2D shadowtex1;
const bool shadowHardwareFiltering1 = false;
uniform sampler2D usam_rtwsm_warpingMap;

in vec2 frag_texCoord;
in vec2 frag_scaledTexCoord;

ivec2 intTexCoord = ivec2(gl_FragCoord.xy);
GBufferData gData;
vec3 g_viewCoord;
vec3 g_viewDir;

uint worldCoordRand[6];

/* RENDERTARGETS:0 */
layout(location = 0) out vec4 rt_out;

float searchBlocker(vec3 shadowTexCoord) {
	#define BLOCKER_SEARCH_LOD SETTING_PCSS_BLOCKER_SEARCH_LOD

	#define BLOCKER_SEARCH_N SETTING_PCSS_BLOCKER_SEARCH_COUNT

	float blockerSearchRange = 0.2;
	uint idxB = frameCounter * BLOCKER_SEARCH_N + worldCoordRand[1];

	float blockerDepth = 0.0f;
	int n = 0;

	for (int i = 0; i < BLOCKER_SEARCH_N; i++) {
		vec2 randomOffset = (r2Seq2(idxB) * 2.0 - 1.0);
		vec3 sampleTexCoord = shadowTexCoord;
		sampleTexCoord.xy += randomOffset * blockerSearchRange * vec2(shadowProjection[0][0], shadowProjection[1][1]);
		vec2 texelSize;
		sampleTexCoord.xy = rtwsm_warpTexCoordTexelSize(usam_rtwsm_warpingMap, sampleTexCoord.xy, texelSize);
		float depth = rtwsm_sampleShadowDepth(shadowtex1, sampleTexCoord, BLOCKER_SEARCH_LOD).r;
		blockerDepth += step(depth, sampleTexCoord.z) * depth;
		n += int(sampleTexCoord.z > depth);
		idxB++;
	}
	blockerDepth /= float(n);

	return n != 0 ? rtwsm_linearDepth(blockerDepth) - rtwsm_linearDepth(shadowTexCoord.z) : 0.0;
}

float calcShadow(float sssFactor) {
	vec3 viewCoord = g_viewCoord;
	float distnaceSq = dot(viewCoord, viewCoord);
	#define NORMAL_OFFSET_DISTANCE_FACTOR 16384.0
	float normalOffset = 1.0 - (NORMAL_OFFSET_DISTANCE_FACTOR / (NORMAL_OFFSET_DISTANCE_FACTOR + distnaceSq));
	float normalDot = (1.0 - abs(dot(gData.normal, g_viewDir)));
	viewCoord += gData.normal * max(normalOffset * 2.0 * (normalDot * 0.6 + 0.4), 0.02);

	vec4 worldCoord = gbufferModelViewInverse * vec4(viewCoord, 1.0);
	vec4 shadowTexCoordCS = coords_shadowDeRotateMatrix(shadowModelView) * shadowProjection * shadowModelView * worldCoord;
	shadowTexCoordCS /= shadowTexCoordCS.w;

	vec3 shadowTexCoord = shadowTexCoordCS.xyz * 0.5 + 0.5;

	float blockerDistance = searchBlocker(shadowTexCoord);
	float penumbraMult = 0.000005 * SETTING_PCSS_VPF * blockerDistance;

	float ssRange = SETTING_PCSS_BPF * 0.01;
	ssRange += SHADOW_MAP_SIZE.x * 1.0 * sssFactor;
	ssRange += SHADOW_MAP_SIZE.x * penumbraMult;
	ssRange = saturate(ssRange);
	ssRange *= 0.2;

	#define SAMPLE_N SETTING_PCSS_SAMPLE_COUNT

	float shadow = 0.0;
	uint idxSS = (frameCounter + worldCoordRand[0]) * SAMPLE_N;

	#define DEPTH_BIAS_DISTANCE_FACTOR 4.0
	float dbfDistanceCoeff = (DEPTH_BIAS_DISTANCE_FACTOR / (DEPTH_BIAS_DISTANCE_FACTOR + distnaceSq));
	float depthBiasFactor = mix(0.002, -0.0005, dbfDistanceCoeff);

	for (int i = 0; i < SAMPLE_N; i++) {
		vec2 randomOffset = (r2Seq2(idxSS) * 2.0 - 1.0);
		vec3 sampleTexCoord = shadowTexCoord;
		sampleTexCoord.xy += ssRange * randomOffset * vec2(shadowProjection[0][0], shadowProjection[1][1]);
		vec2 texelSize;
		sampleTexCoord.xy = rtwsm_warpTexCoordTexelSize(usam_rtwsm_warpingMap, sampleTexCoord.xy, texelSize);
		float depthBias = SHADOW_MAP_SIZE.y * depthBiasFactor / length(texelSize);
		sampleTexCoord.z -= depthBias;
		shadow += rtwsm_sampleShadowDepth(shadowtex0, sampleTexCoord, 0.0);
		idxSS++;
	}
	shadow /= float(SAMPLE_N);

	return mix(shadow, 1.0, linearStep(shadowDistance - 16.0, shadowDistance, length(worldCoord.xz)));
}

void doStuff() {
	float shadow = calcShadow(0.0);
	shadow *= step(0.0, dot(gData.normal, shadowLightPosition));

	vec3 color = gData.albedo * mix(0.5, 1.0, shadow);
	rt_out = vec4(color, 1.0);
}

void main() {
	gbuffer_unpack(texelFetch(usam_gbuffer, ivec2(gl_FragCoord.xy), 0), gData);
	float viewZ = texelFetch(usam_viewZ, ivec2(gl_FragCoord.xy), 0).r;
	g_viewCoord = coords_toViewCoord(frag_texCoord, viewZ, gbufferProjectionInverse);
	g_viewDir = normalize(-g_viewCoord);

	worldCoordRand[0] = uint(rand(g_viewCoord.xyz) * 1024.0);
	worldCoordRand[1] = uint(rand(g_viewCoord.xzy) * 1024.0);
	worldCoordRand[2] = uint(rand(g_viewCoord.yxz) * 1024.0);
	worldCoordRand[3] = uint(rand(g_viewCoord.yzx) * 1024.0);
	worldCoordRand[4] = uint(rand(g_viewCoord.zxy) * 1024.0);
	worldCoordRand[5] = uint(rand(g_viewCoord.zyx) * 1024.0);

	doStuff();
}