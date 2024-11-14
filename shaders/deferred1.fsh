#version 460 compatibility

#include "_Util.glsl"
#include "rtwsm/RTWSM.glsl"

uniform usampler2D usam_gbuffer;
uniform sampler2D usam_viewZ;

uniform sampler2D depthtex0;

const bool generateShadowMipmap = true;
const bool shadowtex0Mipmap = true;

const bool shadowHardwareFiltering0 = true;
uniform sampler2D shadowtex0;
uniform sampler2DShadow shadowtex0HW;

uniform sampler2D usam_rtwsm_warpingMap;

in vec2 frag_texCoord;

ivec2 intTexCoord = ivec2(gl_FragCoord.xy);
GBufferData gData;
vec3 g_viewCoord;
vec3 g_viewDir;

uint coord3Rand[2];

/* RENDERTARGETS:0 */
layout(location = 0) out vec4 rt_out;

float searchBlocker(vec3 shadowTexCoord) {
	#define BLOCKER_SEARCH_LOD SETTING_PCSS_BLOCKER_SEARCH_LOD

	#define BLOCKER_SEARCH_N SETTING_PCSS_BLOCKER_SEARCH_COUNT

	float blockerSearchRange = 0.2;
	uint idxB = frameCounter * BLOCKER_SEARCH_N + coord3Rand[1];

	float blockerDepth = 0.0f;
	int n = 0;

	for (int i = 0; i < BLOCKER_SEARCH_N; i++) {
		vec2 randomOffset = (r2Seq2(idxB) * 2.0 - 1.0);
		vec3 sampleTexCoord = shadowTexCoord;
		sampleTexCoord.xy += randomOffset * blockerSearchRange * vec2(shadowProjection[0][0], shadowProjection[1][1]);
		vec2 texelSize;
		sampleTexCoord.xy = rtwsm_warpTexCoordTexelSize(usam_rtwsm_warpingMap, sampleTexCoord.xy, texelSize);
		float depth = rtwsm_sampleShadowDepth(shadowtex0, sampleTexCoord, BLOCKER_SEARCH_LOD).r;
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
	worldCoord = mix(worldCoord, vec4(0.0, 1.0, 0.0, 1.0), float(gData.materialID == 65534u));

	vec4 shadowTexCoordCS = global_shadowRotationMatrix * shadowProjection * shadowModelView * worldCoord;
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
	uint idxSS = (frameCounter + coord3Rand[0]) * SAMPLE_N;

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
		shadow += rtwsm_sampleShadowDepth(shadowtex0HW, sampleTexCoord, 0.0);
		idxSS++;
	}
	shadow /= float(SAMPLE_N);

	return mix(shadow, 1.0, linearStep(shadowDistance - 16.0, shadowDistance, length(worldCoord.xz)));
}

vec3 calcDirectLighting(vec3 L, vec3 N, vec3 V, vec3 albedo, float shadow) {
	vec3 directLight = vec3(0.0);
	float ambient = 100.0;
	directLight += ambient * gData.albedo;

	vec3 H = normalize(L + V);
	float NDotL = saturate(dot(N, L));
	float NDotV = saturate(dot(N, V));
	float NDotH = saturate(dot(N, H));
	float LDotV = saturate(dot(L, V));

	vec3 sunRadiance = global_sunRadiance.rgb * global_sunRadiance.a;
	vec3 diffuseV = bsdfs_diffuseHammon(NDotL, NDotV, NDotH, LDotV, albedo, 0.5);
	directLight += shadow * diffuseV * gData.albedo * sunRadiance;

	return directLight;
}

void doStuff() {
	float shadow = calcShadow(0.0);

	vec3 directLight = calcDirectLighting(sunPosition * 0.01, gData.normal, g_viewDir, gData.albedo, shadow);

	rt_out = vec4(directLight, 1.0);
}

void main() {
	gbuffer_unpack(texelFetch(usam_gbuffer, ivec2(gl_FragCoord.xy), 0), gData);
	float viewZ = texelFetch(usam_viewZ, ivec2(gl_FragCoord.xy), 0).r;
	g_viewCoord = coords_toViewCoord(frag_texCoord, viewZ, gbufferProjectionInverse);
	g_viewDir = normalize(-g_viewCoord);

	coord3Rand[0] = hash31(floatBitsToUint(g_viewCoord.xyz)) & 1023u;
	coord3Rand[1] = hash31(floatBitsToUint(g_viewCoord.xzy)) & 1023u;

	doStuff();
}