#version 460 compatibility

#include "utils/Common.glsl"
#include "utils/Coords.glsl"
#include "utils/R2Seqs.glsl"
#include "utils/Rand.glsl"
#include "rtwsm/RTWSM.glsl"

uniform sampler2D TEX_G_VIEWCOORD;
uniform sampler2D TEX_G_ALBEDO;
uniform sampler2D TEX_G_NORMAL;

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

/* DRAWBUFFERS:0 */
layout(location = 0) out vec4 rt_out;

vec3 gViewCoord;
vec4 gAlbedo;
vec3 gWorldCoord;
vec3 gWorldNormal;
vec3 gViewNormal;

uint worldCoordRand[6];

float searchBlocker(vec3 shadowCoord, vec2 texelSize) {
	const float BLOCKER_SEARCH_LOD = 0.0;

	#define BLOCKER_SEARCH_N 4

	vec2 blockerSearchRange = texelSize * 512.0;
	uint idxB = frameCounter * BLOCKER_SEARCH_N + worldCoordRand[1];

	float blockerDepth = 0.0f;
	int n = 0;

	#if BLOCKER_SEARCH_N == 1
	vec2 offset = r2Seq2(idxB) * blockerSearchRange - blockerSearchRange * 0.5;
	float depth = rtwsm_sampleShadowDepthOffset(shadowtex1, shadowCoord.xy, BLOCKER_SEARCH_LOD, offset).r;
	blockerDepth += step(depth, shadowCoord.z) * depth;
	n += int(shadowCoord.z > depth);
	#else
	for (int i = 0; i < BLOCKER_SEARCH_N; i++) {
		vec2 offset = r2Seq2(idxB) * blockerSearchRange - blockerSearchRange * 0.5;
		float depth = rtwsm_sampleShadowDepthOffset(shadowtex1, shadowCoord.xy, BLOCKER_SEARCH_LOD, offset).r;
		blockerDepth += step(depth, shadowCoord.z) * depth;
		n += int(shadowCoord.z > depth);
		idxB++;
	}
	blockerDepth /= float(n);
	#endif

	return n != 0 ? rtwsm_linearDepth(shadowCoord.z) - rtwsm_linearDepth(blockerDepth) : 0.0;
}

float calcShadow(float sssFactor) {
	vec3 viewCoord = gViewCoord;

	float dist = length(gWorldCoord);
	float lightDot = dot(gViewNormal, normalize(shadowLightPosition)) * 0.5 + 0.5;

	float minNormalOffset = 0.01;
	float maxNormalOffset = clamp(dist * 0.01, 0.01, 0.5);
	viewCoord += gViewNormal * mix(minNormalOffset, maxNormalOffset, lightDot);

	float minlightDirOffset = 0.001;
	float maxlightDirlOffset = clamp(dist * 0.01, 0.005, 8.0);
	float lightDirOffset = -shadowProjection[2][2] * mix(minlightDirOffset, maxlightDirlOffset, lightDot);

	vec4 worldCoord = gbufferModelViewInverse * vec4(viewCoord, 1.0);
	vec4 shadowCoordCS = shadowProjection * (shadowModelView * worldCoord);
	shadowCoordCS /= shadowCoordCS.w;

	vec3 shadowCoord = shadowCoordCS.xyz * 0.5 + 0.5;

	vec2 texelSize;
	shadowCoord.xy = rtwsm_warpTexCoordTexelSize(usam_rtwsm_warpingMap, shadowCoord.xy, texelSize);
	lightDirOffset *= 0.01 / (texelSize.x + texelSize.y);

	float blockerDistance = searchBlocker(shadowCoord, texelSize);
	float penumbraMult = 64.0 * blockerDistance;

	#define SAMPLE_N 16

	vec2 ssRange = mix(
			texelSize * 256.0,
			texelSize * 4096.0 / max(sqrt(dist), 1.0),
			sssFactor
	);
	ssRange += texelSize * penumbraMult;

	float shadow = 0.0;
	uint idxSS = (frameCounter + worldCoordRand[0]) * SAMPLE_N;
	uint idxSSS = (frameCounter + worldCoordRand[1]) * SAMPLE_N;

	#if SAMPLE_N == 1
	vec2 offset = r2Seq2(idxSS) * ssRange - ssRange * 0.5;
	vec3 zOffsetShadowCoord = shadowCoord;
	zOffsetShadowCoord.z -= mix(1.0, r2Seq1(idxSSS), sssFactor) * lightDirOffset;
	shadow += rtwsm_sampleShadowDepthOffset(shadowtex0, zOffsetShadowCoord, 0.0, offset);
	#else
	for (int i = 0; i < SAMPLE_N; i++) {
		vec2 offset = r2Seq2(idxSS) * ssRange - ssRange * 0.5;
		vec3 zOffsetShadowCoord = shadowCoord;
		zOffsetShadowCoord.z -= mix(1.0, r2Seq1(idxSSS), sssFactor) * lightDirOffset;
		shadow += rtwsm_sampleShadowDepthOffset(shadowtex0, zOffsetShadowCoord, 0.0, offset);
		idxSS++;
		idxSSS++;
	}
	shadow /= float(SAMPLE_N);
	#endif

	return shadow;
}

void doStuff() {
	float sssFactor = 0.0;
	float shadow = calcShadow(sssFactor);

	vec3 color = gAlbedo.rgb * mix(0.5, 1.0, shadow);
	rt_out = vec4(color, 1.0);
}

void main() {
	gViewCoord = texelFetch(TEX_G_VIEWCOORD, ivec2(gl_FragCoord.xy), 0).xyz;
	gWorldCoord = (gbufferModelViewInverse * vec4(gViewCoord, 1.0)).xyz;
	gAlbedo = texelFetch(TEX_G_ALBEDO, ivec2(gl_FragCoord.xy), 0);
	gViewNormal = texelFetch(TEX_G_NORMAL, ivec2(gl_FragCoord.xy), 0).xyz * 2.0 - 1.0;
	gWorldNormal = normalize(mat3(gbufferModelViewInverse) * gViewNormal);


	worldCoordRand[0] = uint(rand(gWorldCoord.xyz) * 1024.0);
	worldCoordRand[1] = uint(rand(gWorldCoord.xzy) * 1024.0);
	worldCoordRand[2] = uint(rand(gWorldCoord.yxz) * 1024.0);
	worldCoordRand[3] = uint(rand(gWorldCoord.yzx) * 1024.0);
	worldCoordRand[4] = uint(rand(gWorldCoord.zxy) * 1024.0);
	worldCoordRand[5] = uint(rand(gWorldCoord.zyx) * 1024.0);

	doStuff();
}