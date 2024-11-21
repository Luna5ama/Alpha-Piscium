#version 460 compatibility

#include "_Util.glsl"

uniform sampler2D usam_temp1;
uniform sampler2D usam_viewZ;
uniform usampler2D usam_lastNZ;

in vec2 frag_texCoord;

/* RENDERTARGETS:12,13 */
layout(location = 0) out vec2 rt_projReject;
layout(location = 1) out uvec2 rt_nz;

uvec2 ndpacking_pack(vec3 normal, float depth) {
	uvec2 packedV;
	packedV.x = packSnorm4x8(vec4(normal, 0.0));
	packedV.y = floatBitsToUint(depth);
	return packedV;
}

void ndpacking_unpack(uvec2 packedV, out vec3 normal, out float depth) {
	normal = unpackSnorm4x8(packedV.x).xyz;
	depth = uintBitsToFloat(packedV.y);
}

void main() {
	ivec2 intTexCoord = ivec2(gl_FragCoord.xy);
	float currZ = texelFetch(usam_viewZ, intTexCoord, 0).r;
	vec3 currN = texelFetch(usam_temp1, intTexCoord, 0).rgb;

	vec3 cameraDelta = cameraPosition - previousCameraPosition;
	vec3 currView = coords_toViewCoord(frag_texCoord, currZ, gbufferProjectionInverse);
	vec4 currScene = gbufferModelViewInverse * vec4(currView, 1.0);
	vec4 curr2PrevScene = coord_sceneCurrToPrev(currScene);
	vec4 curr2PrevView = gbufferPreviousModelView * curr2PrevScene;
	vec4 curr2PrevClip = gbufferProjection * curr2PrevView;

	{
		float prevZ;
		vec3 prevN;
		ndpacking_unpack(texelFetch(usam_lastNZ, intTexCoord, 0).xy, prevN, prevZ);

		vec3 prevView = coords_toViewCoord(frag_texCoord, prevZ, gbufferPreviousProjectionInverse);
		vec4 prevScene = gbufferPreviousModelViewInverse * vec4(prevView, 1.0);
		vec4 prev2CurrScene = coord_scenePrevToCurr(prevScene);
		vec4 prev2CurrClip = gbufferPreviousProjection * gbufferModelView * prev2CurrScene;

		uint flag = 0u;
		flag |= uint(currZ != 1.0) & uint(any(greaterThanEqual(abs(curr2PrevClip.xyz), curr2PrevClip.www)));
		flag |= uint(prevZ != 0.0) & uint(any(greaterThanEqual(abs(prev2CurrClip.xyz), prev2CurrClip.www)));

		rt_projReject.x = float(flag);
	}

	{
		vec4 prevZs = uintBitsToFloat(textureGather(usam_lastNZ, frag_texCoord, 1));

		vec3 diff;
		float dotV = 0.0;
		diff = coords_toViewCoord(frag_texCoord, prevZs.x, gbufferPreviousProjectionInverse) - curr2PrevView.xyz;
		dotV += dot(currN, diff);
		diff = coords_toViewCoord(frag_texCoord, prevZs.y, gbufferPreviousProjectionInverse) - curr2PrevView.xyz;
		dotV += dot(currN, diff);
		diff = coords_toViewCoord(frag_texCoord, prevZs.z, gbufferPreviousProjectionInverse) - curr2PrevView.xyz;
		dotV += dot(currN, diff);
		diff = coords_toViewCoord(frag_texCoord, prevZs.w, gbufferPreviousProjectionInverse) - curr2PrevView.xyz;
		dotV += dot(currN, diff);
		dotV = step(3.0, abs(dotV));

		rt_projReject.y = dotV;
	}

	rt_nz = ndpacking_pack(currN, currZ);
}