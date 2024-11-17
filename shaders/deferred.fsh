#version 460 compatibility

#include "_Util.glsl"

in vec2 frag_texCoord;

/* RENDERTARGETS:0 */
layout(location = 0) out vec4 rt_out;

void main() {
	rt_out = vec4(0.0, 0.0, 0.0, 0.5);

	vec3 sunRadiance = global_sunRadiance.rgb * global_sunRadiance.a;
	vec3 viewCoord = coords_toViewCoord(frag_texCoord, -far, gbufferProjectionInverse);

	vec3 viewDir = normalize(viewCoord);
	float sunDot = saturate(dot(viewDir, sunPosition * 0.01));

	rt_out.rgb = pow(sunDot, 9000.0) * sunRadiance;
}