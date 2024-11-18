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
	float cosSun = saturate(dot(viewDir, sunPosition * 0.01));

//	rt_out.rgb = smoothstep(uval_sunAngularRadius.x * 4.0, uval_sunAngularRadius.x, acos(cosSun)) * sunRadiance;
	rt_out.rgb = pow(cosSun, 9000.0) * sunRadiance;
}