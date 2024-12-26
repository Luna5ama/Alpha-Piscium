#version 460 compatibility

#include "_Util.glsl"
#include "atmosphere/Common.glsl"

in vec2 frag_texCoord;

/* RENDERTARGETS:0 */
layout(location = 0) out vec4 rt_out;

void main() {
	rt_out = vec4(0.0, 0.0, 0.0, 0.5);

	vec3 sunRadiance = global_sunRadiance.rgb * global_sunRadiance.a;
	vec3 viewCoord = coords_toViewCoord(frag_texCoord, -far, gbufferProjectionInverse);

	vec3 viewDir = normalize(viewCoord);
	float cosTheta = saturate(dot(viewDir, uval_sunDirView));

	vec3 viewDirWorld = mat3(gbufferModelViewInverse) * viewDir;

	AtmosphereParameters atmosphere = getAtmosphereParameters();
	vec3 origin = atmosphere_viewToAtm(atmosphere, vec3(0.0));
	origin.y += 1.0;
	vec3 earthCenter = vec3(0.0);
	float earthIntersect = raySphereIntersectNearest(origin, viewDirWorld, earthCenter, atmosphere.bottom);

	float sunCosTheta = cos(uval_sunAngularRadius * 1.0);

	// https://www.shadertoy.com/view/slSXRW
	float offset = sunCosTheta - cosTheta;
	float gaussianBloom = exp(-offset * 50000.0) * 0.5;
	float invBloom = 1.0 / (0.02 + offset * 300.0) * 0.01;

	float sunV = step(sunCosTheta, cosTheta) + gaussianBloom + invBloom;

	rt_out.rgb = step(earthIntersect, 0.0) * sunV * sunRadiance;
}