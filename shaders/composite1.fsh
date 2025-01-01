#version 460 compatibility

#include "_Util.glsl"
#include "atmosphere/Common.glsl"

in vec2 frag_texCoord;

/* RENDERTARGETS:0 */
layout(location = 0) out vec4 rt_out;

// https://www.shadertoy.com/view/slSXRW
float celestialObjWithBloom(vec3 rayDir, vec3 objDir, float objAngularRadius, float gaussianMul) {
	const float minSunCosTheta = cos(objAngularRadius);

	float cosTheta = dot(rayDir, objDir);

	float offset = minSunCosTheta - cosTheta;
	float gaussianBloom = exp(-offset * 50000.0 * gaussianMul) * 0.5;
	float invBloom = 1.0 / (0.02 + offset * 300.0 * gaussianMul) * 0.01;
	return mix(gaussianBloom + invBloom, 1.0, float(cosTheta >= minSunCosTheta));
}

void main() {
	rt_out = vec4(0.0, 0.0, 0.0, 0.5);

	vec3 sunRadiance = global_sunRadiance.rgb * global_sunRadiance.a;
	vec3 viewCoord = coords_toViewCoord(frag_texCoord, -far, gbufferProjectionInverse);

	vec3 viewDir = normalize(viewCoord);
	vec3 viewDirWorld = mat3(gbufferModelViewInverse) * viewDir;

	AtmosphereParameters atmosphere = getAtmosphereParameters();
	vec3 origin = atmosphere_viewToAtm(atmosphere, vec3(0.0));
	origin.y = max(origin.y, atmosphere.bottom + 0.5);
	vec3 earthCenter = vec3(0.0);
	float earthIntersect = raySphereIntersectNearest(origin, viewDirWorld, earthCenter, atmosphere.bottom);

	const float moonAngularRadius = 0.528611 * PI / 180.0;

	float sunV = celestialObjWithBloom(viewDir, uval_sunDirView, uval_sunAngularRadius, 1.0);
	float moonV = celestialObjWithBloom(viewDir, uval_moonDirView, moonAngularRadius, 4.0);

	rt_out.rgb += sunV * sunRadiance;
	rt_out.rgb += moonV * sunRadiance * MOON_RADIANCE_MUL * 10.0;
	rt_out.rgb *= step(earthIntersect, 0.0);
}