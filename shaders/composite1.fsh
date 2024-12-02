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
	float cosSun = saturate(dot(viewDir, uval_sunDirView));

	vec3 viewDirWorld = mat3(gbufferModelViewInverse) * viewDir;

	AtmosphereParameters atmosphere = getAtmosphereParameters();
	vec3 origin = atmosphere_viewToAtm(atmosphere, vec3(0.0));
	vec3 earthCenter = vec3(0.0);
	float earthIntersect = raySphereIntersectNearest(origin, viewDirWorld, earthCenter, atmosphere.bottom);

	rt_out.rgb = step(earthIntersect, 0.0) * pow(cosSun, 6000.0) * sunRadiance;
}