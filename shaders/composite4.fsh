#version 460 compatibility

#include "_Util.glsl"

uniform sampler2D usam_main;

/* RENDERTARGETS:0 */
layout(location = 0) out vec4 rt_out;

void main() {
	ivec2 intTexCoord = ivec2(gl_FragCoord.xy);

	vec4 centerSample = texelFetch(usam_main, intTexCoord, 0);

	vec4 sum = centerSample * 4.0;

	sum += texelFetchOffset(usam_main, intTexCoord, 0, ivec2(-1, -1));
	sum += texelFetchOffset(usam_main, intTexCoord, 0, ivec2(0, -1)) * 2.0;
	sum += texelFetchOffset(usam_main, intTexCoord, 0, ivec2(1, -1));

	sum += texelFetchOffset(usam_main, intTexCoord, 0, ivec2(-1, 0)) * 2.0;
	sum += texelFetchOffset(usam_main, intTexCoord, 0, ivec2(1, 0)) * 2.0;

	sum += texelFetchOffset(usam_main, intTexCoord, 0, ivec2(-1, 1));
	sum += texelFetchOffset(usam_main, intTexCoord, 0, ivec2(0, 1)) * 2.0;
	sum += texelFetchOffset(usam_main, intTexCoord, 0, ivec2(1, 1));

	rt_out.rgb = mix(centerSample.rgb, sum.rgb / 16.0, float(sum.a > 0.0));
	rt_out.a = float(sum.a > 0.0);
}
