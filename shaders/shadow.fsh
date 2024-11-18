#version 460 compatibility

#include "_Util.glsl"

uniform sampler2D lightmap;
uniform sampler2D texture;

varying vec2 texcoord;
varying vec4 glcolor;

layout(location = 0) out vec4 rt_out;

void main() {
	vec4 color = textureLod(texture, texcoord, 0.0) * glcolor;

	color.rgb = colors_srgbToLinear(color.rgb);

	uint r2Index = uint(rand_IGN(gl_FragCoord.xy, frameCounter) * 1024.0);
	r2Index += (rand_hash11(floatBitsToUint(gl_FragCoord.z)) & 65535u);
	r2Index += frameCounter;
	float randZ = rand_r2Seq1(r2Index);

	if (color.a < randZ) {
		discard;
	}

	rt_out.rgb = color.rgb;
	rt_out.a = color.a;
	if (color.a == 1.0) {
		rt_out.rgb = vec3(1.0);
		rt_out.a = 0.0;
	}
}