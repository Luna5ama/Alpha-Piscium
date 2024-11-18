#version 460 compatibility

#include "_Util.glsl"

uniform sampler2D lightmap;
uniform sampler2D texture;

in vec2 texcoord;
in vec4 glcolor;
in float frag_viewZ;

layout(location = 0) out vec4 rt_out;

void main() {
	vec4 color = textureLod(texture, texcoord, 0.0) * glcolor;

	color.rgb = colors_srgbToLinear(color.rgb);

	vec2 randCoord = gl_FragCoord.xy;
	randCoord.y += -frag_viewZ;
	float randAlpha = rand_IGN(randCoord, frameCounter);

	if (color.a < randAlpha) {
		discard;
	}

	rt_out.rgb = color.rgb;
	rt_out.a = color.a;
	if (color.a == 1.0) {
		rt_out.rgb = vec3(1.0);
		rt_out.a = 0.0;
	}
}