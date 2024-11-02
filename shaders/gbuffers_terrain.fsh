#version 460 compatibility

uniform sampler2D lightmap;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D texture;

varying vec3 frag_viewCoord;
varying vec4 frag_colorMul;
varying vec2 frag_texCoord;
varying vec2 frag_lightMapCoord;
varying vec3 frag_worldNormal;

#include "utils/Common.glsl"

/* RENDERTARGETS:1,2,3,4,5 */
layout(location = 0) out vec4 rt_gViewCoord;
layout(location = 1) out vec4 rt_gAlbedo;
layout(location = 2) out vec4 rt_gLightMapCoord;
layout(location = 3) out vec4 rt_gNormal;
layout(location = 4) out uint rt_gMaterialID;


void main() {
	rt_gViewCoord.xyz = frag_viewCoord;
	rt_gAlbedo = texture(texture, frag_texCoord) * frag_colorMul;
	rt_gLightMapCoord.xy = frag_lightMapCoord;
	rt_gNormal.xyz = frag_worldNormal * 0.5 + 0.5;
	rt_gMaterialID = 0u;

	if (rt_gAlbedo.a < 0.5) {
		discard;
	}
}