#version 460 compatibility

uniform sampler2D lightmap;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D texture;

varying vec3 frag_viewCoord;
varying vec3 frag_worldNormal;
varying vec2 frag_texCoord;
varying vec2 frag_lightMapCoord;
varying vec4 frag_colorMul;

#include "/utils/Settings.glsl"
#include "/utils/PackedGBuffers.glsl"
#include "/utils/Uniforms.glsl"

/* DRAWBUFFERS:12 */
layout(location = 0) out uvec4 rt_gData1;
layout(location = 1) out uvec4 rt_gData2;

void main() {
	PackedGBufferData data;

	data.specularParams = uvec4(0.0, 0.0, 0.0, 0.0);
	data.materialID = 0u;
	data.worldNormal = frag_worldNormal;

	data.viewCoord = frag_viewCoord;
	data.albedo = texture(texture, frag_texCoord) * frag_colorMul;
	data.lightMapCoord = frag_lightMapCoord;

	if (data.albedo.a < 0.5) {
		discard;
	}

	pgbuffer_pack(rt_gData1, rt_gData2, data);
}