#version 460 compatibility

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in uint vert_survived[];
in vec2 vert_unwarpedTexCoord[];
in vec2 vert_texcoord[];
in vec4 vert_color[];
in float vert_viewZ[];

out vec2 frag_unwarpedTexCoord;
out vec2 frag_texcoord;
out vec4 frag_color;
out float frag_viewZ;
out vec2 frag_texcoordMin;
out vec2 frag_texcoordMax;

void main() {
	uint survived = vert_survived[0] & vert_survived[1] & vert_survived[2];
	if (bool(survived)) {
		vec2 texcoordMin = vec2(1.0);
		vec2 texcoordMax = vec2(0.0);
		for (int i = 0; i < 3; i++) {
			texcoordMin = min(texcoordMin, vert_texcoord[i]);
			texcoordMax = max(texcoordMax, vert_texcoord[i]);
		}

		gl_Position = gl_in[0].gl_Position;
		frag_unwarpedTexCoord = vert_unwarpedTexCoord[0];
		frag_texcoord = vert_texcoord[0];
		frag_color = vert_color[0];
		frag_texcoordMin = texcoordMin;
		frag_texcoordMax = texcoordMax;
		EmitVertex();

		gl_Position = gl_in[1].gl_Position;
		frag_unwarpedTexCoord = vert_unwarpedTexCoord[1];
		frag_texcoord = vert_texcoord[1];
		frag_texcoordMin = texcoordMin;
		frag_texcoordMax = texcoordMax;
		EmitVertex();

		gl_Position = gl_in[2].gl_Position;
		frag_unwarpedTexCoord = vert_unwarpedTexCoord[2];
		frag_texcoord = vert_texcoord[2];
		frag_texcoordMin = texcoordMin;
		frag_texcoordMax = texcoordMax;
		EmitVertex();

		EndPrimitive();
	}
}