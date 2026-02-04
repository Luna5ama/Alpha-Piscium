layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec2 vert_texcoord[];
in vec4 vert_color[];
in vec2 vert_screenPos[];

in uint vert_survived[];
in uint vert_worldNormalMaterialID[];

out vec2 frag_texcoord;
out vec4 frag_color;
out vec2 frag_screenPos;
out uint frag_worldNormalMaterialID;

out uvec2 frag_texcoordMinMax;

void main() {
    uint survived = vert_survived[0] & vert_survived[1] & vert_survived[2];
    if (bool(survived)) {
        vec2 texcoordMin = vec2(1.0);
        vec2 texcoordMax = vec2(0.0);
        texcoordMin = min(texcoordMin, vert_texcoord[0]);
        texcoordMax = max(texcoordMax, vert_texcoord[0]);
        texcoordMin = min(texcoordMin, vert_texcoord[1]);
        texcoordMax = max(texcoordMax, vert_texcoord[1]);
        texcoordMin = min(texcoordMin, vert_texcoord[2]);
        texcoordMax = max(texcoordMax, vert_texcoord[2]);
        uvec2 texcoordMinMaxPacked;
        texcoordMinMaxPacked.x = packSnorm2x16(texcoordMin);
        texcoordMinMaxPacked.y = packSnorm2x16(texcoordMax);

        gl_Position = gl_in[0].gl_Position;
        frag_texcoord = vert_texcoord[0];
        frag_color = vert_color[0];
        frag_screenPos = vert_screenPos[0];
        frag_worldNormalMaterialID = vert_worldNormalMaterialID[0];
        frag_texcoordMinMax = texcoordMinMaxPacked;
        EmitVertex();

        gl_Position = gl_in[1].gl_Position;
        frag_texcoord = vert_texcoord[1];
        frag_color = vert_color[1];
        frag_screenPos = vert_screenPos[1];
        frag_worldNormalMaterialID = vert_worldNormalMaterialID[1];
        frag_texcoordMinMax = texcoordMinMaxPacked;
        EmitVertex();

        gl_Position = gl_in[2].gl_Position;
        frag_texcoord = vert_texcoord[2];
        frag_color = vert_color[2];
        frag_screenPos = vert_screenPos[2];
        frag_worldNormalMaterialID = vert_worldNormalMaterialID[2];
        frag_texcoordMinMax = texcoordMinMaxPacked;
        EmitVertex();

        EndPrimitive();
    }
}