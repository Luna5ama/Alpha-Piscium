layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec4 vert_viewTangent[];
in vec4 vert_colorMul[];
in vec3 vert_viewNormal[];
in vec2 vert_texCoord[];
in vec2 vert_lmCoord[];
in uint vert_materialID[];
in float vert_viewZ[];

out vec4 frag_viewTangent;
out vec4 frag_colorMul;
out vec3 frag_viewNormal;
out vec2 frag_texCoord;
out vec2 frag_lmCoord;
out uint frag_materialID;
out float frag_viewZ;

void main() {
    vec2 pos1 = gl_in[0].gl_Position.xy / gl_in[0].gl_Position.w;
    vec2 pos2 = gl_in[1].gl_Position.xy / gl_in[1].gl_Position.w;
    vec2 pos3 = gl_in[2].gl_Position.xy / gl_in[2].gl_Position.w;
//    int flag = int(determinant(mat2(pos3 - pos1, pos2 - pos1)) > 0.0);
//    flag |= int(dot(vert_viewCoord[0], vert_viewNormal[0]) > 0.0);

//    if (bool(flag)) {
//        for (int i = 0; i < 3; i++) {
//            int j = 2 - i;
//            gl_Position = gl_in[j].gl_Position;
//            frag_viewZ = vert_viewZ[j];
//            frag_viewTangent = vert_viewTangent[j];
//            frag_viewNormal = vert_viewNormal[j];
//            frag_texCoord = vert_texCoord[j];
//            frag_lmCoord = vert_lmCoord[j];
//            frag_colorMul = vert_colorMul[j];
//            frag_materialID = vert_materialID[j];
//            EmitVertex();
//        }
//        EndPrimitive();
//    } else {
        for (int i = 0; i < 3; i++) {
            int j = i;
            gl_Position = gl_in[j].gl_Position;
            frag_viewZ = vert_viewZ[j];
            frag_viewTangent = vert_viewTangent[j];
            frag_viewNormal = vert_viewNormal[j];
            frag_texCoord = vert_texCoord[j];
            frag_lmCoord = vert_lmCoord[j];
            frag_colorMul = vert_colorMul[j];
            frag_materialID = vert_materialID[j];
            EmitVertex();
        }
        EndPrimitive();
//    }
}