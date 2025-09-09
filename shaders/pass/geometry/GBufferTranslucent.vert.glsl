#include "/Base.glsl"

in vec2 mc_Entity;

in vec4 at_tangent;

out vec3 vert_viewTangent;
out vec4 vert_colorMul;
out vec3 vert_viewNormal;
out vec2 vert_texCoord;
out vec2 vert_lmCoord;
out uint vert_materialID;
out vec3 vert_viewCoord;

void main() {
    gl_Position = global_taaJitterMat * ftransform();
    vec4 temp = gbufferProjectionInverse * gl_Position;
    vert_viewCoord = temp.xyz / temp.w;

    vert_viewTangent.xyz = gl_NormalMatrix * at_tangent.xyz;
    vert_viewNormal = gl_NormalMatrix * gl_Normal.xyz;
    vert_texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    vert_lmCoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    vert_colorMul = gl_Color;

    vert_materialID = uint(int(mc_Entity.x)) & 0xFFFFu;
}