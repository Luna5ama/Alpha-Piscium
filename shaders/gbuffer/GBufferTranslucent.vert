#include "/_Base.glsl"

in vec2 mc_Entity;

in vec4 at_tangent;

out vec3 frag_viewTangent;

out vec4 frag_colorMul;
out vec3 frag_viewNormal;
out vec2 frag_texCoord;
out vec2 frag_lmCoord;
out uint frag_materialID;
out vec3 frag_viewCoord;

void main() {
    gl_Position = global_taaJitterMat * ftransform();
    vec4 temp = gbufferProjectionInverse * gl_Position;
    frag_viewCoord = temp.xyz / temp.w;

    frag_viewTangent.xyz = gl_NormalMatrix * at_tangent.xyz;
    frag_viewNormal = gl_NormalMatrix * gl_Normal.xyz;
    frag_texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    frag_lmCoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    frag_colorMul = gl_Color;

    frag_materialID = uint(int(mc_Entity.x)) & 0xFFFFu;
}