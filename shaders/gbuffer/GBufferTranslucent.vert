#include "../_Util.glsl"

in vec2 mc_Entity;

in vec4 at_tangent;

out vec4 frag_viewTangent;

out vec4 frag_colorMul; // 8 x 4 = 32 bits
out vec3 frag_viewNormal; // 11 + 11 + 10 = 32 bits
out vec2 frag_texCoord; // 16 x 2 = 32 bits
out vec2 frag_lmCoord; // 8 x 2 = 16 bits
out uint frag_materialID; // 16 x 1 = 16 bits

out float frag_viewZ;

void main() {
    gl_Position = global_taaJitterMat * ftransform();
    frag_viewZ = -gl_Position.w;

    frag_viewTangent.xyz = gl_NormalMatrix * at_tangent.xyz;
    frag_viewNormal = gl_NormalMatrix * gl_Normal.xyz;
    frag_texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    frag_lmCoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    frag_colorMul = gl_Color;

    frag_materialID = uint(int(mc_Entity.x)) & 0xFFFFu;
    frag_viewTangent.w = float(mc_Entity == 65534u);
}