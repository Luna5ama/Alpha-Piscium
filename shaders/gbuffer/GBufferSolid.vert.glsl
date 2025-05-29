#include "/util/Math.glsl"

#ifdef GBUFFER_PASS_MATERIAL_ID
in vec2 mc_Entity;
#endif

in vec4 at_tangent;

out vec3 frag_viewTangent;

out vec4 frag_colorMul; // 8 x 4 = 32 bits
out vec3 frag_viewNormal; // 11 + 11 + 10 = 32 bits
out vec2 frag_texCoord; // 16 x 2 = 32 bits
out vec2 frag_lmCoord; // 8 x 2 = 16 bits
out uint frag_materialID; // 16 x 1 = 16 bits

out float frag_viewZ;

void main() {
    gl_Position = global_taaJitterMat * ftransform();
    frag_viewZ = -gl_Position.w;

    frag_viewTangent = gl_NormalMatrix * at_tangent.xyz;
    frag_viewNormal = gl_NormalMatrix * gl_Normal.xyz;
    frag_texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    frag_lmCoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    frag_lmCoord.x = linearStep(0.0625, 0.96875, frag_lmCoord.x);
    frag_lmCoord.y = linearStep(0.125, 0.73438, frag_lmCoord.y);
    frag_colorMul = gl_Color;

    #ifdef GBUFFER_PASS_MATERIAL_ID
    frag_materialID = uint(int(mc_Entity.x)) & 0xFFFFu;
    #else
    frag_materialID = 65535u;
    #endif
}