#include "/Base.glsl"

in vec2 mc_Entity;

in vec4 at_tangent;

#ifdef GBUFFER_PASS_DIRECT
#define vertOut_viewTangent frag_viewTangent
#define vertOut_colorMul frag_colorMul
#define vertOut_viewNormal frag_viewNormal
#define vertOut_texCoord frag_texCoord
#define vertOut_lmCoord frag_lmCoord
#define vertOut_materialID frag_materialID
#define vertOut_viewCoord frag_viewCoord
#else
#define vertOut_viewTangent vert_viewTangent
#define vertOut_colorMul vert_colorMul
#define vertOut_viewNormal vert_viewNormal
#define vertOut_texCoord vert_texCoord
#define vertOut_lmCoord vert_lmCoord
#define vertOut_materialID vert_materialID
#define vertOut_viewCoord vert_viewCoord
#endif

out vec3 vertOut_viewTangent;
out vec4 vertOut_colorMul;
out vec3 vertOut_viewNormal;
out vec2 vertOut_texCoord;
out vec2 vertOut_lmCoord;
out uint vertOut_materialID;
out vec3 vertOut_viewCoord;

void main() {
    gl_Position = global_taaJitterMat * ftransform();
    vec4 temp = gbufferProjectionInverse * gl_Position;
    vertOut_viewCoord = temp.xyz / temp.w;

    vertOut_viewTangent.xyz = gl_NormalMatrix * at_tangent.xyz;
    vertOut_viewNormal = gl_NormalMatrix * gl_Normal.xyz;
    vertOut_texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    vertOut_lmCoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    vertOut_colorMul = gl_Color;
    #ifdef GBUFFER_PASS_HAND
    vertOut_colorMul = vec4(1.0);
    #endif

    vertOut_materialID = uint(int(mc_Entity.x)) & 0xFFFFu;
}