#include "/util/Math.glsl"

#ifndef GBUFFER_PASS_DH
in vec2 mc_Entity;
#endif
in vec4 at_tangent;
in vec4 at_midBlock;

out vec4 frag_viewTangent;
out vec4 frag_colorMul;
out vec3 frag_viewNormal;
out vec2 frag_texCoord;
out vec2 frag_lmCoord;
out uint frag_materialID;
out float frag_viewZ;
out uint frag_midBlock;
out vec3 frag_offsetToCenter;

void main() {
    gl_Position = global_taaJitterMat * ftransform();
    frag_viewZ = -gl_Position.w;

    frag_viewNormal = gl_NormalMatrix * normalize(gl_Normal.xyz);
    frag_viewTangent.xyz = gl_NormalMatrix * normalize(at_tangent.xyz);
    frag_viewTangent.w = sign(at_tangent.w);

    frag_texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    frag_lmCoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    frag_colorMul = gl_Color;
    #ifdef GBUFFER_PASS_HAND
    frag_colorMul = vec4(1.0);
    #endif

    #ifdef GBUFFER_PASS_DH
    frag_materialID = 65533u;
    if (dhMaterialId == DH_BLOCK_WATER) {
        frag_materialID = 3u;
    }
    #else
    frag_materialID = uint(int(mc_Entity.x)) & 0xFFFFu;
    #endif

    vec3 offsetToCenter = at_midBlock.xyz / 64.0;
    frag_midBlock = packUnorm4x8(vec4(saturate(abs(offsetToCenter) / 2.0), at_midBlock.w / 15.0));
    frag_offsetToCenter = offsetToCenter;
}