#include "/util/Coords.glsl"
#include "/util/Math.glsl"
#include "/util/NZPacking.glsl"

#ifndef GBUFFER_PASS_DH
in vec2 mc_Entity;
#endif
in vec4 at_tangent;
in vec4 at_midBlock;

#ifdef SETTING_TBN_PACKING
out uint frag_worldTN;
#else
out vec3 frag_worldTangent;
out vec3 frag_worldNormal;// 11 + 11 + 10 = 32 bits
#endif

out vec4 frag_colorMul;
out vec2 frag_texCoord;
out vec2 frag_lmCoord;
out uint frag_materialID;
out float frag_viewZ;
out uint frag_midBlock;
out vec3 frag_offsetToCenter;

void main() {
    gl_Position = global_taaJitterMat * ftransform();
    frag_viewZ = -gl_Position.w;

    vec3 viewNormal = gl_NormalMatrix * normalize(gl_Normal.xyz);
    vec3 viewTangent = gl_NormalMatrix * normalize(at_tangent.xyz);
    vec3 worldNormal = coords_dir_viewToWorld(viewNormal);
    vec3 worldTangent = coords_dir_viewToWorld(viewTangent);
    #ifdef SETTING_TBN_PACKING
    nzpacking_packNormalOct16(frag_worldTN, worldNormal, worldTangent);
    #else
    frag_worldTangent = worldTangent;
    frag_worldNormal = worldNormal;
    #endif

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
    frag_materialID = bitfieldInsert(frag_materialID, uint(at_tangent.w >= 0.0), 30, 1);

    vec3 offsetToCenter = at_midBlock.xyz / 64.0;
    frag_midBlock = packUnorm4x8(vec4(saturate(abs(offsetToCenter) / 2.0), at_midBlock.w / 15.0));
    frag_offsetToCenter = offsetToCenter;
}