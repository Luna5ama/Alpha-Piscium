#include "/util/Math.glsl"
#include "/util/Coords.glsl"
#include "/util/NZPacking.glsl"

#ifdef GBUFFER_PASS_MATERIAL_ID
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

out vec3 frag_colorMul; // 8 x 4 = 32 bits
out vec2 frag_texCoord; // 16 x 2 = 32 bits
out vec2 frag_lmCoord; // 8 x 2 = 16 bits
out uint frag_materialID; // 16 x 1 = 16 bits
out float frag_emissiveOverride;

void main() {
    gl_Position = global_taaJitterMat * ftransform();

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
    frag_lmCoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    frag_lmCoord.x = linearStep(0.0625, 0.96875, frag_lmCoord.x);
    frag_lmCoord.y = linearStep(0.125, 0.73438, frag_lmCoord.y);
    frag_colorMul = gl_Color.rgb;
    #ifdef GBUFFER_PASS_DH
    frag_emissiveOverride = 0.0;
    if (dhMaterialId == DH_BLOCK_LAVA) {
        frag_emissiveOverride = 1.0;
    }
    if (dhMaterialId == DH_BLOCK_ILLUMINATED) {
        frag_emissiveOverride = 0.8;
    }
    #else
    frag_emissiveOverride = at_midBlock.w;
    #endif

    #ifdef GBUFFER_PASS_MATERIAL_ID
    frag_materialID = uint(int(mc_Entity.x)) & 0xFFFFu;
    #else
    frag_materialID = 65535u;
    #endif

    frag_materialID = bitfieldInsert(frag_materialID, uint(at_tangent.w >= 0.0), 30, 1);
}