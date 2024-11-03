#include "../_Util.glsl"

#ifdef GBUFFER_PASS_TEXTURED
uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;
#endif

in vec4 frag_colorMul; // 8 x 4 = 32 bits
in vec3 frag_viewNormal; // 11 + 11 + 10 = 32 bits
in vec2 frag_texCoord; // 16 x 2 = 32 bits
in vec2 frag_lmCoord; // 8 x 2 = 16 bits
flat in uint frag_materialID; // 16 x 1 = 16 bits

in float frag_viewZ; // 32 bits

/* RENDERTARGETS:1,2 */
layout(location = 0) out uvec4 rt_gbuffer;
layout(location = 1) out float rt_viewZ;

void main() {
    vec4 albedoTemp = frag_colorMul;

    GBufferData gData;

    gData.roughness = 0.0;
    #ifdef GBUFFER_PASS_TEXTURED
    albedoTemp *= texture(gtexture, frag_texCoord);
    #endif
    gData.albedo = albedoTemp.rgb;

    gData.f0 = 0.0;
    gData.emissive = 0.0;
    gData.porositySSS = 0.0;
    // TODO: hardcoded PBR + LABPBR

    gData.normal = frag_viewNormal;
    // TODO: normal map

    gData.lmCoord = frag_lmCoord;
    gData.materialID = frag_materialID;

    gbuffer_pack(rt_gbuffer, gData);
    rt_viewZ = frag_viewZ;

    if (albedoTemp.a < 0.5) {
        discard;
    }
}