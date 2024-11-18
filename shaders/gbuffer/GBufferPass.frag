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

    #ifdef GBUFFER_PASS_TEXTURED
    albedoTemp *= texture(gtexture, frag_texCoord);
    #endif

    #ifdef GBUFFER_PASS_ALPHA_TEST
    if (albedoTemp.a < 0.1) {
        discard;
    }
    #endif

    #ifdef GBUFFER_PASS_TRANLUCENT
    uint r2Index = 0u;
    r2Index += uint(rand_IGN(gl_FragCoord.xy, frameCounter) * 64.0);
    r2Index += (rand_hash11(floatBitsToUint(gl_FragCoord.z)) & 255u);
    float randAlpha = rand_r2Seq1(r2Index);

//    float randAlpha = rand_IGN(gl_FragCoord.xy, frameCounter);

    if (albedoTemp.a < randAlpha) {
        discard;
    }
    #endif

    gData.albedo = albedoTemp.rgb;

    #if defined(GBUFFER_PASS_TEXTURED) && defined(MC_TEXTURE_FORMAT_LAB_PBR)
    vec4 normalSample = textureLod(normals, frag_texCoord, 0.0);
    vec4 specularSample = textureLod(specular, frag_texCoord, 0.0);

    gData.materialAO = normalSample.b;
    gData.pbrSpecular = specularSample;

    // TODO: normal map
    gData.normal = frag_viewNormal;

    #else
    // TODO: hardcoded PBR
    gData.materialAO = 1.0;
    gData.pbrSpecular = vec4(0.0, 1.0, 0.0, 0.0);

    gData.normal = frag_viewNormal;
    #endif

    gData.lmCoord = frag_lmCoord;
    gData.materialID = frag_materialID;

    gbuffer_pack(rt_gbuffer, gData);

    #ifdef GBUFFER_PASS_VIEWZ_OVERRIDE
    rt_viewZ = GBUFFER_PASS_VIEWZ_OVERRIDE;
    #else
    rt_viewZ = frag_viewZ;
    #endif
}