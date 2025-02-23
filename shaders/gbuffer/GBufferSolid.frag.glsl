#include "/util/Colors.glsl"
#include "/util/Dither.glsl"
#include "/util/Math.glsl"
#include "/util/Rand.glsl"
#include "/util/GBuffers.glsl"

uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform usampler2D usam_gbufferData;
uniform sampler2D usam_gbufferViewZ;

in vec3 frag_viewTangent;

in vec4 frag_colorMul; // 8 x 4 = 32 bits
in vec3 frag_viewNormal; // 11 + 11 + 10 = 32 bits
in vec2 frag_texCoord; // 16 x 2 = 32 bits
in vec2 frag_lmCoord; // 8 x 2 = 16 bits
flat in uint frag_materialID; // 16 x 1 = 16 bits

in float frag_viewZ; // 32 bits

#ifndef GBUFFER_PASS_ALPHA_TEST
layout(early_fragment_tests) in;
#endif

/* RENDERTARGETS:5,8,9 */
layout(location = 0) out vec4 rt_tempColor;
layout(location = 1) out uvec4 rt_gbufferData;
layout(location = 2) out float rt_gbufferViewZ;

ivec2 texelPos = ivec2(gl_FragCoord.xy);
float noiseIGN = rand_IGN(gl_FragCoord.xy, frameCounter);

vec4 processAlbedo() {
    vec4 albedo = frag_colorMul;

    #ifdef GBUFFER_PASS_TEXTURED
    albedo *= textureLod(gtexture, frag_texCoord, 0.0);
    #endif

    #ifdef GBUFFER_PASS_ENTITY_COLOR
    albedo.rgb = mix(albedo.rgb, entityColor.rgb, entityColor.a);
    #endif

    #ifdef GBUFFER_PASS_ALPHA_TEST
    if (albedo.a < 0.1) {
        discard;
    }
    #endif

    #ifdef SETTING_DEBUG_WHITE_WORLD
    return vec4(1.0);
    #else
    return albedo;
    #endif
}

#ifdef GBUFFER_PASS_ARMOR_GLINT
GBufferData processOutput() {
    GBufferData gData;

    GBufferData gDataPrev;
    gbuffer_unpack(texelFetch(usam_gbufferData, texelPos, 0), gDataPrev);
    gData.materialAO = gDataPrev.materialAO;
    gData.pbrSpecular = gDataPrev.pbrSpecular;

    gData.normal = gDataPrev.normal;
    gData.lmCoord = gDataPrev.lmCoord;
    gData.materialID = gDataPrev.materialID;

    return gData;
}
#else
GBufferData processOutput() {
    GBufferData gData;

    #if defined(GBUFFER_PASS_TEXTURED)
    vec4 normalSample = textureLod(normals, frag_texCoord, 0.0);
    vec4 specularSample = textureLod(specular, frag_texCoord, 0.0);

    gData.materialAO = normalSample.b;
    gData.pbrSpecular = specularSample;
    gData.lmCoord = frag_lmCoord;
    gData.materialID = frag_materialID;

    const float _1o255 = 1.0 / 255.0;
    float emissiveS = linearStep(1.0, _1o255, gData.pbrSpecular.a);
    emissiveS *= step(_1o255, gData.pbrSpecular.a);

    gData.pbrSpecular.a = emissiveS;

    #if !defined(SETTING_NORMAL_MAPPING) || (!defined(SETTING_NORMAL_MAPPING_HANDHELD) && defined(GBUFFER_PASS_HAND))
    gData.normal = frag_viewNormal;
    #else
    vec3 bitangent = cross(frag_viewTangent, frag_viewNormal);
    mat3 tbn = mat3(frag_viewTangent, bitangent, frag_viewNormal);
    vec3 tagentNormal;
    tagentNormal.xy = normalSample.rg * 2.0 - 1.0;
    tagentNormal.z = sqrt(saturate(1.0 - dot(tagentNormal.xy, tagentNormal.xy)));
    vec3 mappedNormal = normalize(tbn * tagentNormal);
    gData.normal = normalize(mix(frag_viewNormal, mappedNormal, SETTING_NORMAL_MAPPING_STRENGTH));
    #endif

    #else
    gData.normal = frag_viewNormal;
    gData.materialAO = 1.0;
    gData.pbrSpecular = vec4(0.0, 1.0, 0.0, 1.0);
    gData.lmCoord = frag_lmCoord;
    gData.materialID = 65534u;
    #endif

    gData.lmCoord = dither_u8(gData.lmCoord, noiseIGN);

    return gData;
}
#endif

float processViewZ() {
    #if defined(GBUFFER_PASS_VIEWZ_OVERRIDE)
    return GBUFFER_PASS_VIEWZ_OVERRIDE;
    #elif defined(GBUFFER_PASS_ARMOR_GLINT)
    return texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    #else
    return frag_viewZ;
    #endif
}

void main() {
    vec4 albedo = processAlbedo();
    float viewZ = processViewZ();
    GBufferData gData = processOutput();

    #ifdef GBUFFER_PASS_ARMOR_GLINT
    gData.materialID = 65532u;
    albedo.rgb *= albedo.rgb;

    float glintEmissive = colors_srgbLuma(albedo.rgb);
    glintEmissive *= 0.1;
    gData.pbrSpecular.a = saturate(gData.pbrSpecular.a + glintEmissive);
    #endif

    #ifdef GBUFFER_PASS_PARTICLE
    gData.materialID = 65533u;
//    float particleEmissive = colors_srgbLuma(albedo.rgb * albedo.rgb);
//    gData.pbrSpecular.a = saturate(gData.pbrSpecular.a + particleEmissive);
    #endif

    #ifdef GBUFFER_PASS_HAND
    gData.isHand = true;
    #else
    gData.isHand = false;
    #endif

    rt_tempColor = albedo;
    rt_gbufferViewZ = viewZ;
    gbuffer_pack(rt_gbufferData, gData);
}