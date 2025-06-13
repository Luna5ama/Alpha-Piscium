#include "/util/Colors.glsl"
#include "/util/Dither.glsl"
#include "/util/Math.glsl"
#include "/util/Rand.glsl"
#include "/util/GBufferData.glsl"

uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferViewZ;

in vec3 frag_viewTangent;

in vec4 frag_colorMul;// 8 x 4 = 32 bits
in vec3 frag_viewNormal;// 11 + 11 + 10 = 32 bits
in vec2 frag_texCoord;// 16 x 2 = 32 bits
in vec2 frag_lmCoord;// 8 x 2 = 16 bits
flat in uint frag_materialID;// 16 x 1 = 16 bits

in float frag_viewZ;// 32 bits

#ifndef GBUFFER_PASS_ALPHA_TEST
layout(early_fragment_tests) in;
#endif

#ifdef GBUFFER_PASS_NO_LIGHTING
/* RENDERTARGETS:6,10 */
layout(location = 0) out vec4 rt_color;
layout(location = 1) out float rt_gbufferViewZ;
#else
/* RENDERTARGETS:8,9,10 */
layout(location = 0) out uvec4 rt_gbufferData32UI;
layout(location = 1) out vec4 rt_gbufferData8UN;
layout(location = 2) out float rt_gbufferViewZ;
#endif

vec2 dUVdx = dFdx(frag_texCoord);
vec2 dUVdy = dFdy(frag_texCoord);

ivec2 texelPos = ivec2(gl_FragCoord.xy);
float noiseIGN = rand_IGN(texelPos, frameCounter);

vec4 albedo;
float viewZ;

GBufferData gData = gbufferData_init();

void processAlbedo() {
    albedo = frag_colorMul;

    #ifdef GBUFFER_PASS_TEXTURED
    vec4 sample1 = textureGrad(gtexture, frag_texCoord, dUVdx * 0.5, dUVdy * 0.5);
    vec4 sample2 = textureGrad(gtexture, frag_texCoord, dUVdx * 0.25, dUVdy * 0.25);
    albedo *= vec4(sample2.rgb, sample1.a);
    #endif

    #ifdef GBUFFER_PASS_ENTITY_COLOR
    albedo.rgb = mix(albedo.rgb, entityColor.rgb, entityColor.a);
    #endif

    #ifdef GBUFFER_PASS_ALPHA_TEST
    if (albedo.a < 0.1) {
        discard;
    }
    #endif

    #ifdef GBUFFER_PASS_ARMOR_GLINT
    albedo.rgb *= albedo.rgb;
    #endif

    #ifdef SETTING_DEBUG_WHITE_WORLD
    albedo.rgb = vec3(1.0);
    #endif
}

void processViewZ() {
    #if defined(GBUFFER_PASS_VIEWZ_OVERRIDE)
    viewZ = GBUFFER_PASS_VIEWZ_OVERRIDE;
    #elif defined(GBUFFER_PASS_ARMOR_GLINT)
    viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    #else
    viewZ = frag_viewZ;
    #endif
}

void processData2() {
    gData.albedo = albedo.rgb;
    #ifdef GBUFFER_PASS_HAND
    gData.isHand = true;
    #else
    gData.isHand = false;
    #endif
}

#ifdef GBUFFER_PASS_ARMOR_GLINT
void processData1() {
    GBufferData gDataPrev;
    gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos, 0), gDataPrev);
    gData.pbrSpecular = gDataPrev.pbrSpecular;

    gData.geometryNormal = gDataPrev.geometryNormal;
    gData.normal = gDataPrev.normal;
    gData.lmCoord = gDataPrev.lmCoord;
    gData.materialID = gDataPrev.materialID;

    float glintEmissive = colors_sRGB_luma(albedo.rgb);
    glintEmissive *= 0.1;
    gData.pbrSpecular.a = saturate(gData.pbrSpecular.a + glintEmissive);
}
#else
void processData1() {
    #if defined(GBUFFER_PASS_TEXTURED)
    vec4 normalSample = textureGrad(normals, frag_texCoord, dUVdx, dUVdy);
    vec4 specularSample = textureGrad(specular, frag_texCoord, dUVdx, dUVdy);

    gData.pbrSpecular = specularSample;
    gData.lmCoord = frag_lmCoord;
    gData.lmCoord.y *= normalSample.b;
    gData.materialID = frag_materialID;

    float emissiveS = specularSample.a;
    emissiveS *= float(specularSample.a < 1.0);

    gData.pbrSpecular.a = emissiveS;

    gData.geometryNormal = frag_viewNormal;
    #if !defined(SETTING_NORMAL_MAPPING)
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
    gData.pbrSpecular = vec4(0.0, 1.0, 0.0, 1.0);
    gData.lmCoord = frag_lmCoord;
    gData.materialID = 65534u;
    #endif

    gData.lmCoord = dither_u8(gData.lmCoord, noiseIGN);

    #ifdef GBUFFER_PASS_PARTICLE
    gData.materialID = 65533u;
    if (textureQueryLevels(gtexture) == 1) {
        float particleEmissive = pow2(colors_sRGB_luma(albedo.rgb * albedo.rgb));
        gData.pbrSpecular.a = saturate(gData.pbrSpecular.a + particleEmissive);
    }
    #endif
}
#endif

void main() {
    processAlbedo();
    processViewZ();

    #ifdef GBUFFER_PASS_NO_LIGHTING
    rt_color = albedo;
    rt_gbufferViewZ = viewZ;
    #else
    processData1();
    processData2();

    gbufferData1_pack(rt_gbufferData32UI, gData);
    gbufferData2_pack(rt_gbufferData8UN, gData);
    rt_gbufferViewZ = viewZ;
    #endif
}